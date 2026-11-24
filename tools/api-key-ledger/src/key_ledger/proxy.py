"""Loopback-only HTTP gateway and read-only dashboard routes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.cookiejar import CookieJar, DefaultCookiePolicy
from importlib.resources import files
from typing import Any

import httpx
from anyio import CancelScope
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Config, Credential, Dialect, Pool
from .ledger import GROUP_COLUMNS, REQUEST_SECONDS, Attempt, Ledger
from .usage import SSEUsage, Usage

_LOCAL_HOST = re.compile(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?", re.ASCII | re.I)
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,159}", re.ASCII)
_MODEL_ID = re.compile(r"[\w][\w./:@+\-]{0,255}")
_ANTHROPIC_HEADER = re.compile(r"[A-Za-z0-9,_. \-]{1,1024}", re.ASCII)
_MAX_REQUEST = 8 * 1024 * 1024
_MAX_JSON_RESPONSE = 16 * 1024 * 1024
_FAILOVER_AUTH = {401, 402, 403}
_SECURITY_HEADERS = (
    (b"cache-control", b"no-store"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (
        b"content-security-policy",
        (
            b"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            b"base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        ),
    ),
)


def _error(status: int, code: str, message: str, **detail: Any) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": code, "code": code, "message": message, **detail}},
        status_code=status,
    )


class LocalOnly:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        hosts = request.headers.getlist("host")
        origins = request.headers.getlist("origin")

        async def guarded_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *_SECURITY_HEADERS]
            await send(message)

        peer = scope.get("client")
        if peer is None or peer[0] not in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}:
            response = _error(403, "remote_client", "Only local clients can access the ledger.")
        elif len(hosts) != 1 or not _LOCAL_HOST.fullmatch(hosts[0]):
            response = _error(400, "invalid_host", "Use a loopback address to access the ledger.")
        elif (origins and origins != [f"http://{hosts[0]}"]) or request.headers.get(
            "sec-fetch-site"
        ) == "cross-site":
            response = _error(403, "cross_origin", "Cross-origin access is disabled.")
        else:
            await self.app(scope, receive, guarded_send)
            return
        await response(scope, receive, guarded_send)


class _NoCookies(DefaultCookiePolicy):
    def set_ok(self, cookie: Any, request: Any) -> bool:
        return False


class _BodyTooLarge(Exception):
    pass


class _StreamInterrupted(RuntimeError):
    pass


async def _bounded_body(chunks: AsyncIterator[bytes], limit: int) -> bytes:
    body = bytearray()
    async for chunk in chunks:
        if len(body) + len(chunk) > limit:
            raise _BodyTooLarge
        body.extend(chunk)
    return bytes(body)


def _cooldown(value: str | None, default: int) -> float:
    if value is None:
        return float(default)
    value = value.strip()
    if re.fullmatch(r"[0-9]{1,10}", value):
        return float(int(value))
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0, (retry_at - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return float(default)


def parse_since(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)).timestamp()


def _safe_label(value: str | None, secrets: set[str]) -> str | None:
    if value is None or not _LABEL.fullmatch(value) or any(secret in value for secret in secrets):
        return None
    return value


def _safe_model(value: object, secrets: set[str]) -> str | None:
    if (
        not isinstance(value, str)
        or not _MODEL_ID.fullmatch(value)
        or any(secret in value for secret in secrets)
    ):
        return None
    return value


def _upstream_headers(
    request: Request, pool: Pool, credential: Credential, streaming: bool
) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "accept": "text/event-stream" if streaming else "application/json",
        "accept-encoding": "identity",
    }
    if pool.dialect == "openai":
        headers["authorization"] = f"Bearer {credential.value}"
    else:
        headers["x-api-key"] = credential.value
        headers["anthropic-version"] = "2023-06-01"
        for name in ("anthropic-version", "anthropic-beta"):
            value = request.headers.get(name, "")
            if _ANTHROPIC_HEADER.fullmatch(value):
                headers[name] = value
    return headers


async def _close(response: httpx.Response) -> None:
    with CancelScope(shield=True), suppress(httpx.HTTPError):
        await response.aclose()


class _LedgerStream(StreamingResponse):
    def __init__(
        self,
        *,
        upstream: httpx.Response,
        ledger: Ledger,
        pool: Pool,
        attempt: Attempt,
        usage: Usage,
        secrets: set[str],
        deadline: float,
    ) -> None:
        self.upstream = upstream
        self.ledger = ledger
        self.pool = pool
        self.attempt = attempt
        self.secrets = secrets
        self.deadline = deadline
        self.parser = SSEUsage(pool.dialect, usage)
        self.outcome = "partial"
        super().__init__(
            self._relay(), status_code=upstream.status_code, media_type="text/event-stream"
        )

    async def _relay(self) -> AsyncIterator[bytes]:
        async for chunk in self.upstream.aiter_bytes():
            self.parser.feed(chunk)
            yield chunk
        self.parser.finish()
        if self.parser.terminal and not self.parser.errored:
            self.outcome = "success"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with asyncio.timeout_at(self.deadline):
                await super().__call__(scope, receive, send)
        except (httpx.HTTPError, TimeoutError):
            # On a stream failure, closing the connection preserves the failure signal
            # for SDKs without starting a second spend-bearing upstream attempt.
            raise _StreamInterrupted("upstream stream interrupted") from None
        finally:
            usage = self.parser.usage
            usage.model = _safe_model(usage.model, self.secrets)
            try:
                self.ledger.finish(
                    self.attempt,
                    usage=usage,
                    price=self.pool.pricing.get(usage.model or ""),
                    outcome=self.outcome,
                    http_status=self.upstream.status_code,
                )
            finally:
                await _close(self.upstream)


class Gateway:
    def __init__(self, config: Config, credentials: Mapping[tuple[str, str], Credential]) -> None:
        self.config = config
        self.ledger = Ledger(config, credentials)
        self.secrets = {credential.value for credential in credentials.values()}

    async def proxy(self, request: Request, dialect: Dialect) -> Response:
        pool_id = request.path_params.get("pool_id") or self.config.default_pool(dialect)
        pool = self.config.pools.get(pool_id) if pool_id else None
        if pool is None or pool.dialect != dialect or pool_id is None:
            return _error(404, "pool_not_found", "A matching pool is required for this endpoint.")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            return _error(415, "json_required", "Send an application/json request.")
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            return _error(415, "encoding_unsupported", "Send an uncompressed JSON request.")
        try:
            async with asyncio.timeout(30):
                raw = await _bounded_body(request.stream(), _MAX_REQUEST)
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError
            streaming = data.get("stream") is True
            if (
                streaming
                and dialect == "openai"
                and pool.inject_stream_usage
                and "stream_options" not in data
            ):
                data["stream_options"] = {"include_usage": True}
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(body) > _MAX_REQUEST:
                raise _BodyTooLarge
        except _BodyTooLarge:
            return _error(413, "request_too_large", "The JSON request exceeds the size limit.")
        except TimeoutError:
            return _error(408, "request_timeout", "The JSON request took too long to arrive.")
        except (ValueError, RecursionError, ClientDisconnect):
            return _error(400, "invalid_json", "The request must contain a JSON object.")

        secrets = set(self.secrets)
        for name in ("authorization", "x-api-key", "proxy-authorization"):
            for header in request.headers.getlist(name):
                parts = header.strip().split(maxsplit=1)
                if parts:
                    secrets.add(parts[-1])
        sticky = request.headers.get("x-ledger-sticky-key", "").strip()
        if len(sticky) > 256:
            return _error(400, "invalid_sticky_key", "The sticky scope exceeds 256 characters.")
        sticky_key = hashlib.sha256(sticky.encode()).hexdigest() if sticky else None
        tag = _safe_label(request.headers.get("x-ledger-tag"), secrets)
        model = _safe_model(data.get("model"), secrets)
        tried: set[str] = set()
        client: httpx.AsyncClient = request.app.state.client

        while True:
            attempt, health = self.ledger.reserve(
                pool_id, model=model, sticky_key=sticky_key, tag=tag, tried=tried
            )
            if attempt is None:
                return _error(
                    503,
                    "pool_exhausted",
                    "No key slot is currently eligible for this request.",
                    pool_id=pool_id,
                    health=health,
                )
            tried.add(attempt.key_alias)
            result = await self._send(
                client,
                request,
                pool,
                attempt,
                body,
                streaming=streaming,
                model=model,
                secrets=secrets,
            )
            if result is not None:
                return result

    async def _send(
        self,
        client: httpx.AsyncClient,
        request: Request,
        pool: Pool,
        attempt: Attempt,
        body: bytes,
        *,
        streaming: bool,
        model: str | None,
        secrets: set[str],
    ) -> Response | None:
        usage = Usage(model)
        upstream: httpx.Response | None = None
        relay = False
        deadline = asyncio.get_running_loop().time() + REQUEST_SECONDS

        def finish(outcome: str, cooldown: float = 0) -> None:
            usage.model = _safe_model(usage.model, secrets)
            self.ledger.finish(
                attempt,
                usage=usage,
                price=pool.pricing.get(usage.model or ""),
                outcome=outcome,
                http_status=upstream.status_code if upstream else None,
                cooldown=cooldown,
            )

        try:
            async with asyncio.timeout_at(deadline):
                outbound = httpx.Request(
                    "POST",
                    pool.endpoint,
                    content=body,
                    headers=_upstream_headers(request, pool, attempt.credential, streaming),
                )
                upstream = await client.send(outbound, stream=True)
                status = upstream.status_code
                if status == 429:
                    finish(
                        "rate_limited",
                        _cooldown(upstream.headers.get("retry-after"), pool.cooldown_seconds),
                    )
                    return None
                if status in _FAILOVER_AUTH:
                    finish("quarantined")
                    return None
                if not 200 <= status < 300:
                    finish("upstream_error")
                    return _error(
                        status if status >= 400 else 502,
                        "upstream_error",
                        "The upstream rejected the request.",
                        upstream_status=status,
                    )
                if upstream.headers.get("content-encoding", "identity").lower() != "identity":
                    finish("upstream_error")
                    return _error(
                        502, "upstream_encoding", "The upstream sent unsupported encoding."
                    )
                media_type = (
                    upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                if media_type == "text/event-stream":
                    response = _LedgerStream(
                        upstream=upstream,
                        ledger=self.ledger,
                        pool=pool,
                        attempt=attempt,
                        usage=usage,
                        secrets=secrets,
                        deadline=deadline,
                    )
                    relay = True
                    return response
                payload = await _bounded_body(upstream.aiter_bytes(), _MAX_JSON_RESPONSE)
                result = json.loads(payload)
                if not isinstance(result, dict) or result.get("error"):
                    raise ValueError
                usage.update(pool.dialect, result)
                finish("success")
                return Response(payload, status_code=status, media_type="application/json")
        except asyncio.CancelledError:
            finish("interrupted")
            raise
        except (httpx.HTTPError, TimeoutError, ValueError, RecursionError, _BodyTooLarge):
            finish("upstream_error")
            return _error(502, "upstream_error", "A valid upstream response was not received.")
        finally:
            if upstream is not None and not relay:
                await _close(upstream)


def create_app(
    config: Config,
    credentials: Mapping[tuple[str, str], Credential],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    gateway = Gateway(config, credentials)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            cookies=CookieJar(policy=_NoCookies()),
            timeout=httpx.Timeout(90, connect=10, pool=10),
        ) as client:
            app.state.client = client
            yield

    async def openai(request: Request) -> Response:
        return await gateway.proxy(request, "openai")

    async def anthropic(request: Request) -> Response:
        return await gateway.proxy(request, "anthropic")

    def health(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    def keys(request: Request) -> JSONResponse:
        return JSONResponse({"keys": gateway.ledger.key_statuses()})

    def events(request: Request) -> JSONResponse:
        raw = request.query_params.get("limit", "50")
        if not re.fullmatch(r"[0-9]{1,3}", raw) or not 1 <= int(raw) <= 200:
            return _error(400, "invalid_limit", "Limit must be between 1 and 200.")
        return JSONResponse({"events": gateway.ledger.events(int(raw))})

    def stats(request: Request) -> JSONResponse:
        raw_since = request.query_params.get("since")
        by = request.query_params.get("by", "model")
        if by not in GROUP_COLUMNS:
            return _error(400, "invalid_group", "Unknown summary grouping.")
        try:
            since = parse_since(raw_since) if raw_since else 0
        except (ValueError, OverflowError, OSError):
            return _error(400, "invalid_since", "Use an ISO date or timestamp.")
        return JSONResponse({"by": by, "rows": gateway.ledger.summary(by, since)})

    async def storage_error(request: Request, exc: Exception) -> JSONResponse:
        return _error(503, "ledger_unavailable", "Local ledger storage is unavailable.")

    def asset(name: str, media_type: str) -> Callable[[Request], Response]:
        content = files("key_ledger").joinpath("dashboard", name).read_bytes()

        def serve_asset(request: Request) -> Response:
            return Response(content, media_type=media_type)

        return serve_asset

    return Starlette(
        lifespan=lifespan,
        middleware=[Middleware(LocalOnly)],
        exception_handlers={sqlite3.Error: storage_error},
        routes=[
            Route("/v1/chat/completions", openai, methods=["POST"]),
            Route("/p/{pool_id}/v1/chat/completions", openai, methods=["POST"]),
            Route("/v1/messages", anthropic, methods=["POST"]),
            Route("/p/{pool_id}/v1/messages", anthropic, methods=["POST"]),
            # The Anthropic SDK adds /v1/messages to its base URL, including /v1 bases.
            Route("/v1/v1/messages", anthropic, methods=["POST"]),
            Route("/p/{pool_id}/v1/v1/messages", anthropic, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/api/keys", keys, methods=["GET"]),
            Route("/api/events", events, methods=["GET"]),
            Route("/api/stats", stats, methods=["GET"]),
            Route("/dashboard", asset("index.html", "text/html"), methods=["GET"]),
            Route("/dashboard.css", asset("dashboard.css", "text/css"), methods=["GET"]),
            Route("/dashboard.js", asset("dashboard.js", "text/javascript"), methods=["GET"]),
        ],
    )
