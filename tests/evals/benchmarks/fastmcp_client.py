"""Minimal MCP streamable-HTTP client for the SWE-Interact ``ask_user`` sidecar.

Only the JSON-RPC surface used by the Harbor adapter is implemented: one
``initialize`` handshake (followed by ``notifications/initialized``) and then
``tools/call``. Responses may be plain ``application/json`` or
``text/event-stream``; the ``Mcp-Session-Id`` header returned by ``initialize``
is echoed on every follow-up request, as the MCP streamable-HTTP transport
requires. Everything here is stdlib-only so the same module can run inside a
task container.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROTOCOL_VERSION = "2025-03-26"
_FALLBACK_PROTOCOL_VERSIONS = ("2025-06-18", "2024-11-05")
_CLIENT_NAME = "olympus-interact"
_CLIENT_VERSION = "0.1.0"


class McpError(RuntimeError):
    """Transport, protocol, or JSON-RPC failure while talking to an MCP server."""


class McpToolError(McpError):
    """The MCP server reported ``isError: true`` for a tool call."""

    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        self.message = message
        super().__init__(f"MCP tool {tool!r} failed: {message}")


def parse_sse_messages(text: str) -> list[dict[str, Any]]:
    """Extract JSON-RPC payloads from an SSE response body.

    ``data:`` lines are joined per event (SSE allows multi-line data); comments
    and non-JSON payloads such as ``[DONE]`` are ignored.
    """
    messages: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        if not payload.strip() or payload.strip() == "[DONE]":
            return
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return
        if isinstance(decoded, dict):
            messages.append(decoded)

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        name, separator, value = line.partition(":")
        if not separator or name != "data":
            continue
        if value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    flush()
    return messages


def extract_tool_text(tool: str, result: Any) -> str:
    """Flatten an MCP ``tools/call`` result into the text the model should see."""
    if not isinstance(result, dict):
        raise McpError(f"MCP tool {tool!r} returned a non-object result")
    texts: list[str] = []
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                texts.append(item["text"])
    text = "\n".join(texts)
    if result.get("isError"):
        raise McpToolError(tool, text or "tool reported an error")
    if not text and result.get("structuredContent") is not None:
        return json.dumps(result["structuredContent"], ensure_ascii=False)
    return text


@dataclass
class McpHttpClient:
    """A minimal MCP streamable-HTTP client scoped to one ``ask_user`` tool."""

    url: str
    timeout: float = 60.0
    protocol_versions: tuple[str, ...] = (
        DEFAULT_PROTOCOL_VERSION,
        *_FALLBACK_PROTOCOL_VERSIONS,
    )
    session_id: str | None = None
    protocol_version: str | None = None
    _next_id: int = field(default=0, init=False)

    def initialize(self) -> dict[str, Any]:
        """Perform the ``initialize`` handshake once and notify the server."""
        if self.protocol_version is not None:
            return {"sessionId": self.session_id, "protocolVersion": self.protocol_version}
        errors: list[str] = []
        for version in self.protocol_versions:
            message = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": _CLIENT_NAME, "version": _CLIENT_VERSION},
                },
            }
            self.protocol_version = version
            self.session_id = None
            try:
                response = self._post(message)
            except McpError as exc:
                errors.append(f"{version}: {exc}")
                self.protocol_version = None
                continue
            if response is None:
                errors.append(f"{version}: empty initialize response")
                self.protocol_version = None
                continue
            error = response.get("error")
            if error:
                errors.append(f"{version}: {error}")
                self.protocol_version = None
                continue
            self._notify("notifications/initialized")
            result = response.get("result")
            return result if isinstance(result, dict) else {}
        raise McpError(
            "MCP initialize failed for every protocol version "
            f"({', '.join(self.protocol_versions)}): {'; '.join(errors)}"
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool descriptors."""
        result = self._rpc("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        return [tool for tool in tools if isinstance(tool, dict)] if tools else []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one MCP tool and return its flattened text result."""
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return extract_tool_text(name, result)

    def ask_user(self, question: str, *, tool: str = "ask_user") -> str:
        """Invoke the sidecar's ``ask_user(question)`` tool."""
        return self.call_tool(tool, {"question": question})

    def close(self) -> None:
        """Forget the negotiated session; servers clean up idempotently."""
        self.session_id = None
        self.protocol_version = None

    def __enter__(self) -> McpHttpClient:
        self.initialize()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        if self.protocol_version is None:
            self.initialize()
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": method,
                "params": params,
            }
        )
        if response is None:
            raise McpError(f"MCP {method!r} returned no response")
        error = response.get("error")
        if error:
            raise McpError(f"MCP {method!r} failed: {error}")
        return response.get("result")

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        self._post(message)

    def _post(self, message: dict[str, Any]) -> dict[str, Any] | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": f"{_CLIENT_NAME}/{_CLIENT_VERSION}",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
            if self.protocol_version:
                headers["MCP-Protocol-Version"] = self.protocol_version
        request = urllib.request.Request(
            self.url,
            data=json.dumps(message).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self._capture_session(response.headers.get("mcp-session-id"))
                raw = response.read()
                content_type = (response.headers.get("content-type") or "").lower()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:2000]
            raise McpError(f"MCP HTTP {exc.code} from {self.url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise McpError(f"MCP connection to {self.url} failed: {exc.reason}") from exc
        except OSError as exc:
            raise McpError(f"MCP connection to {self.url} failed: {exc}") from exc
        return self._decode_body(raw, content_type, message.get("id"))

    def _capture_session(self, header_value: str | None) -> None:
        if header_value and header_value.strip():
            self.session_id = header_value.strip()

    @staticmethod
    def _decode_body(
        raw: bytes, content_type: str, request_id: Any
    ) -> dict[str, Any] | None:
        if not raw:
            return None
        text = raw.decode("utf-8", "replace")
        if "text/event-stream" in content_type:
            return McpHttpClient._select_sse_payload(text, request_id)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            # Some proxies mislabel SSE responses; try the SSE parser as a fallback.
            return McpHttpClient._select_sse_payload(text, request_id)
        if isinstance(decoded, list):
            decoded = decoded[-1] if decoded else None
        return decoded if isinstance(decoded, dict) else None

    @staticmethod
    def _select_sse_payload(text: str, request_id: Any) -> dict[str, Any] | None:
        messages = parse_sse_messages(text)
        if not messages:
            return None
        if request_id is None:
            return messages[-1]
        for payload in messages:
            if payload.get("id") == request_id:
                return payload
        for payload in reversed(messages):
            if "result" in payload or "error" in payload:
                return payload
        return None


__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "McpError",
    "McpHttpClient",
    "McpToolError",
    "extract_tool_text",
    "parse_sse_messages",
]
