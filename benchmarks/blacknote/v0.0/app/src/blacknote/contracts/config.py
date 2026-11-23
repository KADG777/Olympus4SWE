"""Configuration contract for BlackNote.

Invariant implemented here: **INV-config-db**.

* All runtime knobs live in the frozen :class:`Settings` dataclass; business
  modules receive it explicitly (no ambient globals) so the web layer can build
  one instance per process and tests can build their own.
* Server side persistence is mandatory: ``database_url`` must point at a real
  database URL (file-backed SQLite by default, PostgreSQL accepted) and the
  data directory must survive restarts.  In-memory storage
  (``sqlite:///:memory:``) is rejected for non-test environments.
* Timeouts are expressed in **seconds** as positive floats and are frozen as
  module constants so every layer shares the same budget.
* Secrets (``secret_key``, ``admin_password``) are only read from the
  environment; defaults exist solely for ``environment == "dev"``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Final

from .errors import ContractError, ErrorCode

__all__ = [
    "ENV_PREFIX",
    "Environment",
    "DEFAULT_DB_TIMEOUT_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_SESSION_TTL_SECONDS",
    "DEFAULT_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "TIMEOUTS",
    "Settings",
    "load_settings",
    "reset_settings_cache",
    "SUPPORTED_DATABASE_SCHEMES",
]

ENV_PREFIX: Final[str] = "BLACKNOTE_"

# --------------------------------------------------------------------------- #
# Timeout constants (seconds) — frozen budget shared by every layer.
# --------------------------------------------------------------------------- #
DEFAULT_DB_TIMEOUT_SECONDS: Final[float] = 5.0
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 15.0
DEFAULT_SESSION_TTL_SECONDS: Final[int] = 60 * 60 * 24 * 7  # 7 days
DEFAULT_IDLE_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final[float] = 3.0

TIMEOUTS: Final[Mapping[str, float | int]] = MappingProxyType(
    {
        "db": DEFAULT_DB_TIMEOUT_SECONDS,
        "request": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "session_ttl": DEFAULT_SESSION_TTL_SECONDS,
        "idle": DEFAULT_IDLE_TIMEOUT_SECONDS,
        "connect": DEFAULT_CONNECT_TIMEOUT_SECONDS,
    }
)

SUPPORTED_DATABASE_SCHEMES: Final[frozenset[str]] = frozenset(
    {"sqlite", "sqlite+aiosqlite", "postgresql", "postgresql+asyncpg", "mysql", "mysql+pymysql"}
)
_DATABASE_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<scheme>[a-z][a-z0-9+.\-]*)://(?P<rest>.*)$", re.IGNORECASE
)


class Environment:
    DEV: Final[str] = "dev"
    TEST: Final[str] = "test"
    STAGING: Final[str] = "staging"
    PROD: Final[str] = "prod"
    VALUES: Final[tuple[str, ...]] = ("dev", "test", "staging", "prod")


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration (INV-config-db)."""

    app_name: str = "BlackNote"
    environment: str = Environment.DEV
    # --- storage -------------------------------------------------------------
    database_url: str = "sqlite:///data/blacknote.db"
    data_dir: str = "data"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_timeout_seconds: float = DEFAULT_DB_TIMEOUT_SECONDS
    db_echo: bool = False
    # --- request / session budgets -------------------------------------------
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    # --- security ------------------------------------------------------------
    secret_key: str | None = None
    secure_cookies: bool = False
    cookie_name: str = "bn_session"
    password_hash_iterations: int = 600_000
    admin_email: str = "admin@blacknote.local"
    admin_password: str | None = None
    # --- pagination / upload mirrors of validation.LIMITS --------------------
    default_page_size: int = 20
    max_page_size: int = 50
    max_image_bytes: int = 5 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp", "image/gif")
    # --- misc ----------------------------------------------------------------
    markdown_enabled: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        # ``extra`` is exposed read-only to keep the frozen contract honest.
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra or {})))
        if self.environment not in Environment.VALUES:
            object.__setattr__(self, "environment", self.environment.strip().lower())
        self.validate()

    # -- derived helpers ------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment in (Environment.PROD, Environment.STAGING)

    @property
    def database_scheme(self) -> str:
        match = _DATABASE_URL_PATTERN.match(self.database_url.strip())
        return match.group("scheme").lower() if match else ""

    @property
    def is_sqlite(self) -> bool:
        return self.database_scheme.startswith("sqlite")

    def timeout(self, name: str) -> float:
        """Uniform access to the timeout budget (``db`` / ``request`` / ...)."""
        mapping = {
            "db": self.db_timeout_seconds,
            "request": self.request_timeout_seconds,
            "connect": self.connect_timeout_seconds,
            "idle": self.idle_timeout_seconds,
            "session_ttl": self.session_ttl_seconds,
        }
        try:
            return float(mapping[name])
        except KeyError:
            raise ContractError(
                ErrorCode.CONFIG_ERROR, f"unknown timeout bucket: {name}"
            ) from None

    # -- validation -----------------------------------------------------------
    def validate(self) -> None:
        """Fail fast on an unusable configuration (raised at start-up)."""
        problems: list[str] = []

        url = (self.database_url or "").strip()
        match = _DATABASE_URL_PATTERN.match(url)
        if not match:
            problems.append("database_url must be a full URL, e.g. sqlite:///data/blacknote.db")
        else:
            scheme, rest = match.group("scheme").lower(), match.group("rest")
            if scheme not in SUPPORTED_DATABASE_SCHEMES:
                problems.append(
                    "database_url scheme must be one of: "
                    + ", ".join(sorted(SUPPORTED_DATABASE_SCHEMES))
                )
            body = rest.split("?", 1)[0].lstrip("/")
            in_memory = body in ("", ":memory:")
            if in_memory and self.environment != Environment.TEST:
                # Server side persistence is a hard product requirement: content
                # must survive a restart.
                problems.append("in-memory database is only allowed in the test environment")
            if not in_memory and scheme.startswith("sqlite") and not body:
                problems.append("sqlite database_url must point at a file path")
            if not scheme.startswith("sqlite") and not body:
                problems.append("database_url must address a host and a database")

        for name in (
            "db_timeout_seconds",
            "request_timeout_seconds",
            "connect_timeout_seconds",
            "idle_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                problems.append(f"{name} must be a positive number of seconds")
        if self.session_ttl_seconds <= 0:
            problems.append("session_ttl_seconds must be a positive number of seconds")
        if self.connect_timeout_seconds > self.request_timeout_seconds:
            problems.append("connect_timeout_seconds must not exceed request_timeout_seconds")
        if self.db_pool_size < 1:
            problems.append("db_pool_size must be >= 1")
        if not (1 <= self.default_page_size <= self.max_page_size):
            problems.append("default_page_size must be between 1 and max_page_size")
        if self.max_page_size < 1:
            problems.append("max_page_size must be >= 1")
        if self.password_hash_iterations < 1000:
            problems.append("password_hash_iterations must be >= 1000")
        if self.max_image_bytes <= 0:
            problems.append("max_image_bytes must be positive")
        if not self.cookie_name.strip():
            problems.append("cookie_name must not be blank")

        if self.is_production:
            if not self.secret_key:
                problems.append("secret_key is required outside dev/test")
            if not self.admin_password:
                problems.append("admin_password is required outside dev/test")
            if not self.secure_cookies:
                problems.append("secure_cookies must be enabled outside dev/test")
            if url.startswith("sqlite"):
                problems.append("a client/server database is required outside dev/test")

        if problems:
            raise ContractError(
                ErrorCode.CONFIG_ERROR,
                "invalid settings: " + "; ".join(problems),
                {"problems": problems},
            )

    # -- IO -------------------------------------------------------------------
    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for meta in fields(self):
            value = getattr(self, meta.name)
            payload[meta.name] = list(value) if isinstance(value, tuple) else value
        payload["extra"] = dict(self.extra)
        if redact:
            for key in ("secret_key", "admin_password"):
                payload[key] = "***" if payload.get(key) else None
            url = str(payload.get("database_url", ""))
            if "@" in url:
                payload["database_url"] = re.sub(r"//[^@/]+@", "//***@", url)
        return payload

    def replace(self, **changes: Any) -> "Settings":
        """Return a validated copy with *changes* applied."""
        merged = dict(self.extra)
        merged.update(changes.pop("extra", {}) or {})
        return replace(self, extra=merged, **changes)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from ``BLACKNOTE_*`` environment variables."""
        source: Mapping[str, str] = os.environ if environ is None else environ
        defaults = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, meta in defaults.items():
            if name == "extra":
                continue
            raw = source.get(ENV_PREFIX + name.upper())
            if raw is None or raw == "":
                continue
            kwargs[name] = _coerce(raw, meta.type, name)
        return cls(**kwargs)


_BOOLEAN_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOLEAN_FALSE = frozenset({"0", "false", "no", "off"})


def _coerce(raw: str, declared: Any, name: str) -> Any:
    text = raw.strip()
    target = str(declared)
    if "bool" in target:
        lowered = text.lower()
        if lowered in _BOOLEAN_TRUE:
            return True
        if lowered in _BOOLEAN_FALSE:
            return False
        raise ContractError(ErrorCode.CONFIG_ERROR, f"{name} must be a boolean")
    if "int" in target and "tuple" not in target:
        try:
            return int(text)
        except ValueError:
            raise ContractError(ErrorCode.CONFIG_ERROR, f"{name} must be an integer") from None
    if "float" in target:
        try:
            return float(text)
        except ValueError:
            raise ContractError(ErrorCode.CONFIG_ERROR, f"{name} must be a number") from None
    if "tuple" in target:
        return tuple(item.strip() for item in text.split(",") if item.strip())
    return text


_CACHE: dict[str, Settings] = {}


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    force_refresh: bool = False,
) -> Settings:
    """Process-wide accessor: ``BLACKNOTE_*`` env overrides dataclass defaults."""
    key = "default" if environ is None else str(sorted(environ.items()))
    if force_refresh or key not in _CACHE:
        _CACHE[key] = Settings.from_env(environ)
    return _CACHE[key]


def reset_settings_cache() -> None:
    """Test helper: drop memoised settings."""
    _CACHE.clear()
