"""Load non-secret settings and resolve credentials at the process boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

STATE_DIR = Path("~/.config/olympus-key-ledger")
DEFAULT_CONFIG = STATE_DIR / "config.yaml"
Dialect = Literal["openai", "anthropic"]
Currency = Literal["USD", "CNY"]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
SecretRef = Annotated[str, StringConstraints(pattern=r"^(env|file):[A-Za-z_][A-Za-z0-9_]*$")]
Rate = Annotated[Decimal, Field(ge=0, max_digits=16, decimal_places=6)]
Cap = Annotated[Decimal, Field(gt=0, max_digits=22, decimal_places=12)]
FILE_LIMIT = 1024 * 1024
DIALECT_PATH: dict[Dialect, str] = {
    "openai": "/chat/completions",
    "anthropic": "/messages",
}


class ConfigError(ValueError):
    """A configuration error whose message is safe to display."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Price(Settings):
    currency: Currency = "USD"
    input_per_1m: Rate
    output_per_1m: Rate
    cached_input_per_1m: Rate
    cache_write_per_1m: Rate


class Budget(Settings):
    daily: Cap | None = None
    total: Cap | None = None


class Slot(Settings):
    key_alias: Identifier
    secret: SecretRef = Field(repr=False)
    budget: Budget = Field(default_factory=Budget)


class Pool(Settings):
    dialect: Dialect
    base_url: str
    rotation: Literal["round_robin", "least_used", "least_cost"] = "round_robin"
    currency: Currency = "USD"
    cooldown_seconds: int = Field(default=60, ge=1, le=86400)
    inject_stream_usage: bool = True
    budget: Budget = Field(default_factory=Budget)
    keys: list[Slot] = Field(min_length=1, max_length=100)
    pricing: dict[str, Price] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def upstream_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if any(not 32 < ord(char) < 127 or char == "\\" for char in value):
            raise ValueError("invalid upstream URL")
        parts = urlsplit(value)
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username is not None
            or "?" in value
            or "#" in value
            or parts.port == 0
        ):
            raise ValueError("expected an HTTP(S) base URL without credentials, query or fragment")
        return value

    @model_validator(mode="after")
    def check_pool(self) -> Self:
        if len({key.key_alias for key in self.keys}) != len(self.keys):
            raise ValueError("key aliases must be unique within a pool")
        if any(price.currency != self.currency for price in self.pricing.values()):
            raise ValueError("all prices must use the pool currency")
        return self

    @property
    def endpoint(self) -> str:
        base = self.base_url
        path = urlsplit(base).path
        needs_version = not path if self.dialect == "openai" else not path.endswith("/v1")
        if needs_version:
            base += "/v1"
        return base + DIALECT_PATH[self.dialect]


class Config(Settings):
    database: Path = STATE_DIR / "ledger.sqlite3"
    secrets_file: Path = STATE_DIR / "secrets.json"
    pools: dict[Identifier, Pool] = Field(min_length=1, max_length=100)
    default_pools: dict[Dialect, Identifier] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_defaults(self) -> Self:
        for dialect, pool_id in self.default_pools.items():
            if pool_id not in self.pools or self.pools[pool_id].dialect != dialect:
                raise ValueError("each default pool must exist and match its dialect")
        return self

    def default_pool(self, dialect: Dialect) -> str | None:
        if dialect in self.default_pools:
            return self.default_pools[dialect]
        matches = [name for name, pool in self.pools.items() if pool.dialect == dialect]
        return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class Credential:
    value: str = field(repr=False)
    fingerprint: str


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    try:
        with path.open("rb") as stream:
            data = stream.read(FILE_LIMIT + 1)
        if len(data) > FILE_LIMIT:
            raise ConfigError("configuration is too large")
        config = Config.model_validate(yaml.safe_load(data))
    except ValidationError as exc:
        fields = sorted({str(error["loc"][0]) for error in exc.errors() if error["loc"]})
        # Mapping keys can be user input. Only display fixed top-level field names.
        fields = [name for name in fields if name in Config.model_fields]
        suffix = f" ({', '.join(fields)})" if fields else ""
        raise ConfigError(f"invalid ledger configuration{suffix}") from None
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError):
        raise ConfigError("cannot read ledger configuration") from None
    config.database = (path.parent / config.database.expanduser()).resolve()
    config.secrets_file = (path.parent / config.secrets_file.expanduser()).absolute()
    if config.database == config.secrets_file or config.database == path:
        raise ConfigError("database, configuration, and secrets must use separate files")
    return config


def _read_secrets(path: Path) -> dict[str, str]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    getuid = getattr(os, "getuid", None)
    if os.name != "posix" or nofollow is None or nonblock is None or getuid is None:
        raise ConfigError("file secrets require POSIX 0600 permissions; use env references here")
    try:
        fd = os.open(path, os.O_RDONLY | nofollow | nonblock)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != getuid()
            ):
                raise ConfigError("secrets file must be a regular file owned by you with mode 0600")
            payload = stream.read(FILE_LIMIT + 1)
        if len(payload) > FILE_LIMIT:
            raise ConfigError("secrets file is too large")
        data = json.loads(payload)
    except (OSError, ValueError, RecursionError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError("cannot read secrets file") from None
    if not isinstance(data, dict) or any(not isinstance(value, str) for value in data.values()):
        raise ConfigError("secrets file must map names to strings")
    return {str(name): str(value) for name, value in data.items()}


def resolve_credentials(
    config: Config, environ: Mapping[str, str] | None = None
) -> dict[tuple[str, str], Credential]:
    env = os.environ if environ is None else environ
    secrets: dict[str, str] | None = None
    result: dict[tuple[str, str], Credential] = {}
    for pool_id, pool in config.pools.items():
        used: set[str] = set()
        for slot in pool.keys:
            kind, name = slot.secret.split(":", 1)
            if kind == "env":
                key = env.get(name, "")
            else:
                if secrets is None:
                    secrets = _read_secrets(config.secrets_file)
                key = secrets.get(name, "")
            if not re.fullmatch(r"[!-~]{1,4096}", key):
                raise ConfigError(f"missing or invalid credential for {pool_id}/{slot.key_alias}")
            digest = hashlib.sha256(key.encode("ascii")).hexdigest()
            if digest in used:
                raise ConfigError(f"duplicate credentials in pool {pool_id}")
            used.add(digest)
            result[pool_id, slot.key_alias] = Credential(key, digest[:16])
    return result
