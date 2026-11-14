"""Data-driven provider registry: builtin neutral specs plus external overrides."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from olympus.providers.types import ProviderError

_DEFAULT_USER_PROVIDERS_PATH = Path("~/.olympus/providers.json")


class EndpointSpec(BaseModel):
    """One endpoint entry inside a provider spec; pure data, no vendor logic."""

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    api_keys_env: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    url_suffix_rewrite: dict[str, str] = Field(default_factory=dict)


class ProviderSpec(BaseModel):
    """A provider identity: dialect plus connection/auth data."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dialect: Literal["openai-chat", "anthropic-messages"] = "openai-chat"
    aliases: list[str] = Field(default_factory=list)
    default_model: str
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    api_keys_env: str | None = None
    endpoints: list[EndpointSpec] = Field(default_factory=list)
    env_file: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    compat: dict[str, Any] = Field(default_factory=dict)
    default: bool = False


BUILTIN_PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        dialect="openai-chat",
        aliases=["openai-compatible"],
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        base_url_env="OPENAI_BASE_URL",
        api_key_env="OPENAI_API_KEY",
        api_keys_env="OPENAI_API_KEYS",
    ),
    ProviderSpec(
        id="anthropic",
        dialect="anthropic-messages",
        default_model="claude-3-7-sonnet-latest",
        base_url="https://api.anthropic.com",
        base_url_env="ANTHROPIC_BASE_URL",
        api_key_env="ANTHROPIC_API_KEY",
    ),
)


def _read_providers_file(path: Path) -> list[ProviderSpec]:
    expanded = path.expanduser()
    if not expanded.is_file():
        return []
    try:
        data = json.loads(expanded.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Invalid JSON in provider config {expanded}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProviderError(f"Provider config {expanded} must be a JSON object")
    raw = data.get("providers")
    if raw is None:
        raise ProviderError(f"Provider config {expanded} must contain a 'providers' key")

    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(value, dict):
                raise ProviderError(f"Provider entry '{key}' in {expanded} must be an object")
            entry: dict[str, Any] = dict(value)
            entry["id"] = entry.get("id") or key
            entries.append(entry)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ProviderError(f"Every provider entry in {expanded} must be an object")
            entries.append(dict(item))
    else:
        raise ProviderError(f"'providers' in {expanded} must be a list or an object")

    specs: list[ProviderSpec] = []
    for entry in entries:
        try:
            specs.append(ProviderSpec.model_validate(entry))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
                for err in exc.errors(include_input=False)
            )
            raise ProviderError(f"Invalid provider spec in {expanded}: {details}") from exc
    return specs


class ProviderRegistry:
    """Lookup table of provider specs keyed by normalized lowercase id."""

    def __init__(self, specs: Sequence[ProviderSpec]) -> None:
        self._specs: dict[str, ProviderSpec] = {}
        self._default_order: list[str] = []
        for spec in specs:
            self._register(spec)

    @classmethod
    def load(
        cls,
        *,
        user_providers_path: Path | None = _DEFAULT_USER_PROVIDERS_PATH,
        project_providers_path: Path | None = None,
        explicit_providers_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ProviderRegistry:
        """Merge builtin specs with user, project, and explicit JSON overrides."""
        proc_env = os.environ if environ is None else environ
        candidates: list[Path] = []
        if user_providers_path is not None:
            candidates.append(user_providers_path)
        project_path = (
            project_providers_path
            if project_providers_path is not None
            else Path.cwd() / ".olympus" / "providers.json"
        )
        candidates.append(project_path)
        explicit_path = explicit_providers_path
        if explicit_path is None:
            from_env = proc_env.get("OLYMPUS_PROVIDERS_FILE", "").strip()
            if from_env:
                explicit_path = Path(from_env)
        if explicit_path is not None:
            candidates.append(explicit_path)

        specs = list(BUILTIN_PROVIDERS)
        for path in candidates:
            specs.extend(_read_providers_file(path))
        return cls(specs)

    def _register(self, spec: ProviderSpec) -> None:
        key = spec.id.strip().lower()
        if not key:
            raise ProviderError("Provider spec 'id' must be non-empty")
        normalized = spec.model_copy(update={"id": key})
        self._specs[key] = normalized
        if normalized.default:
            self._default_order.append(key)

    def list_ids(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ProviderSpec | None:
        needle = name.strip().lower()
        if not needle:
            return None
        direct = self._specs.get(needle)
        if direct is not None:
            return direct
        for spec in self._specs.values():
            if any(alias.strip().lower() == needle for alias in spec.aliases):
                return spec
        return None

    def choose_auto(self, proc_env: Mapping[str, str]) -> str:
        """Resolve the ``auto`` provider id without touching credentials."""
        explicit = proc_env.get("OLYMPUS_PROVIDER", "").strip()
        if explicit:
            return explicit
        if proc_env.get("OLYMPUS_BASE_URL", "").strip():
            return "openai"
        for candidate in self._specs.values():
            if _spec_has_env_key(candidate, proc_env):
                return candidate.id
        for spec_id in reversed(self._default_order):
            default_spec = self._specs.get(spec_id)
            if default_spec is not None and default_spec.default:
                return default_spec.id
        return "openai"


def _spec_has_env_key(spec: ProviderSpec, env: Mapping[str, str]) -> bool:
    for name in (spec.api_key_env, spec.api_keys_env):
        if name and env.get(name, "").strip():
            return True
    for endpoint in spec.endpoints:
        for name in (endpoint.api_key_env, endpoint.api_keys_env):
            if name and env.get(name, "").strip():
                return True
    return False
