"""Profile PATCH normalisation for ``IF-account-profile``.

The frozen contract ships :class:`~blacknote.contracts.interfaces.ProfilePatch`
covering ``nickname`` / ``bio`` / ``avatar_url``.  The account domain also owns
three identity detail fields required by this TaskNode — ``specialty``,
``affiliation`` and ``expertise_tags`` — so this module is the single place
that widens an incoming patch (a contract ``ProfilePatch``, a mapping from a
web form, or a duck-typed equivalent) into a validated payload:

* aliases are resolved (``nickname`` → ``username``, the storage column);
* ``None`` values mean "leave unchanged" (PATCH semantics);
* privilege / credential keys are refused (``role`` → PERMISSION_DENIED,
  ``email`` / ``password`` → VALIDATION_ERROR), so a profile PATCH can never
  escalate an account (**INV-role-check**);
* every size bound comes from :data:`~blacknote.contracts.validation.LIMITS`
  and is enforced by :func:`~blacknote.contracts.validation.check_fields`
  (**INV-field-validation**) — no local numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from blacknote.contracts.errors import ContractError, ErrorCode, FieldError
from blacknote.contracts.interfaces import ProfilePatch
from blacknote.contracts.validation import (
    LIMITS,
    TAG_PATTERN,
    FieldRule,
    check_fields,
    clean_text,
    normalize_tag,
)

from .models import PROFILE_PATCH_LIMITS

__all__ = [
    "DENIED_PRIVILEGE_FIELDS",
    "MULTILINE_PATCH_FIELDS",
    "PATCH_ALIASES",
    "UNSUPPORTED_PATCH_FIELDS",
    "ProfilePatchPayload",
    "build_profile_rules",
    "normalise_profile_patch",
    "profile_field_rules",
]

#: Keys that would change an account's privilege level (**INV-role-check**).
DENIED_PRIVILEGE_FIELDS: Final[frozenset[str]] = frozenset(
    {"id", "user_id", "role", "password_hash"}
)
#: Real account data, but not mutable through a profile PATCH in the MVP.
UNSUPPORTED_PATCH_FIELDS: Final[frozenset[str]] = frozenset(
    {"email", "password", "created_at", "updated_at"}
)
#: ``incoming key → storage column``.
PATCH_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "nickname": "username",
        "name": "username",
        "display_name": "username",
        "tags": "expertise_tags",
        "expertise_tags": "expertise_tags",
    }
)
#: Free-form fields whose internal newlines must survive (Markdown body).
MULTILINE_PATCH_FIELDS: Final[frozenset[str]] = frozenset({"bio"})


@dataclass(frozen=True)
class ProfilePatchPayload:
    """Validated, storage-ready profile delta."""

    values: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values or {}))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.values)

    def is_empty(self) -> bool:
        return not self.values

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def profile_field_rules() -> tuple[FieldRule, ...]:
    """Field rules for the widened patch — bounds stay owned by ``LIMITS``."""
    rules = [
        FieldRule.from_limit(
            name,
            bucket,
            required=False,
            allow_blank=(name != "username"),
        )
        for name, bucket in PROFILE_PATCH_LIMITS.items()
    ]
    rules.append(
        FieldRule(
            name="expertise_tags",
            kind="list",
            required=False,
            allow_blank=True,
            item_rule=FieldRule(
                name="item",
                min_length=int(LIMITS["tag"]["min"]),
                max_length=int(LIMITS["tag"]["max"]),
                pattern=TAG_PATTERN,
            ),
            max_items=int(LIMITS["tags"]["max"]),
        )
    )
    return tuple(rules)


#: Contract-style alias (``build_*``) kept for callers that prefer the verb form.
build_profile_rules = profile_field_rules


def _as_mapping(patch: Any) -> dict[str, Any]:
    if patch is None:
        return {}
    if isinstance(patch, ProfilePatch):
        return {
            "nickname": patch.nickname,
            "bio": patch.bio,
            "avatar_url": patch.avatar_url,
        }
    if isinstance(patch, Mapping):
        return dict(patch)
    to_payload = getattr(patch, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        if isinstance(payload, Mapping):
            return dict(payload)
    if any(hasattr(patch, name) for name in ("nickname", "username", "bio", "avatar_url")):
        return {
            name: getattr(patch, name)
            for name in ("nickname", "username", "bio", "avatar_url")
            if hasattr(patch, name)
        }
    raise ContractError.validation(
        [FieldError("__patch__", "invalid_type",
                    "profile patch must be a mapping or a ProfilePatch")]
    )


def _as_tag_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [part for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    raise ContractError.validation(
        [FieldError("expertise_tags", "invalid_type", "expertise_tags must be a list")]
    )


def _clean_expertise_tags(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    for item in _as_tag_list(value):
        try:
            tag = normalize_tag(item)
        except ContractError as exc:
            raise ContractError(
                ErrorCode.VALIDATION_ERROR,
                exc.envelope.message,
                {"fields": list(exc.envelope.details.get("fields", ()))},
            ) from exc
        if tag not in out:
            out.append(tag)
    return tuple(out)


def normalise_profile_patch(patch: Any) -> ProfilePatchPayload:
    """Validate *patch* and return the storage-ready delta.

    Raises :class:`~blacknote.contracts.errors.ContractError` with
    ``PERMISSION_DENIED`` (privilege key), or ``VALIDATION_ERROR``
    (unknown / unsupported / oversized / empty patch).
    """
    raw = _as_mapping(patch)
    unknown: list[str] = []
    requested: dict[str, Any] = {}
    for key, value in raw.items():
        name = PATCH_ALIASES.get(str(key), str(key))
        if name in DENIED_PRIVILEGE_FIELDS:
            raise ContractError(
                ErrorCode.PERMISSION_DENIED,
                "a profile patch may not change account privileges",
                {"field": name},
            )
        if name in UNSUPPORTED_PATCH_FIELDS:
            raise ContractError.validation(
                [FieldError(name, "not_supported",
                            "this field cannot be changed through a profile patch")]
            )
        if name not in PROFILE_PATCH_LIMITS and name != "expertise_tags":
            unknown.append(name)
            continue
        requested.setdefault(name, value)

    if unknown:
        raise ContractError.validation(
            [FieldError(field, "unknown_field", "unknown profile field") for field in unknown]
        )

    present = {name: value for name, value in requested.items() if value is not None}
    if not present:
        raise ContractError.validation(
            [FieldError("profile", "empty_patch",
                        "at least one profile field must be provided")]
        )

    payload_for_check = {name: present.get(name) for name in PROFILE_PATCH_LIMITS}
    payload_for_check["expertise_tags"] = present.get("expertise_tags")
    check_fields(payload_for_check, profile_field_rules(), partial=True)

    cleaned: dict[str, Any] = {}
    for name in PROFILE_PATCH_LIMITS:
        if name in present:
            cleaned[name] = clean_text(
                present[name], collapse=name not in MULTILINE_PATCH_FIELDS
            )
    if "expertise_tags" in present:
        cleaned["expertise_tags"] = _clean_expertise_tags(present["expertise_tags"])
    return ProfilePatchPayload(values=cleaned)
