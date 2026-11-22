"""Identifier contract for BlackNote.

Invariant implemented here: **INV-id-format**.

* Every public resource id (user, note, comment, session token) is a
  **UUIDv4** rendered as the canonical lowercase hyphenated string
  (``xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx`` with ``y`` in ``[89ab]``).
* No other representation (int autoincrement, braces, ``urn:``, uppercase,
  base64 short-ids) is accepted at a module boundary.
* The nil UUID and UUIDs of another version/variant are rejected.
* Validation never mutates the input: use :func:`normalize_id` when a tolerant
  parse is required, otherwise reject with ``VALIDATION_ERROR``.
"""

from __future__ import annotations

import re
import uuid
from typing import Final, Iterable

from .errors import FieldError, ContractError

__all__ = [
    "ID_VERSION",
    "ID_MAX_LENGTH",
    "UUID4_PATTERN",
    "NIL_ID",
    "new_id",
    "generate_id",
    "is_valid_id",
    "normalize_id",
    "parse_id",
    "require_id",
    "require_ids",
    "coerce_id",
]

ID_VERSION: Final[int] = 4
ID_MAX_LENGTH: Final[int] = 36
NIL_ID: Final[str] = "00000000-0000-0000-0000-000000000000"

#: Canonical textual form of a UUIDv4 (lowercase, hyphenated, no braces / urn).
UUID4_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def new_id() -> str:
    """Return a fresh UUIDv4 as a canonical lowercase string."""
    return str(uuid.uuid4())


#: Backwards/forwards friendly alias.
generate_id = new_id


def is_valid_id(value: object) -> bool:
    """Return True only for canonical UUIDv4 strings."""
    if not isinstance(value, str):
        return False
    if len(value) != ID_MAX_LENGTH or not UUID4_PATTERN.match(value):
        return False
    if value == NIL_ID:
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return (
        str(parsed) == value
        and parsed.version == ID_VERSION
        and parsed.variant == uuid.RFC_4122
    )


def coerce_id(value: object) -> str | None:
    """Tolerantly convert *value* into a canonical id, or ``None``."""
    if isinstance(value, uuid.UUID):
        return str(value) if value.version == ID_VERSION and value.variant == uuid.RFC_4122 else None
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    candidate = candidate.strip("{}")
    if candidate.startswith("urn:uuid:"):
        candidate = candidate[len("urn:uuid:"):]
    return candidate if is_valid_id(candidate) else None


def normalize_id(value: object, field_name: str = "id") -> str:
    """Same as :func:`coerce_id` but raises ``VALIDATION_ERROR`` on failure."""
    coerced = coerce_id(value)
    if coerced is None:
        raise ContractError.validation(
            [FieldError(field=field_name, code="invalid_id",
                        message="must be a canonical UUIDv4 string")]
        )
    return coerced


def parse_id(value: object, field_name: str = "id") -> uuid.UUID:
    """Parse *value* into a :class:`uuid.UUID` (v4, RFC 4122) or raise."""
    text = normalize_id(value, field_name)
    return uuid.UUID(text)


def require_id(value: object, field_name: str = "id") -> str:
    """Strict validation: the input must already be canonical (no coercion)."""
    if not is_valid_id(value):
        raise ContractError.validation(
            [FieldError(field=field_name, code="invalid_id",
                        message="must be a canonical UUIDv4 string")]
        )
    return value  # type: ignore[return-value]


def require_ids(values: Iterable[object] | None, field_name: str = "ids") -> tuple[str, ...]:
    """Validate a sequence of ids, de-duplicating while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in values or ():
        text = require_id(item, field_name)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)
