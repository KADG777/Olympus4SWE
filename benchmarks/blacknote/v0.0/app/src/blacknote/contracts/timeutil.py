"""Time contract for BlackNote.

Invariant implemented here: **INV-time-encoding**.

* The process clock is always UTC aware (:func:`now_utc`); naive datetimes are
  rejected everywhere in the code base.
* The wire/storage representation is an ISO-8601 timestamp in UTC with a
  literal ``Z`` designator and microsecond precision, e.g.
  ``2026-02-14T09:30:00.000000Z``.  Fixed width keeps lexicographic ordering
  equal to chronological ordering.
* Parsing tolerates the equivalent ``+00:00`` form and non-UTC offsets (they
  are converted to UTC), but never accepts a naive string unless the caller
  explicitly opts in with ``assume_utc=True``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, UTC
from typing import Any, Final

from .errors import ContractError, FieldError

__all__ = [
    "UTC",
    "UTC_DESIGNATOR",
    "ISO8601_UTC_PATTERN",
    "now_utc",
    "now_iso",
    "to_iso",
    "from_iso",
    "is_iso8601",
    "to_epoch_ms",
    "from_epoch_ms",
    "is_expired",
    "seconds_until",
    "ensure_aware",
]

UTC_DESIGNATOR: Final[str] = "Z"
_CANONICAL_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%f"
_ISO_FRACTIONLESS: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}(?::?\d{2})?)?$"
)
#: Canonical output format produced by :func:`to_iso`.
ISO8601_UTC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)


def now_utc() -> datetime:
    """Timezone-aware current time in UTC."""
    return datetime.now(UTC)


def ensure_aware(value: datetime, *, field_name: str = "timestamp") -> datetime:
    """Return *value* converted to UTC, rejecting naive datetimes."""
    if not isinstance(value, datetime):
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp", "expected a datetime instance")]
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ContractError.validation(
            [FieldError(field_name, "naive_datetime",
                        "timezone-aware UTC datetimes are required (INV-time-encoding)")]
        )
    return value.astimezone(UTC)


def now_iso() -> str:
    """Canonical ISO-8601 UTC string for the current instant."""
    return to_iso(now_utc())


def to_iso(value: datetime) -> str:
    """Encode an aware datetime as the canonical UTC ISO-8601 string."""
    return ensure_aware(value).strftime(_CANONICAL_FORMAT) + UTC_DESIGNATOR


def from_iso(
    value: Any,
    *,
    field_name: str = "timestamp",
    assume_utc: bool = False,
) -> datetime:
    """Decode an ISO-8601 string into an aware UTC datetime.

    ``assume_utc=True`` allows naive input (interpreted as UTC); by default
    naive input is rejected so typos cannot silently shift the clock.
    """
    if isinstance(value, datetime):
        return ensure_aware(value, field_name=field_name)
    if not isinstance(value, str) or not value.strip():
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp", "expected a non-empty ISO-8601 string")]
        )
    text = value.strip()
    if not _ISO_FRACTIONLESS.match(text):
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp", "not a valid ISO-8601 timestamp")]
        )
    candidate = text.replace("z", "Z")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:  # pragma: no cover - guarded by regex
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp", str(exc))]
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        if not assume_utc:
            raise ContractError.validation(
                [FieldError(field_name, "naive_timestamp",
                            "timestamp must carry a UTC offset or the 'Z' designator")]
            )
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def is_iso8601(value: object, *, canonical_only: bool = False) -> bool:
    """Cheap format check used by validators and tests."""
    if not isinstance(value, str):
        return False
    if canonical_only:
        return bool(ISO8601_UTC_PATTERN.match(value.strip()))
    try:
        from_iso(value)
    except ContractError:
        return False
    return True


def to_epoch_ms(value: datetime) -> int:
    """Milliseconds since the unix epoch (UTC)."""
    return int(ensure_aware(value).timestamp() * 1000)


def from_epoch_ms(value: int | float) -> datetime:
    """Inverse of :func:`to_epoch_ms`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError.validation(
            [FieldError("epoch_ms", "invalid_number", "expected an integer millisecond value")]
        )
    return datetime.fromtimestamp(value / 1000, UTC)


def is_expired(expires_at: Any, *, now: datetime | None = None) -> bool:
    """True when *expires_at* is in the past.

    ``None`` means "never expires".  Invalid input fails closed (treated as
    expired) because an unparsable expiry must not grant access.
    """
    if expires_at is None:
        return False
    try:
        moment = expires_at if isinstance(expires_at, datetime) else from_iso(expires_at)
    except ContractError:
        return True
    reference = ensure_aware(now) if now is not None else now_utc()
    return moment <= reference


def seconds_until(expires_at: Any, *, now: datetime | None = None) -> float | None:
    """Seconds remaining until *expires_at* (``None`` when there is no expiry)."""
    if expires_at is None:
        return None
    try:
        moment = expires_at if isinstance(expires_at, datetime) else from_iso(expires_at)
    except ContractError:
        return 0.0
    reference = ensure_aware(now) if now is not None else now_utc()
    return max(0.0, (moment - reference).total_seconds())
