"""Shared error contract for BlackNote.

Invariant implemented here: **INV-error-envelope**.

Rules frozen by this module (every other layer MUST reuse them verbatim):

* Any failure crossing a module / HTTP boundary is expressed as an
  :class:`ErrorEnvelope`.  Raw exceptions, stack traces or provider specific
  payloads are never leaked to callers.
* ``code`` is always one of :class:`ErrorCode` (a ``StrEnum``, therefore it is
  directly comparable with its upper-snake-case string form, e.g.
  ``ErrorCode.AUTH_REQUIRED == "AUTH_REQUIRED"``).
* The envelope is JSON-safe and has exactly the keys
  ``{"code", "message", "details", "request_id"}`` after ``to_dict()``.
* Every code has a deterministic HTTP status mapping (:func:`status_for`).
* Field level validation problems are carried in
  ``details["fields"]`` as a list of ``{"field", "code", "message"}``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "ErrorCode",
    "ErrorEnvelope",
    "FieldError",
    "ContractError",
    "HTTP_STATUS",
    "DEFAULT_MESSAGES",
    "ENVELOPE_KEYS",
    "FIELD_ERRORS_KEY",
    "status_for",
    "message_for",
    "make_error",
    "validation_error",
    "not_found_error",
    "permission_error",
    "auth_error",
    "is_error",
    "as_envelope",
    "unwrap",
]


class ErrorCode(StrEnum):
    """Stable business error codes (frozen public contract)."""

    # generic / transport
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    # authN / authZ  (INV-auth-session, INV-role-check)
    AUTH_REQUIRED = "AUTH_REQUIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    # account domain (IF-account-*)
    EMAIL_TAKEN = "EMAIL_TAKEN"
    USER_NOT_FOUND = "USER_NOT_FOUND"
    # note domain (IF-note-*)
    NOTE_NOT_FOUND = "NOTE_NOT_FOUND"
    # interaction domain (IF-interaction-*)
    COMMENT_NOT_FOUND = "COMMENT_NOT_FOUND"

    @property
    def http_status(self) -> int:
        """Deterministic HTTP status for this code."""
        return status_for(self)

    @property
    def default_message(self) -> str:
        return message_for(self)


HTTP_STATUS: Final[Mapping[str, int]] = {
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.AUTH_REQUIRED: 401,
    ErrorCode.INVALID_TOKEN: 401,
    ErrorCode.SESSION_EXPIRED: 401,
    ErrorCode.INVALID_CREDENTIALS: 401,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.NOTE_NOT_FOUND: 404,
    ErrorCode.COMMENT_NOT_FOUND: 404,
    ErrorCode.USER_NOT_FOUND: 404,
    ErrorCode.EMAIL_TAKEN: 409,
    ErrorCode.CONFLICT: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.CONFIG_ERROR: 500,
    ErrorCode.DATABASE_ERROR: 500,
}

DEFAULT_MESSAGES: Final[Mapping[str, str]] = {
    ErrorCode.VALIDATION_ERROR: "One or more fields are invalid.",
    ErrorCode.AUTH_REQUIRED: "Authentication is required to perform this action.",
    ErrorCode.INVALID_TOKEN: "The session token is not valid.",
    ErrorCode.SESSION_EXPIRED: "The session has expired, please sign in again.",
    ErrorCode.INVALID_CREDENTIALS: "Email or password is incorrect.",
    ErrorCode.PERMISSION_DENIED: "You are not allowed to perform this action.",
    ErrorCode.NOT_FOUND: "The requested resource does not exist.",
    ErrorCode.NOTE_NOT_FOUND: "The note does not exist or is no longer available.",
    ErrorCode.COMMENT_NOT_FOUND: "The comment does not exist or is no longer available.",
    ErrorCode.USER_NOT_FOUND: "The user does not exist.",
    ErrorCode.EMAIL_TAKEN: "That email address is already registered.",
    ErrorCode.CONFLICT: "The request conflicts with the current state.",
    ErrorCode.RATE_LIMITED: "Too many requests, please slow down.",
    ErrorCode.TIMEOUT: "The operation timed out.",
    ErrorCode.INTERNAL_ERROR: "Internal server error.",
    ErrorCode.CONFIG_ERROR: "Server configuration is invalid.",
    ErrorCode.DATABASE_ERROR: "Storage layer failure.",
}

ENVELOPE_KEYS: Final[tuple[str, ...]] = ("code", "message", "details", "request_id")
FIELD_ERRORS_KEY: Final[str] = "fields"


def status_for(code: str | ErrorCode) -> int:
    """Return the HTTP status for *code* (500 for unknown codes, fail closed)."""
    try:
        return HTTP_STATUS[_coerce_code(code)]
    except (KeyError, ValueError):
        return 500


def message_for(code: str | ErrorCode) -> str:
    """Return the default human readable message for *code*."""
    try:
        return DEFAULT_MESSAGES[_coerce_code(code)]
    except (KeyError, ValueError):
        return "Unexpected error."


def _coerce_code(code: str | ErrorCode) -> ErrorCode:
    if isinstance(code, ErrorCode):
        return code
    if not isinstance(code, str):
        raise TypeError(f"error code must be a str or ErrorCode, got {type(code).__name__}")
    return ErrorCode(code.strip().upper())


def _jsonify(value: Any) -> Any:
    """Best-effort conversion of *value* into JSON-safe primitives."""
    if isinstance(value, StrEnum):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_jsonify(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonify(to_dict())
    return str(value)


@dataclass(frozen=True)
class FieldError:
    """A single field level problem, rendered inside ``details['fields']``."""

    field: str
    code: str = "invalid"
    message: str = "Invalid value."

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("FieldError.field must be a non-empty string")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("FieldError.code must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ErrorEnvelope:
    """The only error shape allowed across module boundaries."""

    code: ErrorCode
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _coerce_code(self.code))
        message = self.message if isinstance(self.message, str) and self.message.strip() else ""
        if not message:
            message = message_for(self.code)
        object.__setattr__(self, "message", message)
        details = self.details if self.details else {}
        if not isinstance(details, Mapping):
            raise TypeError("ErrorEnvelope.details must be a mapping")
        object.__setattr__(self, "details", dict(details))

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ErrorEnvelope":
        if not isinstance(payload, Mapping):
            raise TypeError("error envelope payload must be a mapping")
        if "code" not in payload:
            raise ValueError("error envelope payload requires a 'code'")
        return cls(
            code=payload["code"],
            message=payload.get("message") or "",
            details=payload.get("details") or {},
            request_id=payload.get("request_id"),
        )

    def with_details(self, **kwargs: Any) -> "ErrorEnvelope":
        merged = dict(self.details)
        merged.update(kwargs)
        return ErrorEnvelope(self.code, self.message, merged, self.request_id)

    def with_request_id(self, request_id: str | None) -> "ErrorEnvelope":
        return ErrorEnvelope(self.code, self.message, dict(self.details), request_id)

    def with_field_errors(self, field_errors: Sequence[FieldError]) -> "ErrorEnvelope":
        return self.with_details(
            **{FIELD_ERRORS_KEY: [fe.to_dict() for fe in _as_field_errors(field_errors)]}
        )

    # -- serialisation --------------------------------------------------------
    @property
    def http_status(self) -> int:
        return status_for(self.code)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": str(self.code),
            "message": self.message,
            "details": _jsonify(self.details),
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @property
    def field_errors(self) -> tuple[FieldError, ...]:
        raw = self.details.get(FIELD_ERRORS_KEY) or ()
        out: list[FieldError] = []
        for item in raw:
            if isinstance(item, FieldError):
                out.append(item)
            elif isinstance(item, Mapping):
                out.append(
                    FieldError(
                        field=str(item.get("field", "")),
                        code=str(item.get("code", "invalid")),
                        message=str(item.get("message", "")),
                    )
                )
        return tuple(out)

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"[{self.code}] {self.message}"


def _as_field_errors(items: Any) -> list[FieldError]:
    errors: list[FieldError] = []
    for item in items or ():
        if isinstance(item, FieldError):
            errors.append(item)
        elif isinstance(item, Mapping):
            errors.append(
                FieldError(
                    field=str(item.get("field", "")),
                    code=str(item.get("code", "invalid")),
                    message=str(item.get("message", "")),
                )
            )
        else:  # pragma: no cover - defensive
            raise TypeError(f"expected FieldError or mapping, got {type(item).__name__}")
    return errors


def make_error(
    code: str | ErrorCode,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> ErrorEnvelope:
    """Build an :class:`ErrorEnvelope`; *message* defaults to the canned text."""
    return ErrorEnvelope(
        code=_coerce_code(code),
        message=message or message_for(code),
        details=dict(details or {}),
        request_id=request_id,
    )


def validation_error(
    field_errors: Sequence[FieldError] | None = (),
    message: str | None = None,
    **details: Any,
) -> ErrorEnvelope:
    payload = dict(details)
    errors = _as_field_errors(field_errors)
    if errors:
        payload[FIELD_ERRORS_KEY] = [fe.to_dict() for fe in errors]
    return make_error(ErrorCode.VALIDATION_ERROR, message, payload)


def not_found_error(code: str | ErrorCode = ErrorCode.NOT_FOUND, **details: Any) -> ErrorEnvelope:
    return make_error(code, None, details)


def permission_error(**details: Any) -> ErrorEnvelope:
    return make_error(ErrorCode.PERMISSION_DENIED, None, details)


def auth_error(**details: Any) -> ErrorEnvelope:
    return make_error(ErrorCode.AUTH_REQUIRED, None, details)


class ContractError(Exception):
    """Exception wrapper that always carries a valid :class:`ErrorEnvelope`.

    Raising this from a service is equivalent to returning the envelope; the
    web layer converts it into an HTTP response via :meth:`to_envelope`.
    """

    def __init__(
        self,
        code: str | ErrorCode,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
        *,
        field_errors: Sequence[FieldError] | None = None,
        request_id: str | None = None,
    ) -> None:
        envelope = make_error(code, message, details, request_id=request_id)
        if field_errors:
            envelope = envelope.with_field_errors(field_errors)
        super().__init__(str(envelope))
        self._envelope = envelope

    @property
    def envelope(self) -> ErrorEnvelope:
        return self._envelope

    @property
    def code(self) -> ErrorCode:
        return self._envelope.code

    @property
    def http_status(self) -> int:
        return self._envelope.http_status

    @property
    def field_errors(self) -> tuple[FieldError, ...]:
        return self._envelope.field_errors

    def to_envelope(self) -> ErrorEnvelope:
        return self._envelope

    def to_dict(self) -> dict[str, Any]:
        return self._envelope.to_dict()

    @classmethod
    def validation(
        cls,
        field_errors: Sequence[FieldError],
        message: str | None = None,
        **details: Any,
    ) -> "ContractError":
        return cls(ErrorCode.VALIDATION_ERROR, message, details, field_errors=field_errors)

    @classmethod
    def auth_required(cls, message: str | None = None, **details: Any) -> "ContractError":
        return cls(ErrorCode.AUTH_REQUIRED, message, details)

    @classmethod
    def permission_denied(cls, message: str | None = None, **details: Any) -> "ContractError":
        return cls(ErrorCode.PERMISSION_DENIED, message, details)

    @classmethod
    def from_envelope(
        cls, value: "ErrorEnvelope | ContractError | Mapping[str, Any]"
    ) -> "ContractError":
        """Rebuild an exception out of any error artefact."""
        if isinstance(value, ContractError):
            return value
        envelope = value if isinstance(value, ErrorEnvelope) else ErrorEnvelope.from_dict(value)
        return cls(envelope.code, envelope.message, envelope.details, request_id=envelope.request_id)


def is_error(value: Any) -> bool:
    """True when *value* is an error artefact (used with union return types)."""
    return isinstance(value, (ErrorEnvelope, ContractError))


def as_envelope(value: Any) -> ErrorEnvelope:
    """Normalise any error artefact into an :class:`ErrorEnvelope`."""
    if isinstance(value, ErrorEnvelope):
        return value
    if isinstance(value, ContractError):
        return value.envelope
    if isinstance(value, Mapping):
        return ErrorEnvelope.from_dict(value)
    raise TypeError(f"not an error artefact: {type(value).__name__}")


def unwrap(value: Any) -> Any:
    """Return the payload or raise the embedded error (helper for callers)."""
    if is_error(value):
        raise ContractError.from_envelope(value)
    return value
