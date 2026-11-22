"""Rich return envelope for internal callers (``mod-web``, ``mod-autotest``).

The frozen ``IF-interaction-actions`` signatures return bare values, and failures
are raised as :class:`~blacknote.contracts.errors.ContractError` carrying the
single canonical envelope (``OQ-A2``).  This dataclass is the *internal* seam:
the web layer may either call the frozen methods and catch, or call the
``*_result`` variants and branch on :attr:`InteractionResult.ok` without
exceptions — both paths execute the same use case, so they cannot disagree.

The raised and the returned envelope are the same object shape:
``InteractionResult.from_error(exc).envelope == exc.envelope``, and
:meth:`InteractionResult.to_dict` renders the complete response payload —
including the transport ``status`` — so a handler needs no try/except at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from blacknote.contracts.errors import (
    ContractError,
    ErrorEnvelope,
    ErrorCode,
    as_envelope,
    is_error,
    make_error,
    status_for,
)
from blacknote.contracts.interfaces import Counters

__all__ = ["InteractionResult"]

T = TypeVar("T")


def _clean_counters(value: Any) -> Counters:
    if value is None:
        return Counters()
    if isinstance(value, Counters):
        return value
    if isinstance(value, Mapping):
        return Counters(**{k: value[k] for k in ("likes", "favorites", "comments") if k in value})
    raise TypeError(f"unsupported counters payload: {type(value).__name__}")


@dataclass(frozen=True)
class InteractionResult(Generic[T]):
    """Outcome of one interaction use case."""

    ok: bool
    value: T | None = None
    actor_id: str | None = None
    counters: Counters = Counters()
    changed: bool = False
    entity_id: str | None = None
    error: ErrorCode | str | None = None
    message: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "counters", _clean_counters(self.counters))
        object.__setattr__(
            self, "details", dict(self.details) if isinstance(self.details, Mapping) else {}
        )
        if self.error is not None:
            object.__setattr__(self, "error", ErrorCode(str(self.error)))
        if not self.ok and self.error is None:
            object.__setattr__(self, "error", ErrorCode.INTERNAL_ERROR)
        if self.ok:
            # a success never carries an error artefact
            object.__setattr__(self, "message", None)
            object.__setattr__(self, "details", {})
            object.__setattr__(self, "request_id", None)

    # -- failure construction -------------------------------------------------
    @classmethod
    def failure(
        cls,
        code: ErrorCode | str,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
        *,
        value: Any = None,
        request_id: str | None = None,
    ) -> "InteractionResult[Any]":
        return cls(
            ok=False,
            value=value,
            error=ErrorCode(str(code)),
            message=message,
            details=details or {},
            request_id=request_id,
        )

    @classmethod
    def from_error(cls, exc: Any) -> "InteractionResult[Any]":
        """Wrap a :class:`ContractError` (or anything else) as a failure.

        The contract helpers own the envelope shape, so the wrapped result and
        the raised exception describe exactly the same error: the platform
        helpers normalise any error artefact (:func:`is_error` / :func:`as_envelope`),
        and anything unexpected becomes ``INTERNAL_ERROR`` with its message kept.
        """
        if is_error(exc):
            envelope: ErrorEnvelope = as_envelope(exc)
            return cls(
                ok=False,
                error=envelope.code,
                message=envelope.message,
                details=envelope.details,
                request_id=envelope.request_id,
            )
        return cls(
            ok=False,
            error=ErrorCode.INTERNAL_ERROR,
            message=str(exc) or "unexpected interaction failure",
            details={"cause": [type(exc).__name__]},
        )

    # -- projections ----------------------------------------------------------
    @property
    def code(self) -> ErrorCode | None:
        return self.error if self.ok else (self.error or ErrorCode.INTERNAL_ERROR)

    @property
    def is_failure(self) -> bool:
        return not self.ok

    @property
    def status(self) -> int:
        if self.ok:
            return 200
        return status_for(str(self.code))

    @property
    def envelope(self) -> ErrorEnvelope | None:
        """The canonical envelope, identical to the one that would be raised."""
        if self.ok:
            return None
        return make_error(str(self.code), self.message, self.details, request_id=self.request_id)

    def to_error(self) -> ContractError | None:
        """The exception form of this failure (``None`` when the call worked)."""
        if self.ok:
            return None
        return ContractError(
            str(self.code), self.message, self.details, request_id=self.request_id
        )

    def unwrap(self) -> T:
        """Return the value or raise the platform ``ContractError`` envelope."""
        if self.ok:
            return self.value  # type: ignore[return-value]
        raise self.to_error()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "actor_id": self.actor_id,
            "counters": self.counters.to_dict(),
            "changed": self.changed,
            "entity_id": self.entity_id,
            "error": None if self.error is None else str(self.error),
            "message": self.message,
            "details": dict(self.details),
            "request_id": self.request_id,
            # the transport hint, so a handler can render the response from
            # this dict alone (200 when ok, the contract status otherwise)
            "status": self.status,
        }
