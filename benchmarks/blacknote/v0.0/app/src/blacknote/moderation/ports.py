"""Ports and adapters of ``mod-moderation``.

The moderation layer owns *no* content table: notes live in ``mod-note`` and
comments in ``mod-interaction``.  It therefore talks to them through two tiny,
duck-typed ports (the same discipline ``mod-note`` applies to the account layer,
so the four Wave-1 modules can be built in parallel without importing each
other):

``ModerationTargetStore``
    ``load(target_id)`` + ``apply_moderation(target_id, state)`` — read a row
    and persist **only** the moderation flag.
``AuditLogStore``
    ``append(entry)`` (plus read-back helpers) for the audit trail.

:func:`adapt_target_store` teaches that port to any of the shapes the platform
actually ships:

* a moderation-native store (``blacknote.moderation.stores``, pass-through);
* the note service seam ``apply_moderation_state`` / ``inspect_note_record``
  (``mod-note`` already exposes exactly that pair);
* a plain repository with ``get(id)`` / ``save(record)`` (what
  ``InMemoryNoteRepository`` and the future comment repository look like);
* a mapping ``{id: row}`` (test fixtures and dev wiring).

Records may be dataclasses, ``NoteRecord``-like objects exposing ``changed()``,
or plain dicts: :func:`with_moderation` writes the canonical
:class:`~blacknote.contracts.visibility.ModerationState` **and** keeps the
boolean view (``moderated`` / ``blocked``) consistent, and drops keys a given
record type does not declare.

This module imports ``blacknote.contracts`` and the standard library only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol, runtime_checkable

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import coerce_id, is_valid_id
from blacknote.contracts.roles import Role, parse_role
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.visibility import ModerationState

from .models import (
    ModerationTargetKind,
    flag_field_for,
    moderation_state_for,
    parse_target_kind,
)

__all__ = [
    "ModerationTargetStore",
    "AuditLogStore",
    "PortRecord",
    "adapt_target_store",
    "adapt_audit_store",
    "adapt_sessions",
    "with_moderation",
    "target_state_for",
    "canonical_target_id",
    "record_key",
]

PortRecord = Any

_READ_SEAMS: Final[tuple[str, ...]] = (
    "load",
    "inspect_note_record",
    "get_record",
    "get",
    "find",
    "find_by_id",
)
_SAVE_SEAMS: Final[tuple[str, ...]] = ("save", "update", "put", "upsert", "store")


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class ModerationTargetStore(Protocol):
    """What the service needs from a note/comment store (flag writes only)."""

    def load(self, target_id: str) -> PortRecord | None:  # pragma: no cover - protocol
        ...

    def apply_moderation(
        self, target_id: str, state: ModerationState
    ) -> PortRecord | None:  # pragma: no cover - protocol
        ...


@runtime_checkable
class AuditLogStore(Protocol):
    """Append-only persistence for :class:`~blacknote.moderation.models.ModerationAuditLog`."""

    def append(self, entry: Any) -> Any:  # pragma: no cover - protocol
        ...

    def list(
        self,
        *,
        action: Any = None,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        target_id: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:  # pragma: no cover - protocol
        ...

    def count(self, *, action: Any = None) -> int:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Record level flag writer
# --------------------------------------------------------------------------- #
def target_state_for(kind: ModerationTargetKind | str) -> ModerationState:
    """Canonical ``ModerationState`` written for *kind*."""
    return moderation_state_for(parse_target_kind(kind))


def _changes(kind: str, state: ModerationState, updated_at: str | None) -> dict[str, Any]:
    """The flag delta: canonical enum + consistent boolean + refresh stamp."""
    flag = flag_field_for(kind)
    return {
        "moderation": state,
        flag: state is target_state_for(kind),
        "updated_at": updated_at,
    }


def _declared_names(record: Any) -> set[str] | None:
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return {meta.name for meta in dataclasses.fields(record)}
    return None


def _filter(changes: Mapping[str, Any], declared: set[str] | None) -> dict[str, Any]:
    if declared is None:
        return dict(changes)
    return {name: value for name, value in changes.items() if name in declared}


def with_moderation(
    record: PortRecord,
    kind: ModerationTargetKind | str,
    state: ModerationState,
    *,
    updated_at: str | None = None,
) -> PortRecord:
    """Return a copy of *record* carrying the new moderation flag.

    Supported shapes: mappings, (frozen) dataclasses — including
    ``blacknote.note.models.NoteRecord`` — and objects exposing ``changed()``.
    Anything else is a wiring bug and raises ``INTERNAL_ERROR`` rather than
    silently writing a half-updated row.
    """
    target_kind = parse_target_kind(kind)
    if not isinstance(state, ModerationState):
        state = ModerationState(str(state))
    changes = _changes(target_kind, state, updated_at)

    if isinstance(record, Mapping):
        payload = dict(record)
        payload["moderation"] = str(state)
        payload[flag_field_for(target_kind)] = changes[flag_field_for(target_kind)]
        if updated_at is not None and ("updated_at" in payload or "moderation" in payload):
            payload["updated_at"] = updated_at
        return payload  # type: ignore[return-value]

    declared = _declared_names(record)
    subset = _filter(changes, declared)
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        return dataclasses.replace(record, **subset)  # type: ignore[return-value]

    changed = getattr(record, "changed", None)
    if callable(changed):
        try:
            return changed(**subset)
        except TypeError:
            # a stricter ``changed()`` signature: retry with the enum only
            return changed(moderation=state)

    if getattr(type(record), "__dataclass_fields__", None):  # dataclass *class* row
        return dataclasses.replace(record, **subset)  # type: ignore[return-value]

    setter = getattr(record, "set_moderation", None)
    if callable(setter):
        setter(state)
        return record

    raise ContractError(
        ErrorCode.INTERNAL_ERROR,
        "moderation store returned an unsupported record shape",
        {"record_type": type(record).__name__, "kind": target_kind},
    )


def record_key(record: PortRecord) -> str | None:
    """Primary key of *record* (used to keep repository keys canonical)."""
    if record is None:
        return None
    if isinstance(record, Mapping):
        return coerce_id(record.get("id"))
    return coerce_id(getattr(record, "id", None))


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class _NativeStore:
    """Pass-through for objects already speaking ``ModerationTargetStore``."""

    def __init__(self, target: Any, *, kind: str) -> None:
        self._target = target
        self._kind = kind

    def load(self, target_id: str) -> PortRecord | None:
        return self._target.load(target_id)

    def apply_moderation(self, target_id: str, state: ModerationState) -> PortRecord | None:
        return self._target.apply_moderation(target_id, state)


class _NoteServiceStore:
    """Adapter over the ``mod-note`` seam ``apply_moderation_state``.

    Reads go through a storage probe (``inspect_note_record`` / ``repository`` /
    ``get``) on purpose: the public ``get_note`` read path filters through the
    shared visibility predicate, so an already unlisted note would look absent.
    Moderation must still be able to re-apply (idempotent) or restore a flag.
    """

    def __init__(self, target: Any, *, kind: str) -> None:
        self._target = target
        self._kind = kind
        self._apply = target.apply_moderation_state
        self._reader = _pick_reader(target)

    def load(self, target_id: str) -> PortRecord | None:
        if self._reader is None:
            return None
        try:
            return self._reader(target_id)
        except ContractError:
            return None

    def apply_moderation(self, target_id: str, state: ModerationState) -> PortRecord | None:
        try:
            return self._apply(target_id, state)
        except ContractError as exc:
            if exc.code in (ErrorCode.NOTE_NOT_FOUND, ErrorCode.COMMENT_NOT_FOUND):
                return None
            raise


class _RepositoryStore:
    """Adapter over a ``get``/``save`` repository of record-like rows."""

    def __init__(self, target: Any, *, kind: str,
                 clock: Callable[[], str] | None = None) -> None:
        self._target = target
        self._kind = kind
        self._clock = clock or now_iso
        self._reader = _pick_reader(target)
        writer = _pick_writer(target)
        if self._reader is None or writer is None:
            raise ContractError(
                ErrorCode.CONFIG_ERROR,
                "moderation store must expose load()/apply_moderation() or a "
                "get()+save() repository pair",
                {"kind": kind},
            )
        self._writer = writer

    def load(self, target_id: str) -> PortRecord | None:
        try:
            record = self._reader(target_id)
        except ContractError:
            return None
        if record is None and hasattr(self._target, "rows"):  # dict-backed fixtures
            record = self._target.rows.get(target_id)
        return record

    def apply_moderation(self, target_id: str, state: ModerationState) -> PortRecord | None:
        record = self.load(target_id)
        if record is None:
            return None
        updated = with_moderation(record, self._kind, state, updated_at=self._clock())
        self._writer(updated)
        return updated


class _MappingStore:
    """Adapter over ``{id: row}`` mappings (fixtures, quick dev wiring)."""

    def __init__(self, target: Mapping[str, Any], *, kind: str,
                 clock: Callable[[], str] | None = None) -> None:
        self._rows = target
        self._kind = kind
        self._clock = clock or now_iso

    def load(self, target_id: str) -> PortRecord | None:
        return self._rows.get(target_id)

    def apply_moderation(self, target_id: str, state: ModerationState) -> PortRecord | None:
        record = self._rows.get(target_id)
        if record is None:
            return None
        updated = with_moderation(record, self._kind, state, updated_at=self._clock())
        self._rows[target_id] = updated
        return updated


def _pick_reader(target: Any) -> Callable[[str], Any] | None:
    for name in _READ_SEAMS:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    repository = getattr(target, "repository", None)
    getter = getattr(repository, "get", None)
    return getter if callable(getter) else None


def _pick_writer(target: Any) -> Callable[[Any], Any] | None:
    for name in _SAVE_SEAMS:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    return None


def adapt_target_store(
    target: Any,
    *,
    kind: ModerationTargetKind | str,
    clock: Callable[[], str] | None = None,
) -> ModerationTargetStore:
    """Normalise any plausible note/comment store into the moderation port.

    ``clock`` only stamps rows this layer writes itself (repository / mapping
    shapes); the native stores and the note service seam use their own clock.
    """
    if target is None:
        raise ContractError(
            ErrorCode.CONFIG_ERROR, "a note/comment store is required", {"kind": str(kind)}
        )
    target_kind = parse_target_kind(kind)
    if isinstance(target, ModerationTargetKind):  # defensive: wiring mistake
        raise ContractError(ErrorCode.CONFIG_ERROR, "target store expected, kind given")
    if isinstance(target, _NativeStore | _NoteServiceStore | _RepositoryStore | _MappingStore):
        return target  # type: ignore[return-value]
    if isinstance(target, Mapping):
        return _MappingStore(target, kind=target_kind, clock=clock)
    if all(callable(getattr(target, name, None)) for name in ("load", "apply_moderation")):
        return _NativeStore(target, kind=target_kind)
    if callable(getattr(target, "apply_moderation_state", None)):
        return _NoteServiceStore(target, kind=target_kind)
    return _RepositoryStore(target, kind=target_kind, clock=clock)


# --------------------------------------------------------------------------- #
# Audit store adaptation
# --------------------------------------------------------------------------- #
def adapt_audit_store(target: Any) -> AuditLogStore:
    """Accept a native store or anything with ``append`` (+ optional reads)."""
    from .stores import InMemoryAuditLogStore, WrappingAuditLogStore

    if target is None:
        return InMemoryAuditLogStore()
    if isinstance(target, AuditLogStore) and callable(getattr(target, "list", None)):
        return target  # type: ignore[return-value]
    if callable(getattr(target, "append", None)):
        return WrappingAuditLogStore(target)
    raise ContractError(
        ErrorCode.CONFIG_ERROR,
        "audit store must expose append() (and ideally list()/count())",
    )


# --------------------------------------------------------------------------- #
# Session resolver adaptation (INV-auth-session plumbing)
# --------------------------------------------------------------------------- #
def _to_principal(candidate: object, token: str) -> Principal | None:
    """Normalise whatever the injected dependency returns into a ``Principal``."""
    if candidate is None or isinstance(candidate, ContractError):
        return None
    if isinstance(candidate, Principal):
        return candidate
    if isinstance(candidate, Mapping):
        payload = dict(candidate)
        user_id = payload.pop("user_id", None) or payload.pop("id", None)
        payload.pop("token", None)
        if not is_valid_id(user_id):
            return None
        try:
            role = parse_role(payload.pop("role", Role.USER))
        except ContractError:
            role = Role.USER
        return Principal(
            user_id=str(user_id),
            token=token,
            role=role,
            issued_at=payload.get("issued_at"),
            expires_at=payload.get("expires_at"),
            nickname=payload.get("nickname"),
        )
    user_id = getattr(candidate, "user_id", None) or getattr(candidate, "id", None)
    if not is_valid_id(user_id):
        return None
    return Principal(
        user_id=str(user_id),
        token=token,
        role=getattr(candidate, "role", Role.USER),
        issued_at=getattr(candidate, "issued_at", None),
        expires_at=getattr(candidate, "expires_at", None),
        nickname=getattr(candidate, "nickname", None),
    )


def adapt_sessions(
    resolver: object,
) -> Callable[[str], Principal | None] | None:
    """Turn any plausible session port into a ``contracts.require_auth`` resolver.

    Accepts ``None`` (the contract guard then fails closed), a fixed
    :class:`Principal`, an object with ``resolve`` / ``current_user``, a
    ``token → principal`` mapping or a plain callable.  The returned callable
    only *describes* the identity — ``require_admin`` keeps ownership of the
    authorisation decision (INV-role-check).
    """
    if resolver is None:
        return None
    if isinstance(resolver, Principal):
        fixed = resolver
        return lambda _token: fixed
    if hasattr(resolver, "resolve"):
        raw_resolve = resolver.resolve  # type: ignore[attr-defined]
    elif hasattr(resolver, "current_user"):
        raw_resolve = resolver.current_user  # type: ignore[attr-defined]
    elif isinstance(resolver, Mapping):
        table = dict(resolver)

        def raw_resolve(token: str, _table: dict = table) -> Any:  # type: ignore[misc]
            return _table.get(token)

    elif callable(resolver):
        raw_resolve = resolver
    else:
        raise ContractError(
            ErrorCode.CONFIG_ERROR,
            "session resolver must expose resolve()/current_user() or be callable",
        )

    def _resolve(token: str) -> Principal | None:
        return _to_principal(raw_resolve(token), token)

    return _resolve


def canonical_target_id(value: object) -> str | None:
    """Canonical UUIDv4 form of *value*, or ``None``.

    A malformed id is **not** a validation failure here: the moderation contract
    freezes only ``AUTH_REQUIRED`` / ``PERMISSION_DENIED`` / ``NOTE_NOT_FOUND`` /
    ``COMMENT_NOT_FOUND``, and existence must never leak, so the caller maps
    ``None`` onto the same 404 payload as a missing row.
    """
    return coerce_id(value)
