"""State flags and audit trail of ``mod-moderation``.

Two families of value objects live here:

``ModeratableNote`` / ``ModeratableComment``
    The *flag bearing* row shape the moderation layer reads and writes when a
    takedown is executed.  They are deliberately narrow (identity + lifecycle
    flags, no content): the real rows keep living in ``mod-note`` /
    ``mod-interaction`` and the adapters in :mod:`blacknote.moderation.ports`
    translate between the two shapes.  Every store that speaks the moderation
    port may also keep these records as-is.

``ModerationAuditLog``
    One appended-only security/moderation record
    (``action ∈ {note_unlisted, comment_blocked, privilege_denied}``).

Contract rules honoured here
----------------------------
* ``INV-id-format`` — ``id`` and every ``*_id`` reference are canonical UUIDv4
  strings (validated on construction, coerced on read).
* ``INV-time-encoding`` — ``created_at`` is always the canonical UTC
  ``...Z`` ISO-8601 string, so rows sort chronologically as text.
* ``INV-visibility-predicate`` — the records only *store* ``status`` /
  ``visibility`` / ``moderation``.  They expose a convenience boolean view of
  the moderation flag (``Note.moderated`` / ``Comment.blocked``) but never
  decide visibility: ``contracts.visibility.is_visible_to`` stays the single
  predicate, and the boolean always agrees with the canonical
  :class:`~blacknote.contracts.visibility.ModerationState` it is derived from.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.ids import coerce_id, is_valid_id, new_id
from blacknote.contracts.roles import Role
from blacknote.contracts.timeutil import from_iso, now_iso, to_iso
from blacknote.contracts.visibility import (
    ContentStatus,
    ModerationState,
    Visibility,
    visibility_of,
)

__all__ = [
    "ModerationAction",
    "ModerationTargetKind",
    "AUDIT_INTERFACE_ID",
    "TARGET_KIND_NOTE",
    "TARGET_KIND_COMMENT",
    "TARGET_STATE",
    "FLAG_FIELD",
    "ACTION_TARGET_KIND",
    "AUDIT_COLUMNS",
    "parse_action",
    "parse_target_kind",
    "moderation_state_for",
    "flag_field_for",
    "flag_of",
    "moderation_of",
    "ModerationAuditLog",
    "ModeratableNote",
    "ModeratableComment",
]

TARGET_KIND_NOTE: Final[str] = "note"
TARGET_KIND_COMMENT: Final[str] = "comment"

#: The frozen contract whose execution these records audit
#: (:data:`blacknote.contracts.interfaces.INTERFACES["IF-moderation-actions"]`).
AUDIT_INTERFACE_ID: Final[str] = "IF-moderation-actions"


class ModerationAction(StrEnum):
    """Vocabulary of ``ModerationAuditLog.action`` (frozen, lowercase)."""

    NOTE_UNLISTED = "note_unlisted"          # admin 下架笔记
    COMMENT_BLOCKED = "comment_blocked"      # admin 屏蔽评论
    PRIVILEGE_DENIED = "privilege_denied"    # 越权尝试（INV-role-check）


class ModerationTargetKind(StrEnum):
    """What a moderation action operates on."""

    NOTE = TARGET_KIND_NOTE
    COMMENT = TARGET_KIND_COMMENT


#: The canonical contract state each takedown writes (INV-visibility-predicate:
#: only ``ModerationState`` values are ever persisted, no private flag vocabulary).
TARGET_STATE: Final[Mapping[str, ModerationState]] = MappingProxyType(
    {
        TARGET_KIND_NOTE: ModerationState.UNLISTED,
        TARGET_KIND_COMMENT: ModerationState.BLOCKED,
    }
)

#: ``target kind → boolean field name`` of the persisted row.
#: ``unlist_note`` flips ``Note.moderated``, ``block_comment`` flips
#: ``Comment.blocked``; both are pure derivations of ``moderation``.
FLAG_FIELD: Final[Mapping[str, str]] = MappingProxyType(
    {
        TARGET_KIND_NOTE: "moderated",
        TARGET_KIND_COMMENT: "blocked",
    }
)

#: ``audit action → target kind`` (``privilege_denied`` has no reachable target).
ACTION_TARGET_KIND: Final[Mapping[str, str | None]] = MappingProxyType(
    {
        str(ModerationAction.NOTE_UNLISTED): TARGET_KIND_NOTE,
        str(ModerationAction.COMMENT_BLOCKED): TARGET_KIND_COMMENT,
        str(ModerationAction.PRIVILEGE_DENIED): None,
    }
)


def parse_action(value: object) -> ModerationAction:
    """Coerce *value* into a :class:`ModerationAction` (fails closed)."""
    if isinstance(value, ModerationAction):
        return value
    if not isinstance(value, str):
        raise ContractError.validation(
            [FieldError("action", "invalid_action", "audit action must be a string")]
        )
    try:
        return ModerationAction(value.strip().lower())
    except ValueError:
        raise ContractError.validation(
            [FieldError("action", "invalid_action",
                        f"action must be one of: {', '.join(str(a) for a in ModerationAction)}")]
        ) from None


def parse_target_kind(value: object) -> str:
    """Coerce *value* into a target kind string, rejecting anything else."""
    if isinstance(value, ModerationTargetKind):
        return str(value)
    if isinstance(value, str):
        try:
            return str(ModerationTargetKind(value.strip().lower()))
        except ValueError:
            pass
    raise ContractError.validation(
        [FieldError("target_type", "invalid_target_kind",
                    f"target type must be one of: {TARGET_KIND_NOTE}, {TARGET_KIND_COMMENT}")]
    )


def moderation_state_for(kind: object) -> ModerationState:
    """The single ``ModerationState`` a takedown of *kind* must persist."""
    return TARGET_STATE[parse_target_kind(kind)]


def flag_field_for(kind: object) -> str:
    """Name of the boolean flag column backing *kind*."""
    return FLAG_FIELD[parse_target_kind(kind)]


# --------------------------------------------------------------------------- #
# Generic flag readers (no judgment: they only project stored columns)
# --------------------------------------------------------------------------- #
def _attr(record: Any, name: str) -> Any:
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def moderation_of(record: Any) -> ModerationState:
    """Canonical moderation state of *record* (absent → ``visible``)."""
    state = visibility_of(record).moderation
    if isinstance(state, ModerationState):
        return state
    if state is None:
        return ModerationState.VISIBLE
    if isinstance(state, str):
        try:
            return ModerationState(state.strip())
        except ValueError:
            pass
    # An unparsable flag is a corrupt row: surface it instead of inventing a
    # state.  The shared predicate already treats it as invisible, so this
    # layer never has to re-decide visibility.
    raise ContractError(
        "DATABASE_ERROR", "stored moderation flag is not a contract value",
        {"moderation": str(state)},
    )


def flag_of(record: Any, kind: object) -> bool:
    """Boolean view (``Note.moderated`` / ``Comment.blocked``) of *record*.

    Read from the canonical ``moderation`` column so the boolean and the enum
    can never disagree, even for rows written by another layer.
    """
    field_name = flag_field_for(kind)
    declared = _attr(record, field_name)
    state = moderation_of(record)
    derived = state is moderation_state_for(kind)
    if declared is not None and not isinstance(declared, bool):
        # tolerate stringly-typed stores ("true"/"1") but never invent a value
        declared = str(declared).strip().lower() in ("true", "1", "yes", "y")
    return bool(declared) or derived


# --------------------------------------------------------------------------- #
# Timestamp helpers (INV-time-encoding)
# --------------------------------------------------------------------------- #
def canonical_timestamp(value: Any, field_name: str = "created_at") -> str:
    """Force *value* into the canonical ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` form."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return now_iso()
    try:
        return to_iso(from_iso(value, field_name=field_name))
    except ContractError as exc:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp",
                        "must be an ISO-8601 UTC timestamp")]
        ) from exc


def _enum(value: Any, kind: Any, field_name: str) -> Any:
    """Coerce into a contract lifecycle enum; unknown values fail closed."""
    if isinstance(value, kind):
        return value
    if value is None:
        return None
    try:
        return kind(str(value).strip())
    except ValueError:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_choice",
                        f"must be one of: {', '.join(str(item) for item in kind)}")]
        ) from None


def _optional_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not is_valid_id(text):
        raise ContractError.validation(
            [FieldError(field_name, "invalid_id", "must be a canonical UUIDv4 string")]
        )
    return text


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
#: Column order of the persisted audit row (kept identical to the dataclass
#: fields so ``to_row`` / ``from_row`` / the SQL schema can never drift).
AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "action",
    "user_id",
    "note_id",
    "comment_id",
    "reason",
    "details",
    "created_at",
)


@dataclass(frozen=True)
class ModerationAuditLog:
    """One immutable moderation/security record (append-only store)."""

    id: str
    action: str
    user_id: str | None = None
    note_id: str | None = None
    comment_id: str | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict, compare=False)
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", coerce_id(self.id) or new_id())
        object.__setattr__(self, "action", str(parse_action(self.action)))
        object.__setattr__(self, "user_id", _optional_id(self.user_id, "user_id"))
        object.__setattr__(self, "note_id", _optional_id(self.note_id, "note_id"))
        object.__setattr__(self,
                           "comment_id", _optional_id(self.comment_id, "comment_id"))
        reason = None if self.reason is None else str(self.reason).strip() or None
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "details", dict(self.details or {}))
        object.__setattr__(self, "created_at", canonical_timestamp(self.created_at))

    # -- constructors ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        action: ModerationAction | str,
        *,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        id: str | None = None,
    ) -> ModerationAuditLog:
        """Build a fully validated entry (id/timestamp filled when omitted)."""
        return cls(
            id=id or new_id(),
            action=str(parse_action(action)),
            user_id=user_id,
            note_id=note_id,
            comment_id=comment_id,
            reason=reason,
            details=dict(details or {}),
            created_at=created_at or now_iso(),
        )

    @classmethod
    def denied(
        cls,
        *,
        user_id: str | None,
        operation: str,
        note_id: str | None = None,
        comment_id: str | None = None,
        reason: str | None = None,
        actual_role: str | None = None,
        created_at: str | None = None,
    ) -> ModerationAuditLog:
        """One ``privilege_denied`` record for a rejected admin action."""
        details: dict[str, Any] = {
            "operation": str(operation),
            "required_roles": [str(Role.ADMIN)],
            "interface": AUDIT_INTERFACE_ID,
        }
        if actual_role:
            details["actual_role"] = str(actual_role)
        return cls.create(
            ModerationAction.PRIVILEGE_DENIED,
            user_id=user_id,
            note_id=note_id,
            comment_id=comment_id,
            reason=reason or "administrator role required",
            details=details,
            created_at=created_at,
        )

    # -- views ----------------------------------------------------------------
    @property
    def target_kind(self) -> str | None:
        """``note`` / ``comment`` / ``None`` for a refused attempt."""
        return ACTION_TARGET_KIND[str(self.action)]

    @property
    def target_id(self) -> str | None:
        """The referenced content id, whichever kind it is."""
        return self.note_id or self.comment_id

    @property
    def is_denial(self) -> bool:
        return self.action == str(ModerationAction.PRIVILEGE_DENIED)

    # -- persistence ----------------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        """JSON-safe, column-ordered mapping for the SQL store."""
        return {
            "id": self.id,
            "action": self.action,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "comment_id": self.comment_id,
            "reason": self.reason,
            "details": json.dumps(dict(self.details), ensure_ascii=False, sort_keys=True),
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> ModerationAuditLog:
        """Rebuild an entry from a mapping / ``sqlite3.Row``."""
        data = dict(row)
        details = data.get("details")
        if isinstance(details, str):
            try:
                details = json.loads(details) if details.strip() else {}
            except json.JSONDecodeError:
                details = {"raw": details}
        return cls(
            id=data["id"],
            action=data["action"],
            user_id=data.get("user_id"),
            note_id=data.get("note_id"),
            comment_id=data.get("comment_id"),
            reason=data.get("reason"),
            details=dict(details or {}),
            created_at=data.get("created_at") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Wire shape (admin console / JSON envelope payloads)."""
        return {
            "id": self.id,
            "action": self.action,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "comment_id": self.comment_id,
            "reason": self.reason,
            "details": dict(self.details),
            "created_at": self.created_at,
        }

    def matches(
        self,
        *,
        action: ModerationAction | str | None = None,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        target_id: str | None = None,
    ) -> bool:
        """Filter helper shared by every audit store."""
        if action is not None and self.action != str(parse_action(action)):
            return False
        if user_id is not None and self.user_id != coerce_id(user_id):
            return False
        if note_id is not None and self.note_id != coerce_id(note_id):
            return False
        if comment_id is not None and self.comment_id != coerce_id(comment_id):
            return False
        return target_id is None or self.target_id == coerce_id(target_id)


# --------------------------------------------------------------------------- #
# Flag bearing content records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModeratableNote:
    """Note row as seen by the moderation layer (flags only, no body)."""

    id: str
    author_id: str
    title: str = ""
    status: str = str(ContentStatus.PUBLISHED)
    visibility: str = str(Visibility.PUBLIC)
    moderation: ModerationState = ModerationState.VISIBLE
    moderated: bool = False
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", coerce_id(self.id) or _bad_id("note_id"))
        if not is_valid_id(self.author_id):
            raise ContractError.validation(
                [FieldError("author_id", "invalid_id", "must be a canonical UUIDv4 string")]
            )
        object.__setattr__(self, "status",
                           _enum(self.status, ContentStatus, "status") or ContentStatus.PUBLISHED)
        object.__setattr__(self, "visibility",
                           _enum(self.visibility, Visibility, "visibility") or Visibility.PUBLIC)
        state = _enum(self.moderation, ModerationState, "moderation") or ModerationState.VISIBLE
        object.__setattr__(self, "moderation", state)
        # The boolean is a pure projection of the contract enum.
        object.__setattr__(self, "moderated", state is ModerationState.UNLISTED)
        created = canonical_timestamp(self.created_at)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at",
                           canonical_timestamp(self.updated_at or created))

    # -- convenience ----------------------------------------------------------
    @property
    def is_tombstone(self) -> bool:
        return self.status == str(ContentStatus.DELETED)

    def changed(self, **changes: Any) -> ModeratableNote:
        """``NoteRecord``-compatible copy helper (unknown keys are dropped)."""
        allowed = {name: value for name, value in changes.items()
                   if name in {meta.name for meta in self.__dataclass_fields__.values()}}
        return replace(self, **allowed)

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        payload["visibility"] = str(self.visibility)
        payload["moderation"] = str(self.moderation)
        return payload

    @classmethod
    def from_row(cls, row: Any) -> ModeratableNote:
        data = dict(row)
        return cls(
            id=data["id"],
            author_id=data["author_id"],
            title=data.get("title", ""),
            status=data.get("status", ContentStatus.PUBLISHED),
            visibility=data.get("visibility", Visibility.PUBLIC),
            moderation=data.get("moderation", ModerationState.VISIBLE),
            created_at=data.get("created_at") or "",
            updated_at=data.get("updated_at") or data.get("created_at") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_row()


@dataclass(frozen=True)
class ModeratableComment:
    """Comment row as seen by the moderation layer (flags only, no body)."""

    id: str
    note_id: str
    author_id: str
    parent_comment_id: str | None = None
    status: str = str(ContentStatus.PUBLISHED)
    moderation: ModerationState = ModerationState.VISIBLE
    blocked: bool = False
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", coerce_id(self.id) or _bad_id("comment_id"))
        object.__setattr__(self, "note_id", coerce_id(self.note_id) or _bad_id("note_id"))
        if not is_valid_id(self.author_id):
            raise ContractError.validation(
                [FieldError("author_id", "invalid_id", "must be a canonical UUIDv4 string")]
            )
        if self.parent_comment_id is not None:
            object.__setattr__(
                self, "parent_comment_id",
                coerce_id(self.parent_comment_id) or _bad_id("parent_comment_id"),
            )
        object.__setattr__(self, "status",
                           _enum(self.status, ContentStatus, "status") or ContentStatus.PUBLISHED)
        state = _enum(self.moderation, ModerationState, "moderation") or ModerationState.VISIBLE
        object.__setattr__(self, "moderation", state)
        object.__setattr__(self, "blocked", state is ModerationState.BLOCKED)
        created = canonical_timestamp(self.created_at)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at",
                           canonical_timestamp(self.updated_at or created))

    @property
    def is_reply(self) -> bool:
        return self.parent_comment_id is not None

    @property
    def is_tombstone(self) -> bool:
        return self.status == str(ContentStatus.DELETED)

    def changed(self, **changes: Any) -> ModeratableComment:
        allowed = {name: value for name, value in changes.items()
                   if name in {meta.name for meta in self.__dataclass_fields__.values()}}
        return replace(self, **allowed)

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        payload["moderation"] = str(self.moderation)
        return payload

    @classmethod
    def from_row(cls, row: Any) -> ModeratableComment:
        data = dict(row)
        return cls(
            id=data["id"],
            note_id=data["note_id"],
            author_id=data["author_id"],
            parent_comment_id=data.get("parent_comment_id"),
            status=data.get("status", ContentStatus.PUBLISHED),
            moderation=data.get("moderation", ModerationState.VISIBLE),
            created_at=data.get("created_at") or "",
            updated_at=data.get("updated_at") or data.get("created_at") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_row()


def _bad_id(field_name: str) -> Any:
    """Helper used inside ``object.__setattr__`` calls that always raises."""
    raise ContractError.validation(
        [FieldError(field_name, "invalid_id", "must be a canonical UUIDv4 string")]
    )
