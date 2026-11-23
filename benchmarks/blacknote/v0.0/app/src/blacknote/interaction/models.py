"""Storage level models of the interaction layer (``mod-interaction``).

Three row shapes live here, mirroring the three relations the layer owns:

:class:`LikeRecord`      ``note_likes``      — one ``(user_id, note_id)`` edge.
:class:`FavoriteRecord`  ``note_favorites``  — one ``(user_id, note_id)`` edge.
:class:`CommentRecord`   ``comments``        — one comment, optionally threaded
                                             under another comment.

Rules honoured here
-------------------
* ``INV-id-format`` — every primary key is a UUIDv4 string created with
  ``contracts.ids.new_id()``; foreign references are named ``note_id`` /
  ``user_id`` / ``parent_comment_id``.  The surrogate ``id`` of a like/favorite
  row is a real id too (never an auto-increment), while idempotency is enforced
  by the **``(user_id, note_id)`` uniqueness**, not by the primary key.
* ``INV-time-encoding`` — ``created_at`` / ``updated_at`` are canonical UTC
  ISO-8601 strings ending in ``Z``, so lexicographic ordering equals
  chronological ordering (comment lists are ordered by ``created_at ASC``).
* ``INV-visibility-predicate`` — a :class:`CommentRecord` only *stores*
  ``status`` / ``moderation``; it never decides whether it may be shown.
  ``is_hidden_for_public()`` deliberately does not exist: reads funnel through
  :func:`blacknote.contracts.visibility.is_visible_to`.
* ``INV-concurrency-counters`` — the models carry **no** counter field.  Likes,
  favorites and comments are the source of truth; the note-side cache columns
  are maintained transactionally by :mod:`blacknote.interaction.repository`.

This module imports ``blacknote.contracts`` and the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.ids import is_valid_id, normalize_id
from blacknote.contracts.interfaces import CommentView
from blacknote.contracts.timeutil import from_iso, now_iso, to_iso
from blacknote.contracts.validation import clean_text
from blacknote.contracts.visibility import ContentStatus, ModerationState

__all__ = [
    "LikeRecord",
    "FavoriteRecord",
    "CommentRecord",
    "relation_key",
    "ANONYMOUS_NICKNAME",
]

#: Fallback display name when the identity directory has nothing to say.
ANONYMOUS_NICKNAME = "Anonymous"


def _canonical_timestamp(value: Any, field_name: str) -> str:
    """Force a timestamp into the canonical ``...Z`` wire form (INV-time-encoding).

    ``None`` / ``""`` means "write it now": the models stay constructible without
    a clock argument while never storing a non-canonical value.
    """
    if value is None or value == "":
        return now_iso()
    try:
        return to_iso(from_iso(value, field_name=field_name))
    except ContractError as exc:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp",
                        "must be a canonical ISO-8601 UTC value")]
        ) from exc


def _enum(value: Any, kind: Any, field_name: str) -> Any:
    """Coerce *value* into one of the frozen lifecycle enums, fail closed."""
    if isinstance(value, kind):
        return value
    try:
        return kind(str(value).strip())
    except ValueError:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_choice",
                        f"must be one of: {', '.join(str(item) for item in kind)}")]
        ) from None


def relation_key(user_id: Any, note_id: Any) -> tuple[str, str] | None:
    """Canonical ``(user_id, note_id)`` key of a like/favorite edge.

    Returns ``None`` when either side is not a valid id, which keeps the
    uniqueness key space free of garbage without raising inside the store.
    """
    if not is_valid_id(user_id) or not is_valid_id(note_id):
        return None
    return (str(user_id), str(note_id))


@dataclass(frozen=True)
class LikeRecord:
    """One ``user likes note`` edge (unique per ``(user_id, note_id)``)."""

    id: str
    user_id: str
    note_id: str
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "id"))
        object.__setattr__(self, "user_id", normalize_id(self.user_id, "user_id"))
        object.__setattr__(self, "note_id", normalize_id(self.note_id, "note_id"))
        object.__setattr__(self, "created_at", _canonical_timestamp(self.created_at, "created_at"))

    @property
    def key(self) -> tuple[str, str]:
        """The uniqueness key backing idempotent likes."""
        return (self.user_id, self.note_id)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "LikeRecord":
        data = dict(row)
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            note_id=data["note_id"],
            created_at=data.get("created_at") or "",
        )


@dataclass(frozen=True)
class FavoriteRecord:
    """One ``user favorites note`` edge (unique per ``(user_id, note_id)``)."""

    id: str
    user_id: str
    note_id: str
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "id"))
        object.__setattr__(self, "user_id", normalize_id(self.user_id, "user_id"))
        object.__setattr__(self, "note_id", normalize_id(self.note_id, "note_id"))
        object.__setattr__(self, "created_at", _canonical_timestamp(self.created_at, "created_at"))

    @property
    def key(self) -> tuple[str, str]:
        """The uniqueness key backing idempotent favorites."""
        return (self.user_id, self.note_id)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "note_id": self.note_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "FavoriteRecord":
        data = dict(row)
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            note_id=data["note_id"],
            created_at=data.get("created_at") or "",
        )


@dataclass(frozen=True)
class CommentRecord:
    """One persisted comment, threaded through ``parent_comment_id``.

    Replies may target any comment of the same note, so threads nest to
    arbitrary depth; the note id is denormalised onto every row (cheap
    ``WHERE note_id = ?`` listing and a natural defence against cross-note
    parent ids, which the service validates before the write).
    """

    id: str
    note_id: str
    author_id: str
    body: str
    parent_comment_id: str | None = None
    status: ContentStatus = ContentStatus.PUBLISHED
    moderation: ModerationState = ModerationState.VISIBLE
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "comment_id"))
        object.__setattr__(self, "note_id", normalize_id(self.note_id, "note_id"))
        object.__setattr__(self, "author_id", normalize_id(self.author_id, "author_id"))
        if self.parent_comment_id is not None:
            object.__setattr__(
                self, "parent_comment_id",
                normalize_id(self.parent_comment_id, "parent_comment_id"),
            )
        object.__setattr__(self, "body", clean_text(self.body))
        object.__setattr__(self, "status", _enum(self.status, ContentStatus, "status"))
        object.__setattr__(
            self, "moderation", _enum(self.moderation, ModerationState, "moderation")
        )
        created = _canonical_timestamp(self.created_at, "created_at")
        updated = _canonical_timestamp(self.updated_at or self.created_at, "updated_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)

    # -- projections ----------------------------------------------------------
    @property
    def is_reply(self) -> bool:
        return self.parent_comment_id is not None

    @property
    def sort_key(self) -> tuple[str, str]:
        """``(created_at, id)`` — chronological order, stable for equal stamps."""
        return (self.created_at, self.id)

    def changed(self, **changes: Any) -> "CommentRecord":
        """Return an updated copy (records are immutable value objects)."""
        return replace(self, **changes)

    def to_view(self, author_nickname: str | None = None) -> CommentView:
        """Project onto the frozen ``CommentView`` wire type."""
        return CommentView(
            id=self.id,
            note_id=self.note_id,
            author_id=self.author_id,
            author_nickname=author_nickname or ANONYMOUS_NICKNAME,
            body=self.body,
            parent_comment_id=self.parent_comment_id,
            status=self.status,
            created_at=self.created_at,
        )

    # -- persistence helpers --------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "note_id": self.note_id,
            "author_id": self.author_id,
            "body": self.body,
            "parent_comment_id": self.parent_comment_id,
            "status": str(self.status),
            "moderation": str(self.moderation),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "CommentRecord":
        data = dict(row)
        return cls(
            id=data["id"],
            note_id=data["note_id"],
            author_id=data["author_id"],
            body=data.get("body") or data.get("content") or "",
            parent_comment_id=data.get("parent_comment_id"),
            status=data.get("status", ContentStatus.PUBLISHED),
            moderation=data.get("moderation", ModerationState.VISIBLE),
            created_at=data.get("created_at") or "",
            updated_at=data.get("updated_at") or data.get("created_at") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_row()
        payload["blocked"] = self.moderation is ModerationState.BLOCKED
        return payload
