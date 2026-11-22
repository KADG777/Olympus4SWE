"""Storage level model of a note (``mod-note``).

A :class:`NoteRecord` is the *row shape* the note repository persists.  It is
deliberately separate from the frozen wire types in
:mod:`blacknote.contracts.interfaces` (``NoteView`` / ``NoteSummary``): the
record owns lifecycle flags and raw content, the view owns what a specific
actor may see.

Rules honoured here
-------------------
* ``INV-id-format``     — ``id`` / ``author_id`` are canonical UUIDv4 strings.
* ``INV-time-encoding`` — ``created_at`` / ``updated_at`` are canonical UTC
  ISO-8601 strings ending in ``Z``; ordering is therefore lexicographic.
* ``INV-visibility-predicate`` — the record only *stores* ``status`` /
  ``moderation`` / ``visibility``; it never decides whether something may be
  shown (that belongs to ``contracts.visibility.is_visible_to``).
* ``INV-concurrency-counters`` — ``counters`` is a read-through cache owned by
  the interaction layer; this layer initialises it to zeros and preserves it on
  update, it never increments or decrements it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.ids import normalize_id
from blacknote.contracts.interfaces import Counters, NoteSummary, NoteView
from blacknote.contracts.timeutil import from_iso, to_iso
from blacknote.contracts.visibility import ContentStatus, ModerationState, Visibility

__all__ = ["NoteRecord", "ANONYMOUS_NICKNAME"]

#: Fallback display name when the account directory has nothing to say.
ANONYMOUS_NICKNAME = "Anonymous"


def _canonical_timestamp(value: Any, field_name: str) -> str:
    """Force a timestamp into the canonical ``...Z`` wire form (INV-time-encoding)."""
    try:
        return to_iso(from_iso(value, field_name=field_name))
    except ContractError as exc:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_timestamp", "must be a canonical ISO-8601 UTC value")]
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


@dataclass(frozen=True)
class NoteRecord:
    """One persisted note (including soft-deleted tombstones)."""

    id: str
    author_id: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    topic: str | None = None
    image_urls: tuple[str, ...] = ()
    visibility: Visibility = Visibility.PUBLIC
    status: ContentStatus = ContentStatus.PUBLISHED
    moderation: ModerationState = ModerationState.VISIBLE
    counters: Counters = field(default_factory=Counters)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "note_id"))
        object.__setattr__(self, "author_id", normalize_id(self.author_id, "author_id"))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in (self.tags or ())))
        object.__setattr__(self, "image_urls",
                           tuple(str(url) for url in (self.image_urls or ())))
        object.__setattr__(self, "topic", self.topic or None)
        object.__setattr__(self, "visibility", _enum(self.visibility, Visibility, "visibility"))
        object.__setattr__(self, "status", _enum(self.status, ContentStatus, "status"))
        object.__setattr__(self, "moderation",
                           _enum(self.moderation, ModerationState, "moderation"))
        if not isinstance(self.counters, Counters):
            object.__setattr__(self, "counters", Counters(**dict(self.counters or {})))
        created = _canonical_timestamp(self.created_at, "created_at")
        updated = _canonical_timestamp(self.updated_at or self.created_at, "updated_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)

    # -- ordering -------------------------------------------------------------
    @property
    def sort_key(self) -> tuple[str, str]:
        """``(created_at, id)`` — the keyset used by feed ordering and cursors."""
        return (self.created_at, self.id)

    # -- derivations ----------------------------------------------------------
    @property
    def is_tombstone(self) -> bool:
        """True for soft deleted notes (``status=deleted``)."""
        return self.status is ContentStatus.DELETED

    def has_tag(self, tag: str) -> bool:
        """Case-insensitive tag membership (tags keep their canonical spelling)."""
        wanted = str(tag).casefold()
        return any(str(item).casefold() == wanted for item in self.tags)

    def changed(self, **changes: Any) -> "NoteRecord":
        """Return an updated copy (records are immutable value objects)."""
        return replace(self, **changes)

    # -- projections onto the frozen wire types -------------------------------
    def to_view(self, author_nickname: str | None = None) -> NoteView:
        """Full note payload (still subject to the shared visibility predicate)."""
        return NoteView(
            id=self.id,
            author_id=self.author_id,
            author_nickname=author_nickname or ANONYMOUS_NICKNAME,
            title=self.title,
            body=self.body,
            tags=self.tags,
            image_urls=self.image_urls,
            visibility=self.visibility,
            status=self.status,
            moderation=self.moderation,
            counters=self.counters,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_summary(self, author_nickname: str | None = None) -> NoteSummary:
        """Feed/search card (never carries the full body)."""
        return self.to_view(author_nickname).summary()

    # -- persistence helpers --------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        """Flat, JSON-safe mapping used by the SQL repository."""
        return {
            "id": self.id,
            "author_id": self.author_id,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "topic": self.topic,
            "image_urls": list(self.image_urls),
            "visibility": str(self.visibility),
            "status": str(self.status),
            "moderation": str(self.moderation),
            "like_count": self.counters.likes,
            "favorite_count": self.counters.favorites,
            "comment_count": self.counters.comments,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any, tags: Any = None) -> "NoteRecord":
        """Build a record out of a mapping / ``sqlite3.Row``."""
        data = dict(row)
        raw_tags = data.get("tags") or () if tags is None else tags
        if isinstance(raw_tags, str):
            raw_tags = [part for part in raw_tags.split(",") if part]
        counters = data.get("counters")
        if counters is None:
            counters = Counters(
                likes=int(data.get("like_count") or 0),
                favorites=int(data.get("favorite_count") or 0),
                comments=int(data.get("comment_count") or 0),
            )
        return cls(
            id=data["id"],
            author_id=data["author_id"],
            title=data["title"],
            body=data["body"],
            tags=tuple(str(tag) for tag in raw_tags),
            topic=data.get("topic"),
            image_urls=tuple(str(url) for url in (data.get("image_urls") or ())),
            visibility=data.get("visibility", Visibility.PUBLIC),
            status=data.get("status", ContentStatus.PUBLISHED),
            moderation=data.get("moderation", ModerationState.VISIBLE),
            counters=counters,
            created_at=data.get("created_at") or "",
            updated_at=data.get("updated_at") or data.get("created_at") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Wire shape of the record (used by admin tooling and tests)."""
        payload = self.to_row()
        payload["counters"] = self.counters.to_dict()
        return payload
