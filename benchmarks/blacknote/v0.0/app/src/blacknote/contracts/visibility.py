"""Visibility contract for BlackNote.

Invariant implemented here: **INV-visibility-predicate**.

:func:`is_visible_to` is the **only** place in the whole platform where the
question "may this actor see this piece of content?" is answered.  Services,
repositories, templates and API handlers must all route through it (or through
:func:`filter_visible` / :func:`require_visible`) so that the rules can never
drift apart between the feed, the note detail page, the search page, the
comment list and the profile page.

Rules (fail-closed: anything unknown is invisible)
-------------------------------------------------
``status``        ``draft``     → author only (``scope=self``), never public.
                  ``published`` → public when visibility is ``public``.
                  ``deleted``   → invisible to **everybody**, including admins
                                  (tombstone; surfaces as ``*_NOT_FOUND``).
``moderation``    ``visible``   → no effect.
                  ``unlisted``  → admin 下架: removed from every public surface
                                  (feed / search / detail / favorites).
                  ``blocked``   → comment 屏蔽: removed for everybody; only the
                                  admin moderation queue may surface it.
``visibility``    ``public`` / ``private`` → private content is author only.

Scopes
------
``public``     default; what an anonymous or third-party visitor may see.
``self``       the author's own management view (drafts and own unlisted work
               remain listed so they can still be edited/deleted).
``moderation`` admin review queue; requires an admin actor, otherwise it
               silently degrades to ``public``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

from .errors import ContractError, ErrorCode
from .roles import HasRole, is_admin

__all__ = [
    "ContentStatus",
    "ModerationState",
    "Visibility",
    "VisibilityScope",
    "VisibilityTarget",
    "NOTE_STATUS_VALUES",
    "COMMENT_STATUS_VALUES",
    "PUBLIC_STATUS",
    "DELETED_STATUS",
    "BLOCKED_MODERATION",
    "is_visible_to",
    "filter_visible",
    "require_visible",
    "visibility_of",
    "is_interactable",
    "UNKNOWN",
]


class ContentStatus(StrEnum):
    """Lifecycle of any user generated content (notes and comments)."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DELETED = "deleted"


class ModerationState(StrEnum):
    """Outcome of the (manual, admin only) moderation workflow."""

    VISIBLE = "visible"
    UNLISTED = "unlisted"  # note taken down from public surfaces
    BLOCKED = "blocked"  # comment hidden for everybody


class Visibility(StrEnum):
    """Author controlled audience."""

    PUBLIC = "public"
    PRIVATE = "private"


class VisibilityScope(StrEnum):
    """Which surface is asking."""

    PUBLIC = "public"
    SELF = "self"
    MODERATION = "moderation"


#: Statuses a note may hold (comments never use ``draft``).
NOTE_STATUS_VALUES: Final[tuple[str, ...]] = tuple(str(s) for s in ContentStatus)
COMMENT_STATUS_VALUES: Final[tuple[str, ...]] = (
    str(ContentStatus.PUBLISHED),
    str(ContentStatus.DELETED),
)
#: Fail-closed defaults used when a row carries a missing/unknown flag.
PUBLIC_STATUS: Final[str] = str(ContentStatus.PUBLISHED)
DELETED_STATUS: Final[str] = str(ContentStatus.DELETED)
BLOCKED_MODERATION: Final[str] = str(ModerationState.BLOCKED)


@runtime_checkable
class VisibilityTarget(Protocol):
    """Structural contract every visible entity must satisfy.

    ``NoteView``/``NoteSummary``/``CommentView`` implement these attributes; a
    plain mapping or ORM row with the same keys works as well.
    """

    @property
    def author_id(self) -> str | None:  # pragma: no cover - protocol
        ...

    @property
    def status(self) -> str | None: ...

    @property
    def moderation(self) -> str | None: ...

    @property
    def visibility(self) -> str | None: ...


def _attr(content: Any, name: str) -> Any:
    if content is None:
        return None
    if isinstance(content, dict):
        return content.get(name)
    value = getattr(content, name, None)
    if value is None and name == "moderation":
        # tolerate repositories that persist it as "moderation_state"
        value = getattr(content, "moderation_state", None)
    return value


#: Sentinel returned for values that are neither ``None`` nor a canonical member.
UNKNOWN: Final[str] = "unknown"


def _enum(value: Any, kind: Any) -> Any:
    """Coerce *value* into *kind* (a StrEnum).

    ``None`` means "column absent" (the caller applies the default), the
    :data:`UNKNOWN` sentinel means "unparsable" and makes the predicate fail
    closed.  Deliberately **case sensitive**: persisted lifecycle flags are
    canonical lowercase, so ``"PUBLISHED"`` is data corruption, not a synonym.
    """
    if isinstance(value, kind):
        return value
    if value is None:
        return None
    if not isinstance(value, str):
        return UNKNOWN
    try:
        return kind(value.strip())
    except ValueError:
        return UNKNOWN


@dataclass(frozen=True)
class VisibilityFacts:
    """Normalised view of a content row; handy for unit tests and adapters."""

    author_id: str | None
    status: ContentStatus | str | None
    moderation: ModerationState | str | None = ModerationState.VISIBLE
    visibility: Visibility | str | None = Visibility.PUBLIC

    @classmethod
    def from_content(cls, content: Any) -> "VisibilityFacts":
        return cls(
            author_id=_attr(content, "author_id"),
            status=_attr(content, "status"),
            moderation=_attr(content, "moderation"),
            visibility=_attr(content, "visibility"),
        )


def visibility_of(content: Any) -> VisibilityFacts:
    """Extract the :class:`VisibilityFacts` of *content* (dict or object)."""
    return VisibilityFacts.from_content(content)


def _author_matches(actor: HasRole | None, author_id: Any) -> bool:
    if actor is None or not isinstance(author_id, str):
        return False
    return getattr(actor, "user_id", None) == author_id


def is_visible_to(
    actor: HasRole | None,
    content: Any,
    *,
    scope: VisibilityScope | str = VisibilityScope.PUBLIC,
    now: Any = None,
) -> bool:
    """The single visibility predicate of the platform (INV-visibility-predicate).

    ``actor`` is ``None`` for anonymous visitors, otherwise anything exposing
    ``user_id`` + ``role`` (a :class:`~blacknote.contracts.auth_guard.Principal`
    in production).  ``content`` may be ``None`` (missing row) → always False.
    """
    if content is None:
        return False

    try:
        asked = scope if isinstance(scope, VisibilityScope) else VisibilityScope(str(scope).lower())
    except ValueError:
        asked = VisibilityScope.PUBLIC

    facts = content if isinstance(content, VisibilityFacts) else VisibilityFacts.from_content(content)

    status = _enum(facts.status, ContentStatus)
    moderation = _enum(facts.moderation, ModerationState)
    visibility = _enum(facts.visibility, Visibility)
    if moderation is None:
        moderation = ModerationState.VISIBLE
    if visibility is None:
        visibility = Visibility.PUBLIC
    if status == ContentStatus.DELETED:
        return False
    if status is None or status == UNKNOWN:
        return False
    if moderation == ModerationState.BLOCKED:
        # Blocked comments are never served; only the admin queue may peek.
        return asked == VisibilityScope.MODERATION and is_admin(actor)
    if moderation is None or moderation == UNKNOWN:
        return False

    owner = _author_matches(actor, facts.author_id)

    if asked == VisibilityScope.MODERATION:
        return is_admin(actor)

    if asked == VisibilityScope.SELF:
        if not owner:
            return False
        # Own management view: drafts / private / taken-down work stay listed
        # so the author can still edit or delete them.
        return status in (ContentStatus.DRAFT, ContentStatus.PUBLISHED)

    # ---- public surface (feed, note detail, search, comment list, profile) ---
    if moderation == ModerationState.UNLISTED:
        return False
    if status != ContentStatus.PUBLISHED:
        return False  # drafts (including the author's own) are not public here
    if visibility == Visibility.PRIVATE:
        return False
    if visibility is None or visibility == UNKNOWN:
        return False
    return True


def filter_visible(
    actor: HasRole | None,
    contents: Iterable[Any],
    *,
    scope: VisibilityScope | str = VisibilityScope.PUBLIC,
    now: Any = None,
) -> list[Any]:
    """Batch helper: keep only what :func:`is_visible_to` allows."""
    return [
        item
        for item in contents
        if is_visible_to(actor, item, scope=scope, now=now)
    ]


def require_visible(
    actor: HasRole | None,
    content: Any,
    *,
    scope: VisibilityScope | str = VisibilityScope.PUBLIC,
    not_found: ErrorCode | str = ErrorCode.NOTE_NOT_FOUND,
    now: Any = None,
) -> Any:
    """Return *content* or raise ``*_NOT_FOUND``.

    Existence is never disclosed for invisible content: a draft, an unlisted
    note and a missing row all produce the very same 404 payload.
    """
    if not is_visible_to(actor, content, scope=scope, now=now):
        raise ContractError(not_found)
    return content


def is_interactable(
    actor: HasRole | None,
    content: Any,
    *,
    now: Any = None,
) -> bool:
    """Comments/likes/favorites require the note to be public *and* live.

    Backs ``IF-note-exists-for-interaction.is_publicly_interactable``: an
    author cannot like/favourite/comment on their own draft, and a moderated
    (unlisted) or deleted note is never interactable — for admins too, since
    moderation actions operate on ids, not on the public surface.
    """
    return is_visible_to(actor, content, scope=VisibilityScope.PUBLIC, now=now)
