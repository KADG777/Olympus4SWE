"""Frozen interface contracts (Wave 0 deliverable).

Every data structure and every call signature used by the account, note,
interaction and moderation layers is declared **here and only here**, so those
four layers can be implemented in parallel without renegotiating boundaries.

The signatures mirror the DesignSpec interface list one-to-one:

``IF-account-auth``            :class:`AccountAuthApi`
``IF-account-profile``         :class:`AccountProfileApi`
``IF-note-crud``               :class:`NoteCrudApi`
``IF-note-feed``               :class:`NoteFeedApi`
``IF-note-exists-for-interaction`` :class:`NoteExistenceApi`
``IF-interaction-actions``     :class:`InteractionApi`
``IF-moderation-actions``      :class:`ModerationApi`

Conventions locked by this module:

* ``token`` is an opaque session token (UUIDv4 string) — never an object.
  ``token_or_null`` (e.g. :meth:`NoteCrudApi.get_note`) accepts ``None`` for
  anonymous readers.
* Read/write of user content always exposes the author identity
  (``author_id`` + ``author_nickname``); every write operation requires a
  token (``AUTH_REQUIRED``) and may only be performed by the acting user.
* Every operation may return an :class:`~blacknote.contracts.errors.ErrorEnvelope`
  instead of its success payload; :data:`Result` expresses that union and
  :func:`~blacknote.contracts.errors.is_error` discriminates it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Final, Generic, Protocol, TypeVar, Union, runtime_checkable

from .auth_guard import Principal
from .errors import ErrorEnvelope, ErrorCode
from .ids import is_valid_id, normalize_id
from .timeutil import now_iso
from .validation import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    check_fields,
    clean_text,
    normalize_tag,
)
from .visibility import (
    ContentStatus,
    ModerationState,
    Visibility,
    VisibilityScope,
    is_visible_to,
)

__all__ = [
    "T",
    "Result",
    "Token",
    "UserId",
    "NoteId",
    "CommentId",
    "Timestamp",
    "Page",
    "Counters",
    "UserSummary",
    "SessionToken",
    "ProfilePatch",
    "ProfileView",
    "NoteInput",
    "NoteSummary",
    "NoteView",
    "CommentView",
    "CommentThread",
    "PageQuery",
    "FeedQuery",
    "serialize_payload",
    "AccountAuthApi",
    "AccountProfileApi",
    "AccountService",
    "NoteCrudApi",
    "NoteFeedApi",
    "NoteExistenceApi",
    "NoteService",
    "InteractionApi",
    "ModerationApi",
    "InterfaceSpec",
    "INTERFACES",
    "interface_errors",
    "interface_functions",
    "new_timestamp",
    "empty_page",
    "serialize_payload",
]

T = TypeVar("T")

Token = str
UserId = str
NoteId = str
CommentId = str
Timestamp = str

#: ``T | ErrorEnvelope`` — the union every contract function may return.
#: ``Result[NoteView]`` is the annotation used by the frozen Protocols below.
Result = Union[T, ErrorEnvelope]  # type: ignore[type-arg]


# --------------------------------------------------------------------------- #
# Generic containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Page(Generic[T]):
    """Cursor paginated collection (``IF-note-feed``, favorites)."""

    items: tuple[T, ...] = ()
    next_cursor: str | None = None
    total: int | None = None
    limit: int = DEFAULT_PAGE_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if self.total is not None and self.total < 0:
            raise ValueError("Page.total must be >= 0")
        if not (1 <= int(self.limit) <= MAX_PAGE_SIZE):
            raise ValueError(f"Page.limit must be within 1..{MAX_PAGE_SIZE}")

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: Any) -> Any:
        return self.items[index]

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None

    def map(self, transform: Any) -> "Page[Any]":
        return Page(tuple(transform(item) for item in self.items), self.next_cursor,
                    self.total, self.limit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [serialize_payload(item) for item in self.items],
            "next_cursor": self.next_cursor,
            "total": self.total,
            "limit": self.limit,
        }


def serialize_payload(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and not isinstance(value, type):
        return to_dict()
    if isinstance(value, (list, tuple)):
        return [serialize_payload(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): serialize_payload(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Counters:
    """Aggregate counters attached to a note (likes / favorites / comments)."""

    likes: int = 0
    favorites: int = 0
    comments: int = 0

    def __post_init__(self) -> None:
        for name in ("likes", "favorites", "comments"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Counters.{name} must be an integer")
            if value < 0:
                raise ValueError(f"Counters.{name} must not be negative")

    def delta(
        self,
        *,
        likes: int = 0,
        favorites: int = 0,
        comments: int = 0,
    ) -> "Counters":
        """Return a shifted copy — counters never go below zero."""
        return Counters(
            likes=max(0, self.likes + likes),
            favorites=max(0, self.favorites + favorites),
            comments=max(0, self.comments + comments),
        )

    def to_dict(self) -> dict[str, int]:
        return {"likes": self.likes, "favorites": self.favorites, "comments": self.comments}


@dataclass(frozen=True)
class UserSummary:
    """Minimum public identity of an account (``IF-account-auth.current_user``)."""

    id: str
    nickname: str
    role: str = "user"
    avatar_url: str | None = None
    created_at: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "user_id"))
        object.__setattr__(self, "nickname", clean_text(self.nickname) or "Anonymous")
        if self.role not in ("user", "admin"):
            raise ValueError("role must be 'user' or 'admin'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nickname": self.nickname,
            "role": self.role,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SessionToken:
    """Successful register/login payload (INV-auth-session)."""

    token: str
    user: UserSummary
    expires_at: Timestamp | None = None
    token_type: str = "Bearer"

    def __post_init__(self) -> None:
        if not is_valid_id(self.token):
            raise ValueError("session token must be a canonical UUIDv4 string")
        if not isinstance(self.user, UserSummary):
            raise ValueError("SessionToken.user must be a UserSummary")

    @property
    def user_id(self) -> str:
        return self.user.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "user": self.user.to_dict(),
        }


@dataclass(frozen=True)
class ProfilePatch:
    """PATCH payload for :class:`AccountProfileApi.update_profile`."""

    nickname: str | None = None
    bio: str | None = None
    avatar_url: str | None = None

    def __post_init__(self) -> None:
        check_fields(
            {
                "nickname": self.nickname,
                "bio": self.bio,
                "avatar_url": self.avatar_url,
            },
            "profile_patch",
            partial=True,
        )

    def is_empty(self) -> bool:
        return self.nickname is None and self.bio is None and self.avatar_url is None

    def to_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in
            {"nickname": self.nickname, "bio": self.bio, "avatar_url": self.avatar_url}.items()
            if value is not None
        }


@dataclass(frozen=True)
class ProfileView:
    """Public profile page payload."""

    id: str
    nickname: str
    bio: str = ""
    avatar_url: str | None = None
    role: str = "user"
    notes_count: int = 0
    likes_received: int = 0
    favorites_received: int = 0
    created_at: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "user_id"))
        for name in ("notes_count", "likes_received", "favorites_received"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"ProfileView.{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nickname": self.nickname,
            "bio": self.bio,
            "avatar_url": self.avatar_url,
            "role": self.role,
            "notes_count": self.notes_count,
            "likes_received": self.likes_received,
            "favorites_received": self.favorites_received,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class NoteInput:
    """Write payload for create/update note (image + text + tags, A2)."""

    title: str
    body: str
    tags: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    visibility: Visibility = Visibility.PUBLIC

    def __post_init__(self) -> None:
        tags = tuple(dict.fromkeys(normalize_tag(t) for t in (self.tags or ())))
        images = tuple(clean_text(u) for u in (self.image_urls or ()) if clean_text(u))
        visibility = self.visibility
        if isinstance(visibility, str):
            try:
                visibility = Visibility(visibility.strip().lower())
            except ValueError:
                visibility = Visibility.PUBLIC
        check_fields(
            {
                "title": self.title,
                "body": self.body,
                "tags": tags,
                "image_urls": images,
                "visibility": str(visibility),
            },
            "note_input",
        )
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "image_urls", images)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "title", clean_text(self.title, collapse=True))
        object.__setattr__(self, "body", self.body.strip("\r\n"))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NoteInput":
        """Build from a web form / JSON body (``VALIDATION_ERROR`` on failure)."""
        check_fields(payload, "note_input")
        return cls(
            title=payload["title"],
            body=payload["body"],
            tags=tuple(payload.get("tags") or ()),
            image_urls=tuple(payload.get("image_urls") or ()),
            visibility=payload.get("visibility", Visibility.PUBLIC),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "image_urls": list(self.image_urls),
            "visibility": str(self.visibility),
        }


@dataclass(frozen=True)
class NoteSummary:
    """Feed/search/favorites card (never carries the full body)."""

    id: str
    author_id: str
    author_nickname: str
    title: str
    excerpt: str = ""
    cover_image: str | None = None
    tags: tuple[str, ...] = ()
    counters: Counters = field(default_factory=Counters)
    created_at: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "note_id"))
        object.__setattr__(self, "author_id", normalize_id(self.author_id, "author_id"))
        object.__setattr__(self, "tags", tuple(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_nickname": self.author_nickname,
            "title": self.title,
            "excerpt": self.excerpt,
            "cover_image": self.cover_image,
            "tags": list(self.tags),
            "counters": self.counters.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class NoteView:
    """Full note (author identity + lifecycle flags + viewer-independent body)."""

    id: str
    author_id: str
    author_nickname: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    visibility: Visibility = Visibility.PUBLIC
    status: ContentStatus = ContentStatus.PUBLISHED
    moderation: ModerationState = ModerationState.VISIBLE
    counters: Counters = field(default_factory=Counters)
    created_at: Timestamp | None = None
    updated_at: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "note_id"))
        object.__setattr__(self, "author_id", normalize_id(self.author_id, "author_id"))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "image_urls", tuple(self.image_urls))
        object.__setattr__(self, "visibility", Visibility(str(self.visibility)))
        object.__setattr__(self, "status", ContentStatus(str(self.status)))
        object.__setattr__(self, "moderation", ModerationState(str(self.moderation)))

    # -- lifecycle helpers (rules live in visibility.is_visible_to) -----------
    def is_visible_to(
        self,
        actor: Principal | None,
        *,
        scope: VisibilityScope = VisibilityScope.PUBLIC,
    ) -> bool:
        return is_visible_to(actor, self, scope=scope)

    @property
    def is_publicly_available(self) -> bool:
        return self.status is ContentStatus.PUBLISHED and self.moderation is ModerationState.VISIBLE

    def summary(self) -> NoteSummary:
        """Project onto the lightweight feed card."""
        body = self.body.strip()
        return NoteSummary(
            id=self.id,
            author_id=self.author_id,
            author_nickname=self.author_nickname,
            title=self.title,
            excerpt=body[:160],
            cover_image=self.image_urls[0] if self.image_urls else None,
            tags=self.tags,
            counters=self.counters,
            created_at=self.created_at,
        )

    def with_counters(self, counters: Counters) -> "NoteView":
        return replace(self, counters=counters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_nickname": self.author_nickname,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "image_urls": list(self.image_urls),
            "visibility": str(self.visibility),
            "status": str(self.status),
            "moderation": str(self.moderation),
            "counters": self.counters.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class CommentView:
    """A single comment (Markdown-free plain text)."""

    id: str
    note_id: str
    author_id: str
    author_nickname: str
    body: str
    parent_comment_id: str | None = None
    status: ContentStatus = ContentStatus.PUBLISHED
    created_at: Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", normalize_id(self.id, "comment_id"))
        object.__setattr__(self, "note_id", normalize_id(self.note_id, "note_id"))
        object.__setattr__(self, "author_id", normalize_id(self.author_id, "author_id"))
        if self.parent_comment_id is not None:
            object.__setattr__(
                self, "parent_comment_id", normalize_id(self.parent_comment_id, "parent_comment_id")
            )
        object.__setattr__(self, "status", ContentStatus(str(self.status)))

    @property
    def is_reply(self) -> bool:
        return self.parent_comment_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "note_id": self.note_id,
            "author_id": self.author_id,
            "author_nickname": self.author_nickname,
            "body": self.body,
            "parent_comment_id": self.parent_comment_id,
            "status": str(self.status),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CommentThread:
    """Top level comment plus its (already visibility filtered) replies."""

    comment: CommentView
    replies: tuple[CommentView, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.comment, CommentView):
            raise ValueError("CommentThread.comment must be a CommentView")
        object.__setattr__(self, "replies", tuple(self.replies))

    def __len__(self) -> int:
        return 1 + len(self.replies)

    def to_dict(self) -> dict[str, Any]:
        return {"comment": self.comment.to_dict(), "replies": [r.to_dict() for r in self.replies]}


@dataclass(frozen=True)
class PageQuery:
    """Pagination parameters shared by feed / search / favorites."""

    limit: int = DEFAULT_PAGE_SIZE
    cursor: str | None = None

    def __post_init__(self) -> None:
        check_fields({"limit": self.limit, "cursor": self.cursor}, "page_query")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"limit": self.limit}
        if self.cursor:
            payload["cursor"] = self.cursor
        return payload


@dataclass(frozen=True)
class FeedQuery:
    """Feed selection: optional topic/author filter, always paginated.

    ``viewer_token`` may be ``None`` (anonymous browsing of the public feed).
    """

    viewer_token: str | None = None
    topic: str | None = None
    author_id: str | None = None
    limit: int = DEFAULT_PAGE_SIZE
    cursor: str | None = None

    def __post_init__(self) -> None:
        payload: dict[str, Any] = {"limit": self.limit, "cursor": self.cursor}
        if self.topic is not None:
            payload["topic"] = self.topic
        if self.author_id is not None:
            payload["author_id"] = self.author_id
        check_fields(payload, "feed_query", partial=True)
        if self.viewer_token is not None and not is_valid_id(self.viewer_token):
            raise ValueError("viewer_token must be a canonical UUIDv4 string")
        if self.topic:
            object.__setattr__(self, "topic", normalize_tag(self.topic))

    @property
    def is_anonymous(self) -> bool:
        return self.viewer_token is None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"limit": self.limit}
        if self.cursor:
            payload["cursor"] = self.cursor
        if self.topic:
            payload["topic"] = self.topic
        if self.author_id:
            payload["author_id"] = self.author_id
        return payload


# --------------------------------------------------------------------------- #
# Frozen service signatures (Protocols) — implementations live in later waves
# --------------------------------------------------------------------------- #
@runtime_checkable
class AccountAuthApi(Protocol):
    """``IF-account-auth`` — register / login / logout / current_user."""

    def register(self, email: str, password: str) -> "SessionToken | ErrorEnvelope": ...

    def login(self, email: str, password: str) -> "SessionToken | ErrorEnvelope": ...

    def logout(self, token: str) -> "None | ErrorEnvelope": ...

    def current_user(self, token: str) -> "UserSummary | ErrorEnvelope": ...


@runtime_checkable
class AccountProfileApi(Protocol):
    """``IF-account-profile`` — public profile read + self PATCH."""

    def get_profile(self, user_id: str) -> "ProfileView | ErrorEnvelope": ...

    def update_profile(
        self, token: str, patch: ProfilePatch
    ) -> "ProfileView | ErrorEnvelope": ...


@runtime_checkable
class AccountService(AccountAuthApi, AccountProfileApi, Protocol):
    """Full account facade consumed by ``mod-web``."""


@runtime_checkable
class NoteCrudApi(Protocol):
    """``IF-note-crud`` — author-only writes, visibility-filtered reads."""

    def create_note(self, token: str, input: NoteInput) -> "NoteView | ErrorEnvelope": ...

    def update_note(
        self, token: str, note_id: str, input: NoteInput
    ) -> "NoteView | ErrorEnvelope": ...

    def delete_note(self, token: str, note_id: str) -> "None | ErrorEnvelope": ...

    def get_note(
        self, token_or_null: str | None, note_id: str
    ) -> "NoteView | ErrorEnvelope": ...


@runtime_checkable
class NoteFeedApi(Protocol):
    """``IF-note-feed`` — chronological / topic / keyword listings."""

    def list_feed(self, query: FeedQuery) -> "Page[NoteSummary] | ErrorEnvelope": ...

    def search_notes(
        self, keyword: str, query: PageQuery
    ) -> "Page[NoteSummary] | ErrorEnvelope": ...


@runtime_checkable
class NoteExistenceApi(Protocol):
    """``IF-note-exists-for-interaction`` — read-only port for interactions."""

    def is_publicly_interactable(self, note_id: str) -> bool: ...

    def get_note_author_id(self, note_id: str) -> "str | None": ...


@runtime_checkable
class NoteService(NoteCrudApi, NoteFeedApi, NoteExistenceApi, Protocol):
    """Full note facade consumed by ``mod-web`` and ``mod-interaction``."""


@runtime_checkable
class InteractionApi(Protocol):
    """``IF-interaction-actions`` — likes, favorites, comments (all write paths
    require a session token, INV-auth-session)."""

    def like(self, token: str, note_id: str) -> "Counters | ErrorEnvelope": ...

    def unlike(self, token: str, note_id: str) -> "Counters | ErrorEnvelope": ...

    def favorite(self, token: str, note_id: str) -> "Counters | ErrorEnvelope": ...

    def unfavorite(self, token: str, note_id: str) -> "Counters | ErrorEnvelope": ...

    def comment(
        self, token: str, note_id: str, parent_comment_id: "str | None", body: str
    ) -> "CommentView | ErrorEnvelope": ...

    def list_comments(self, note_id: str) -> "list[CommentThread] | ErrorEnvelope": ...

    def list_favorites(
        self, token: str, query: PageQuery
    ) -> "Page[NoteSummary] | ErrorEnvelope": ...


@runtime_checkable
class ModerationApi(Protocol):
    """``IF-moderation-actions`` — admin-only takedowns (MVP manual workflow)."""

    def unlist_note(self, admin_token: str, note_id: str) -> "None | ErrorEnvelope": ...

    def block_comment(self, admin_token: str, comment_id: str) -> "None | ErrorEnvelope": ...


@runtime_checkable
class VisibilityFilterApi(Protocol):
    """Ports that must reuse the single visibility predicate."""

    def is_visible_to(
        self,
        actor: "Principal | None",
        content: Any,
        *,
        scope: VisibilityScope = ...,
    ) -> bool: ...


# --------------------------------------------------------------------------- #
# Machine readable registry of the frozen contracts (review + web wiring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterfaceSpec:
    id: str
    title: str
    protocol: type
    functions: tuple[str, ...]
    errors: tuple[str, ...]
    producer: str
    consumers: tuple[str, ...]
    routes: tuple[tuple[str, str, str], ...] = ()  # (method, path, function)

    def has_error(self, code: str | ErrorCode) -> bool:
        return str(code) in self.errors


class ContractErrorUnknownCode(LookupError):
    """Raised at import time when the registry references an unknown code."""


def _spec(
    iid: str,
    title: str,
    protocol: type,
    functions: Sequence[str],
    errors: Sequence[str],
    producer: str,
    consumers: Sequence[str],
    routes: Sequence[tuple[str, str, str]] = (),
) -> InterfaceSpec:
    known = {str(member) for member in ErrorCode}
    unknown = [code for code in errors if code not in known]
    if unknown:  # defensive: keep the registry honest
        raise ContractErrorUnknownCode(f"{iid}: unknown error codes {unknown}")
    return InterfaceSpec(iid, title, protocol, tuple(functions), tuple(errors), producer,
                         tuple(consumers), tuple(routes))


INTERFACES: Final[Mapping[str, InterfaceSpec]] = MappingProxyType(
    {
        spec.id: spec
        for spec in (
            _spec(
                "IF-account-auth",
                "register / login / logout / current_user",
                AccountAuthApi,
                ("register", "login", "logout", "current_user"),
                ("EMAIL_TAKEN", "VALIDATION_ERROR", "INVALID_CREDENTIALS", "AUTH_REQUIRED"),
                "mod-account",
                ("mod-web",),
                (
                    ("POST", "/api/auth/register", "register"),
                    ("POST", "/api/auth/login", "login"),
                    ("POST", "/api/auth/logout", "logout"),
                    ("GET", "/api/auth/me", "current_user"),
                ),
            ),
            _spec(
                "IF-account-profile",
                "profile read / self update",
                AccountProfileApi,
                ("get_profile", "update_profile"),
                ("AUTH_REQUIRED", "PERMISSION_DENIED", "USER_NOT_FOUND", "VALIDATION_ERROR"),
                "mod-account",
                ("mod-web",),
                (
                    ("GET", "/api/users/{user_id}/profile", "get_profile"),
                    ("PATCH", "/api/me/profile", "update_profile"),
                ),
            ),
            _spec(
                "IF-note-crud",
                "note create / update / delete / detail",
                NoteCrudApi,
                ("create_note", "update_note", "delete_note", "get_note"),
                ("AUTH_REQUIRED", "PERMISSION_DENIED", "NOTE_NOT_FOUND", "VALIDATION_ERROR"),
                "mod-note",
                ("mod-web",),
                (
                    ("POST", "/api/notes", "create_note"),
                    ("PUT", "/api/notes/{note_id}", "update_note"),
                    ("DELETE", "/api/notes/{note_id}", "delete_note"),
                    ("GET", "/api/notes/{note_id}", "get_note"),
                ),
            ),
            _spec(
                "IF-note-feed",
                "feed listing / search",
                NoteFeedApi,
                ("list_feed", "search_notes"),
                ("VALIDATION_ERROR",),
                "mod-note",
                ("mod-web",),
                (
                    ("GET", "/api/feed", "list_feed"),
                    ("GET", "/api/search", "search_notes"),
                ),
            ),
            _spec(
                "IF-note-exists-for-interaction",
                "note existence port for the interaction layer",
                NoteExistenceApi,
                ("is_publicly_interactable", "get_note_author_id"),
                ("NOTE_NOT_FOUND",),
                "mod-note",
                ("mod-interaction",),
            ),
            _spec(
                "IF-interaction-actions",
                "like / favorite / comment / listings",
                InteractionApi,
                (
                    "like",
                    "unlike",
                    "favorite",
                    "unfavorite",
                    "comment",
                    "list_comments",
                    "list_favorites",
                ),
                (
                    "AUTH_REQUIRED",
                    "NOTE_NOT_FOUND",
                    "COMMENT_NOT_FOUND",
                    "VALIDATION_ERROR",
                ),
                "mod-interaction",
                ("mod-web",),
                (
                    ("POST", "/api/notes/{note_id}/like", "like"),
                    ("DELETE", "/api/notes/{note_id}/like", "unlike"),
                    ("POST", "/api/notes/{note_id}/favorite", "favorite"),
                    ("DELETE", "/api/notes/{note_id}/favorite", "unfavorite"),
                    ("POST", "/api/notes/{note_id}/comments", "comment"),
                    ("GET", "/api/notes/{note_id}/comments", "list_comments"),
                    ("GET", "/api/me/favorites", "list_favorites"),
                ),
            ),
            _spec(
                "IF-moderation-actions",
                "admin takedown of notes and comments",
                ModerationApi,
                ("unlist_note", "block_comment"),
                (
                    "AUTH_REQUIRED",
                    "PERMISSION_DENIED",
                    "NOTE_NOT_FOUND",
                    "COMMENT_NOT_FOUND",
                ),
                "mod-moderation",
                ("mod-web",),
                (
                    ("POST", "/api/admin/notes/{note_id}/unlist", "unlist_note"),
                    ("POST", "/api/admin/comments/{comment_id}/block", "block_comment"),
                ),
            ),
        )
    }
)


def interface_errors(interface_id: str) -> tuple[str, ...]:
    """Error codes frozen for *interface_id*."""
    try:
        return INTERFACES[interface_id].errors
    except KeyError:
        raise KeyError(f"unknown interface id: {interface_id}") from None


def interface_functions(interface_id: str) -> tuple[str, ...]:
    """Function names frozen for *interface_id*."""
    try:
        return INTERFACES[interface_id].functions
    except KeyError:
        raise KeyError(f"unknown interface id: {interface_id}") from None


def new_timestamp() -> Timestamp:  # convenience for fixtures/implementations
    """Canonical ``INV-time-encoding`` timestamp."""
    return now_iso()


def empty_page(limit: int = DEFAULT_PAGE_SIZE) -> Page[Any]:
    """Canonical empty result."""
    return Page((), None, 0, limit)
