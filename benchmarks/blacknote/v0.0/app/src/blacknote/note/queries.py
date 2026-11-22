"""Feed/search query objects and opaque keyset cursors (``mod-note``).

The frozen contract only carries ``viewer_token / topic / author_id / limit /
cursor`` (:class:`blacknote.contracts.interfaces.FeedQuery`).  The DesignSpec
also asks for *tag* and *following_only* filtering, so this module extends the
contract type instead of redefining it:

* :class:`NoteFeedQuery` — ``FeedQuery`` + ``tag`` + ``following_only``.  It is
  a subclass, therefore :meth:`NoteService.list_feed` still satisfies the
  ``IF-note-feed`` signature and keeps honouring the contract validation.
* :class:`NoteFilters` — the repository-facing projection of a query.  It holds
  **no** visibility rules: repositories return candidates, the service filters
  them with ``contracts.visibility.is_visible_to`` (INV-visibility-predicate).
* cursor helpers — an opaque, URL-safe ``base64url(created_at|note_id)`` token
  implementing ``INV-time-encoding`` keyset pagination ("next load shows newer
  content" because new notes sort above an already issued cursor).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.ids import is_valid_id
from blacknote.contracts.interfaces import FeedQuery, PageQuery
from blacknote.contracts.timeutil import from_iso
from blacknote.contracts.validation import normalize_tag

__all__ = [
    "CURSOR_SEPARATOR",
    "NoteFeedQuery",
    "NoteFilters",
    "encode_cursor",
    "decode_cursor",
    "coerce_feed_query",
    "coerce_page_query",
]

CURSOR_SEPARATOR: Final[str] = "|"


@dataclass(frozen=True)
class NoteFeedQuery(FeedQuery):
    """Contract feed query plus the two filters the web layer needs."""

    #: any custom tag (the fixed topic filter lives on ``FeedQuery.topic``).
    tag: str | None = None
    #: restrict the feed to accounts the viewer follows.
    following_only: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tag:
            object.__setattr__(self, "tag", normalize_tag(self.tag))
        object.__setattr__(self, "following_only", bool(self.following_only))

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        if self.tag:
            payload["tag"] = self.tag
        if self.following_only:
            payload["following_only"] = True
        return payload

    def with_filters(
        self,
        *,
        author_id: str | None = None,
        following_only: bool | None = None,
    ) -> "NoteFeedQuery":
        """Return a copy with a few fields swapped (used for the following feed)."""
        changes: dict[str, Any] = {}
        if author_id is not None:
            changes["author_id"] = author_id
        if following_only is not None:
            changes["following_only"] = following_only
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class NoteFilters:
    """Repository-facing selection criteria (ordering + keyset window)."""

    topic: str | None = None
    tag: str | None = None
    author_id: str | None = None
    #: whitelist of author ids (``following_only``); ``None`` = no restriction,
    #: an empty tuple matches nothing.
    author_ids: tuple[str, ...] | None = None
    keyword: str | None = None
    #: keyset window: only rows strictly older than ``(before_created_at,
    #: before_id)`` in ``created_at DESC`` order.
    before_created_at: str | None = None
    before_id: str | None = None

    def __post_init__(self) -> None:
        if self.author_ids is not None:
            object.__setattr__(self, "author_ids", tuple(str(item) for item in self.author_ids))
        for name in ("before_created_at",):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _canonical(value))

    @property
    def is_empty_selector(self) -> bool:
        """True when the whitelist makes the query unsatisfiable."""
        return self.author_ids is not None and not self.author_ids

    @property
    def tokens(self) -> tuple[str, ...]:
        """Search terms of :attr:`keyword` (all of them must hit — AND logic).

        Splitting the keyword is what makes ``"edited body"`` match a note whose
        body reads ``"edited **body**"``: the search surface is per word, not per
        exact phrase.
        """
        if not self.keyword:
            return ()
        return tuple(term for term in (part.casefold() for part in str(self.keyword).split())
                     if term)

    def after(self, created_at: str | None, note_id: str | None) -> "NoteFilters":
        """Return the same filters moved to a newer keyset window."""
        return replace(self, before_created_at=_canonical(created_at) if created_at else None,
                       before_id=note_id)


def _canonical(value: str) -> str:
    try:
        from blacknote.contracts.timeutil import to_iso

        return to_iso(from_iso(value, field_name="cursor"))
    except ContractError:
        raise ContractError.validation(
            [FieldError("cursor", "invalid_cursor", "cursor payload is malformed")]
        ) from None


def encode_cursor(created_at: str, note_id: str) -> str:
    """Opaque forward-only pagination token for a ``(created_at, id)`` keyset."""
    raw = f"{created_at}{CURSOR_SEPARATOR}{note_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: Any) -> tuple[str, str]:
    """Inverse of :func:`encode_cursor` → ``(created_at, note_id)``.

    A tampered / expired-format cursor is a client mistake, therefore it is
    reported as ``VALIDATION_ERROR`` (the only error code frozen for
    ``IF-note-feed``).
    """
    if value is None:
        raise ContractError.validation([FieldError("cursor", "required", "cursor is required")])
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii", "ignore")
    if not isinstance(value, str) or not value.strip():
        raise ContractError.validation([FieldError("cursor", "invalid_cursor",
                                                   "cursor must be a non-empty string")])
    token = value.strip()
    padding = "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(token + padding).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ContractError.validation(
            [FieldError("cursor", "invalid_cursor", "cursor payload is malformed")]
        ) from None
    created_at, separator, note_id = decoded.partition(CURSOR_SEPARATOR)
    if not separator or not created_at or not is_valid_id(note_id):
        raise ContractError.validation(
            [FieldError("cursor", "invalid_cursor", "cursor payload is malformed")]
        )
    return _canonical(created_at), note_id


_QUERY_FIELDS: Final[frozenset[str]] = frozenset(
    {"viewer_token", "topic", "author_id", "limit", "cursor", "tag", "following_only"}
)


def _construct(kind: Any, kwargs: Mapping[str, Any]) -> Any:
    """Build a query object, normalising plumbing errors into the envelope.

    The frozen dataclasses validate their own fields and raise
    :class:`ContractError`; a mapping coming off an HTTP form can however carry
    ``limit="ten"`` and blow up with a bare ``TypeError``.  The web layer must
    still see ``VALIDATION_ERROR``, so the plumbing error is translated here
    (never swallowed).
    """
    try:
        return kind(**kwargs)
    except ContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise ContractError.validation(
            [FieldError("query", "invalid_type", f"malformed {kind.__name__}: {exc}")]
        ) from None


def coerce_feed_query(query: FeedQuery | Mapping[str, Any] | None) -> NoteFeedQuery:
    """Accept the contract type, a web form mapping, or ``None``.

    Web handlers routinely hold a ``dict`` of query parameters; converting it
    here keeps ``VALIDATION_ERROR`` semantics identical to the frozen dataclass.
    """
    if isinstance(query, NoteFeedQuery):
        return query
    if isinstance(query, FeedQuery):
        return NoteFeedQuery(
            viewer_token=query.viewer_token,
            topic=query.topic,
            author_id=query.author_id,
            limit=query.limit,
            cursor=query.cursor,
        )
    if query is None:
        return NoteFeedQuery()
    if isinstance(query, Mapping):
        payload = {key: value for key, value in query.items() if key in _QUERY_FIELDS}
        unknown = sorted(set(query) - _QUERY_FIELDS)
        if unknown:
            raise ContractError.validation(
                [FieldError(name, "unknown_filter", "unsupported feed filter") for name in unknown]
            )
        return _construct(NoteFeedQuery, {key: value for key, value in payload.items()
                                        if value is not None})
    raise ContractError.validation(
        [FieldError("query", "invalid_type", "expected a FeedQuery or a mapping")]
    )


def coerce_page_query(query: PageQuery | Mapping[str, Any] | None) -> PageQuery:
    """Same tolerance for the search endpoint's pagination argument."""
    if isinstance(query, PageQuery):
        return query
    if query is None:
        return PageQuery()
    if isinstance(query, Mapping):
        unknown = sorted(set(query) - {"limit", "cursor"})
        if unknown:
            raise ContractError.validation(
                [FieldError(name, "unknown_filter", "unsupported pagination field")
                 for name in unknown]
            )
        payload = {key: value for key, value in query.items() if value is not None}
        return _construct(PageQuery, payload)
    raise ContractError.validation(
        [FieldError("query", "invalid_type", "expected a PageQuery or a mapping")]
    )


def normalize_authors(ids: Iterable[str] | None) -> tuple[str, ...]:
    """Defensive copy of an author whitelist (dropping non canonical ids)."""
    return tuple(str(item) for item in (ids or ()) if is_valid_id(str(item)))
