"""Note content service — ``mod-note`` (``IF-note-crud`` / ``IF-note-feed`` /
``IF-note-exists-for-interaction``).

:class:`NoteService` implements the three frozen protocols of the note domain on
top of an injected repository, session resolver and account directory.

Error handling convention
-------------------------
Every failure is signalled by raising
:class:`~blacknote.contracts.errors.ContractError`, whose
``to_envelope()`` *is* the frozen ``ErrorEnvelope`` — the contract documents
this as equivalent to returning the union member, and it is the only shape the
web layer may serialise.  (Note that constructing ``NoteInput`` / ``FeedQuery``
already raises ``ContractError``, so any caller must handle that exception type
regardless.)  The existence port for the interaction layer never raises: it
fails closed with ``False`` / ``None``.

Guarantees
----------
* writes require a session (``AUTH_REQUIRED``) and only ever act on the caller's
  own notes (``PERMISSION_DENIED`` for every other authenticated actor, admins
  included — admin takedowns belong to the moderation layer, which flips the
  ``moderation`` flag through :meth:`NoteService.apply_moderation_state`).
* soft delete: ``status=deleted`` tombstones stay in storage but are invisible
  everywhere (``NOTE_NOT_FOUND``).
* **every** read path funnels through the shared predicate
  :func:`blacknote.contracts.visibility.is_visible_to` — this module contains no
  local "may this actor see it" logic (scope selection only decides *which*
  contract scope asks the question).
* counters (likes / favorites / comments) are initialised to ``0`` and preserved;
  this layer never mutates them (``INV-concurrency-counters`` owns them in the
  interaction layer).
* timestamps are canonical UTC ISO-8601 with a ``Z`` suffix and feed ordering
  is ``created_at DESC`` keyset pagination via opaque cursors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Final

from blacknote.contracts.auth_guard import Principal, require_auth, resolve_session
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ContractError, ErrorCode, FieldError
from blacknote.contracts.ids import is_valid_id, new_id, require_id
from blacknote.contracts.interfaces import (
    Counters,
    FeedQuery,
    NoteInput,
    NoteSummary,
    NoteView,
    Page,
    PageQuery,
)
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.validation import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, check_fields
from blacknote.contracts.visibility import (
    ContentStatus,
    ModerationState,
    VisibilityScope,
    is_interactable,
    is_visible_to,
)

from .identity import AccountDirectory, NullAccountDirectory, adapt_session_resolver
from .models import ANONYMOUS_NICKNAME, NoteRecord
from .queries import (
    NoteFeedQuery,
    NoteFilters,
    coerce_feed_query,
    coerce_page_query,
    decode_cursor,
    encode_cursor,
    normalize_authors,
)
from .repository import InMemoryNoteRepository, NoteRepository
from .validation import check_image_urls, require_topic, require_write_status

__all__ = ["NoteService", "DEFAULT_SCAN_LIMIT"]

#: Safety bound on how many candidate rows one listing request may scan to apply
#: the visibility predicate and compute ``Page.total``.
DEFAULT_SCAN_LIMIT: Final[int] = 2000


def _first_not_none(*values: Any) -> Any:
    """First meaningful value (``None`` and empty strings are skipped)."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _clamp_limit(limit: Any, ceiling: int) -> int:
    """Reconcile a requested page size with the deployment configuration.

    Bounds are already enforced by the contract (``LIMITS["page_size"]``); this
    helper only keeps the effective value inside
    ``[1, min(LIMITS max, Settings.max_page_size)]`` and falls back to the
    contract default (``DEFAULT_PAGE_SIZE``, i.e. 20 — comfortably above the
    "每页 ≥ 10" feed requirement) when a caller passes something nonsensical.
    """
    upper = max(1, min(int(ceiling), MAX_PAGE_SIZE))
    try:
        requested = int(limit)
    except (TypeError, ValueError):
        return min(DEFAULT_PAGE_SIZE, upper)
    if requested < 1:
        return min(DEFAULT_PAGE_SIZE, upper)
    return min(requested, upper)



class NoteService:
    """Concrete implementation of the frozen ``NoteService`` protocol."""

    def __init__(
        self,
        repository: NoteRepository | None = None,
        *,
        sessions: object | None = None,
        directory: AccountDirectory | None = None,
        settings: Settings | None = None,
        clock: Callable[[], str] | None = None,
        scan_limit: int = DEFAULT_SCAN_LIMIT,
    ) -> None:
        """:param repository: storage port (defaults to an in-process store).
        :param sessions: session resolver port (``resolve``/``current_user``/callable).
        :param directory: identity port supplying nicknames and the follow graph.
        :param settings: frozen configuration (image size/type limits, page size).
        :param clock: injectable UTC clock, used by tests for deterministic feeds.
        """
        self._repo: NoteRepository = (
            repository if repository is not None else InMemoryNoteRepository()
        )
        self._settings = settings or Settings()
        self._resolver = adapt_session_resolver(sessions)
        self._directory: AccountDirectory = (
            directory if directory is not None else NullAccountDirectory()
        )
        self._clock = clock or now_iso
        self._scan_limit = max(int(scan_limit), 1)

    # ------------------------------------------------------------------ #
    # wiring helpers (used by mod-web / tests)
    # ------------------------------------------------------------------ #
    @property
    def repository(self) -> NoteRepository:
        return self._repo

    @property
    def settings(self) -> Settings:
        return self._settings

    def bind_sessions(self, sessions: object) -> "NoteService":
        """Wire (or re-wire) the session resolver port."""
        self._resolver = adapt_session_resolver(sessions)
        return self

    def bind_directory(self, directory: AccountDirectory) -> "NoteService":
        """Wire (or re-wire) the identity port."""
        self._directory = directory
        return self

    # ------------------------------------------------------------------ #
    # IF-note-crud
    # ------------------------------------------------------------------ #
    def create_note(
        self,
        token: str,
        input: NoteInput | Mapping[str, Any],
        *,
        status: ContentStatus | str | None = None,
    ) -> NoteView:
        """Publish (or draft) a note for the authenticated author.

        Raises ``AUTH_REQUIRED`` without touching storage, ``VALIDATION_ERROR``
        listing every offending field, and never assigns interaction counters.
        """
        actor = self._authenticate(token)
        note_input = self._coerce_input(input)
        requested_status = _first_not_none(status, getattr(note_input, "status", None))
        effective_status = require_write_status(requested_status)
        topic = require_topic(note_input.tags)
        image_urls = check_image_urls(note_input.image_urls, self._settings)
        stamp = self._stamp()
        record = NoteRecord(
            id=new_id(),
            author_id=actor.user_id,
            title=note_input.title,
            body=note_input.body,
            tags=note_input.tags,
            topic=topic,
            image_urls=image_urls,
            visibility=note_input.visibility,
            status=effective_status,
            moderation=ModerationState.VISIBLE,
            counters=Counters(),  # like/favorite/comment counts belong to mod-interaction
            created_at=stamp,
            updated_at=stamp,
        )
        self._repo.insert(record)
        return record.to_view(self._nickname(record.author_id))

    def update_note(
        self,
        token: str,
        note_id: str,
        input: NoteInput | Mapping[str, Any],
        *,
        status: ContentStatus | str | None = None,
    ) -> NoteView:
        """Author-only edit; ``PERMISSION_DENIED`` for anybody else.

        Counters and ``created_at`` survive untouched; a draft can be published
        by passing ``status="published"``.
        """
        actor = self._authenticate(token)
        record = self._require_record(note_id)
        self._require_author(actor, record)
        note_input = self._coerce_input(input)
        effective_status = require_write_status(
            _first_not_none(status, getattr(note_input, "status", None))
        )
        topic = require_topic(note_input.tags)
        image_urls = check_image_urls(note_input.image_urls, self._settings)
        updated = record.changed(
            title=note_input.title,
            body=note_input.body,
            tags=note_input.tags,
            topic=topic,
            image_urls=image_urls,
            visibility=note_input.visibility,
            status=effective_status,
            updated_at=self._stamp(),
            # counters deliberately *not* rebuilt: they belong to mod-interaction
        )
        self._repo.save(updated)
        return updated.to_view(self._nickname(updated.author_id))

    def delete_note(self, token: str, note_id: str) -> None:
        """Soft delete (``status=deleted``); tombstones are kept for integrity."""
        actor = self._authenticate(token)
        record = self._require_record(note_id)
        self._require_author(actor, record)
        self._repo.save(record.changed(status=ContentStatus.DELETED, updated_at=self._stamp()))
        return None

    def get_note(self, token_or_null: str | None, note_id: str) -> NoteView:
        """Detail read; invisible content (draft of another, unlisted, deleted,
        missing row) collapses into the very same ``NOTE_NOT_FOUND``."""
        actor = self._viewer(token_or_null)
        record = self._require_record(note_id)
        if not self._readable(actor, record, allow_self_view=True):
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        return record.to_view(self._nickname(record.author_id))

    # ------------------------------------------------------------------ #
    # IF-note-feed
    # ------------------------------------------------------------------ #
    def list_feed(self, query: FeedQuery | NoteFeedQuery | Mapping[str, Any] | None = None
                  ) -> Page[NoteSummary]:
        """Chronological (``created_at`` DESC) cursor paginated feed."""
        feed_query = coerce_feed_query(query)
        actor = self._viewer(feed_query.viewer_token)
        filters = NoteFilters(
            topic=feed_query.topic,
            tag=feed_query.tag,
            author_id=feed_query.author_id,
        )
        if feed_query.following_only:
            filters = replace(filters, author_ids=self._follow_ids(actor))
        if feed_query.cursor:
            filters = filters.after(*decode_cursor(feed_query.cursor))
        # A single-author listing doubles as the author's management view, where
        # their own drafts must stay reachable; the scope decision is delegated
        # to the contract predicate, which only grants SELF to the owner.
        return self._page(filters, limit=feed_query.limit, actor=actor,
                          allow_self_view=feed_query.author_id is not None)

    def search_notes(
        self,
        keyword: str,
        query: PageQuery | Mapping[str, Any] | None = None,
    ) -> Page[NoteSummary]:
        """Keyword search over ``title`` + ``body`` on the public surface.

        Every whitespace separated term must hit (AND); an empty result is a
        first class answer: ``items=()``, ``total=0``, ``next_cursor=None``.
        """
        page_query = coerce_page_query(query)
        check_fields(
            {"keyword": keyword, "limit": page_query.limit, "cursor": page_query.cursor},
            "search",
        )
        filters = NoteFilters(keyword=str(keyword).strip())
        if page_query.cursor:
            filters = filters.after(*decode_cursor(page_query.cursor))
        # ``IF-note-feed.search_notes`` carries no viewer, so search is strictly
        # the anonymous public surface: drafts/private/unlisted/deleted never hit.
        return self._page(filters, limit=page_query.limit, actor=None, allow_self_view=False)

    # ------------------------------------------------------------------ #
    # IF-note-exists-for-interaction (never raises: fails closed)
    # ------------------------------------------------------------------ #
    def is_publicly_interactable(self, note_id: str) -> bool:
        """True when likes/favorites/comments may target *note_id*."""
        record = self._quiet_get(note_id)
        if record is None:
            return False
        return is_interactable(None, record)

    def get_note_author_id(self, note_id: str) -> str | None:
        """Author of a *publicly available* note, else ``None``.

        Drafts, unlisted and deleted notes answer ``None`` so the interaction
        layer cannot leak (or act upon) content that is not on the public
        surface — it must call :meth:`is_publicly_interactable` anyway.
        """
        record = self._quiet_get(note_id)
        if record is None or not is_visible_to(None, record, scope=VisibilityScope.PUBLIC):
            return None
        return record.author_id

    # ------------------------------------------------------------------ #
    # seams used by mod-moderation / mod-web (not part of the frozen API)
    # ------------------------------------------------------------------ #
    def apply_moderation_state(self, note_id: str, state: ModerationState | str) -> NoteView:
        """Write the moderation flag only (``unlist_note`` support, INV-visibility).

        The note layer exposes the *primitive*; deciding who may moderate is the
        moderation layer's job and goes through ``contracts.roles.require_admin``.
        """
        record = self._require_record(note_id)
        target = state if isinstance(state, ModerationState) else ModerationState(str(state))
        if record.moderation is target:
            return record.to_view(self._nickname(record.author_id))
        updated = record.changed(moderation=target, updated_at=self._stamp())
        self._repo.save(updated)
        return updated.to_view(self._nickname(updated.author_id))

    def inspect_note_record(self, note_id: str) -> NoteRecord | None:
        """Storage probe for tests/ops — **not** a read surface.

        It deliberately bypasses the visibility filter so a maintainer can look
        at a tombstone or at a moderated row.  Nothing served to a user may use
        it: ``get_note`` / ``list_feed`` / ``search_notes`` / the existence port
        all decide through
        :func:`blacknote.contracts.visibility.is_visible_to`, and the only
        privileged write seam is :meth:`apply_moderation_state`.
        """
        if not is_valid_id(note_id):
            return None
        return self._repo.get(note_id)

    def list_author_notes(
        self,
        token_or_null: str | None,
        author_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> Page[NoteSummary]:
        """Profile listing (own drafts included for the author themself)."""
        return self.list_feed(
            NoteFeedQuery(viewer_token=token_or_null, author_id=require_id(author_id, "author_id"),
                          limit=limit, cursor=cursor)
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _page(
        self,
        filters: NoteFilters,
        *,
        limit: int,
        actor: Principal | None,
        allow_self_view: bool,
    ) -> Page[NoteSummary]:
        effective_limit = _clamp_limit(limit, self._settings.max_page_size)
        collected: list[NoteRecord] = []
        total = 0
        scanned = 0
        truncated = False
        for record in self._repo.stream(filters):
            if scanned >= self._scan_limit:
                truncated = True
                break
            scanned += 1
            if not self._readable(actor, record, allow_self_view=allow_self_view):
                continue
            total += 1
            if len(collected) < effective_limit:
                collected.append(record)
        has_more = truncated or total > effective_limit
        next_cursor = (
            encode_cursor(collected[-1].created_at, collected[-1].id)
            if has_more and collected
            else None
        )
        summaries = tuple(
            record.to_summary(self._nickname(record.author_id)) for record in collected
        )
        return Page(
            items=summaries,
            next_cursor=next_cursor,
            total=total,
            limit=effective_limit,
        )

    @staticmethod
    def _readable(
        actor: Principal | None,
        record: NoteRecord | NoteView,
        *,
        allow_self_view: bool,
    ) -> bool:
        """Ask the single shared predicate — no visibility rule lives here.

        ``SELF`` is only ever consulted as an *additional* question, and the
        contract grants it exclusively to the owner, so a stranger can never see
        a draft or a taken-down note through this path.
        """
        if allow_self_view and is_visible_to(actor, record, scope=VisibilityScope.SELF):
            return True
        return is_visible_to(actor, record, scope=VisibilityScope.PUBLIC)

    def _authenticate(self, token: str | None) -> Principal:
        """Write gate (INV-auth-session): raises ``AUTH_REQUIRED``."""
        return require_auth(token, self._resolver)

    def _viewer(self, token_or_null: str | None) -> Principal | None:
        """Read gate: an unreadable token simply means "anonymous reader"."""
        if token_or_null is None:
            return None
        return resolve_session(token_or_null, self._resolver)

    def _require_record(self, note_id: str) -> NoteRecord:
        require_id(note_id, "note_id")  # VALIDATION_ERROR on a malformed id
        record = self._repo.get(note_id)
        if record is None or record.is_tombstone:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        return record

    @staticmethod
    def _require_author(actor: Principal, record: NoteRecord) -> None:
        """Ownership rule of ``IF-note-crud`` (admins moderate, they do not edit)."""
        if actor.user_id != record.author_id:
            raise ContractError.permission_denied("only the author may modify this note",
                                                 note_id=record.id)

    def _quiet_get(self, note_id: str) -> NoteRecord | None:
        if not is_valid_id(note_id):
            return None
        try:
            record = self._repo.get(note_id)
        except ContractError:  # pragma: no cover - storage failure leaks as absent
            return None
        return record

    def _nickname(self, user_id: str) -> str:
        try:
            nickname = self._directory.nickname_of(user_id)
        except ContractError:
            nickname = None
        return str(nickname) if nickname else ANONYMOUS_NICKNAME

    def _follow_ids(self, actor: Principal | None) -> tuple[str, ...]:
        if actor is None:
            raise ContractError.validation(
                [FieldError("following_only", "viewer_required",
                            "a viewer_token is required to list followed accounts")]
            )
        return normalize_authors(self._directory.following_ids(actor.user_id))

    def _stamp(self) -> str:
        return self._clock()

    @staticmethod
    def _coerce_input(input: NoteInput | Mapping[str, Any] | Any) -> NoteInput:
        if isinstance(input, NoteInput):
            return input
        if isinstance(input, Mapping):
            return NoteInput.from_payload(dict(input))  # VALIDATION_ERROR on failure
        raise ContractError.validation(
            [FieldError("input", "invalid_type", "expected a NoteInput or a mapping")]
        )

    @staticmethod
    def is_row_publicly_visible(actor: Principal | None, record: NoteRecord | NoteView) -> bool:
        """Expose the shared predicate for downstream layers that list notes."""
        return is_visible_to(actor, record, scope=VisibilityScope.PUBLIC)

