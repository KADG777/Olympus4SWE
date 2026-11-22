"""``InteractionService`` — the ``IF-interaction-actions`` use cases.

Method-by-method contract compliance
------------------------------------
``like`` / ``favorite``
    ``authenticate`` first (``R4-AC3``: an anonymous request raises
    ``AUTH_REQUIRED`` **before** any store call), then the injected note
    existence port decides whether the note may be interacted with
    (``INV-auth-session`` / ``INV-visibility-single-source``; otherwise
    ``NOTE_NOT_FOUND``).  The relation insert and the ``+1`` counter delta then
    happen inside one :class:`~.repository.InteractionStore` transaction; a
    duplicate pair is rejected by the ``(user_id, note_id)`` uniqueness, so the
    counter moves at most once per user.

``unlike`` / ``unfavorite``
    Same order, opposite delta.  The store reports whether a row actually
    disappeared, so a second cancellation applies no ``-1``; the cache itself is
    clamped at zero and can never go negative.

``comment``
    ``AUTH_REQUIRED`` → ``check_fields(..., "comment_input")`` (non-empty body,
    ``<= LIMITS["comment_body"]["max"]``, well formed ``parent_comment_id`` — all
    ``VALIDATION_ERROR``) → note existence → parent must exist *on the same note*
    and must itself be visible (``COMMENT_NOT_FOUND``) → one transactional insert
    plus ``+1`` on ``comment_count``.

``list_comments``
    Anonymous read of the note's comments, chronological (``created_at ASC``),
    assembled into threads; comments whose ancestry contains a blocked (or
    deleted) comment are pruned together with their whole subtree using the
    shared contract predicate.

``list_favorites``
    Authenticated keyset pagination over the caller's favourites, newest first,
    projected through the note port into ``NoteSummary`` rows whose counters come
    from this module's cache inside the same read — two views, one snapshot.

Two shapes, one vocabulary
------------------------------------
The frozen methods above raise ``ContractError``.  Every ``*_result`` sibling
(:meth:`like_result`, :meth:`comment_result`, :meth:`delete_comment_result`, …)
runs the *same* code through :meth:`_attempt` and returns an
:class:`~.result.InteractionResult` instead: business failures become a failure
result carrying the identical envelope, successes add the actor, the
idempotency flag and the post-write counters.  Non-contract exceptions are not
folded — a broken note dependency must stay a server error.

Field limits are *not* re-declared here: the frozen
``blacknote.contracts.validation.LIMITS`` bucket ``comment_body`` is the single
source of truth (``INV-field-validation``), so this layer can never disagree
with the contract table (the design brief quotes 2000 characters while the
frozen contract table says 500 — the contract wins, and the value is read, not
hard-coded, so updating the contract updates this module).

Dependencies
------------
Only ``blacknote.contracts`` and this package are imported.  ``notes`` is the
frozen ``NoteExistenceApi`` (anything with those two methods is accepted through
:func:`~.ports.adapt_note_port`).  Nothing here imports ``mod-note``,
``mod-account`` or ``mod-moderation``, and nothing registers a callback on the
note layer: the dependency direction is strictly interaction → note query
interface (``IF-deps``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final, TypeVar

from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id, new_id, normalize_id
from blacknote.contracts.interfaces import (
    CommentThread,
    CommentView,
    Counters,
    NoteSummary,
    Page,
    PageQuery,
    Principal,
)
from blacknote.contracts.timeutil import from_iso, now_iso, to_iso
from blacknote.contracts.validation import LIMITS, check_fields, clean_text
from blacknote.contracts.visibility import ContentStatus, ModerationState, is_visible_to

from .models import CommentRecord
from .ports import (
    IdentityDirectory,
    NoteDirectory,
    NullIdentityDirectory,
    adapt_directory,
    adapt_note_port,
    adapt_sessions,
    authenticate as authenticate_token,
)
from .queries import (
    assemble_threads,
    checked_note_id,
    clamp_limit,
    coerce_page_query,
    decode_cursor,
    encode_cursor,
    page_of,
    with_counters,
)
from .repository import FavoriteWindow, InteractionStore, InMemoryInteractionStore, Transition
from .result import InteractionResult

#: Value produced by a use case (counters, a comment view, …).
R = TypeVar("R")

__all__ = ["InteractionService", "COMMENT_INPUT_SPEC", "MAX_COMMENT_BODY_LENGTH"]

#: Validation spec name inside ``contracts.validation.SCHEMAS``.
COMMENT_INPUT_SPEC: Final[str] = "comment_input"

#: Read from the contract table — never re-declared (``INV-field-validation``).
MAX_COMMENT_BODY_LENGTH: Final[int] = int(LIMITS["comment_body"]["max"])

#: Hard ceiling for internal pagination loops (corrupt-cursor guard).
_MAX_PAGES: Final[int] = 64


class InteractionService:
    """Implements ``blacknote.contracts.interfaces.InteractionApi``."""

    def __init__(
        self,
        notes: Any = None,
        *,
        store: InteractionStore | None = None,
        sessions: Any = None,
        identity: Any = None,
        directory: Any = None,
        settings: Any = None,
        page_size: int = 20,
        clock: Any = None,
    ) -> None:
        if notes is not None and directory is not None:
            raise ContractError(
                ErrorCode.CONFIG_ERROR,
                "pass a single note dependency (NoteExistenceApi or a directory)",
                {"got": ["notes", "directory"]},
            )
        source = notes if notes is not None else directory
        if source is None:
            # Fail closed, loudly: without the note port nothing is publicly
            # interactable, so a mis-wired composition root cannot silently
            # accept interactions on notes that do not exist.
            raise ContractError(
                ErrorCode.CONFIG_ERROR,
                "InteractionService requires the note existence port (NoteExistenceApi)",
            )
        self._notes: NoteDirectory = adapt_note_port(source)
        self._store: InteractionStore = store if store is not None else InMemoryInteractionStore()
        self._sessions = adapt_sessions(sessions)
        identity_source = identity
        if identity_source is None and hasattr(directory, "nickname_of"):
            identity_source = directory
        self._identity: IdentityDirectory = (
            adapt_directory(identity_source)
            if identity_source is not None
            else NullIdentityDirectory()
        )
        self._settings = settings
        self._page_size = clamp_limit(page_size)
        self._clock = clock if callable(clock) else (lambda: clock or now_iso())

    # ------------------------------------------------------------------ #
    # introspection seams (composition root / web templates)
    # ------------------------------------------------------------------ #
    @property
    def store(self) -> InteractionStore:
        return self._store

    @property
    def note_port(self) -> NoteDirectory:
        """The adapted note dependency (``IF-note-exists-for-interaction``)."""
        return self._notes

    @property
    def settings(self) -> Any:
        """Injected platform settings, if the composition root passed any.

        Read-only diagnostics only: storage decisions are never derived from
        settings here, the store is always wired explicitly (``open_store``).
        """
        return self._settings

    def is_publicly_interactable(self, note_id: Any) -> bool:
        """Pure delegation to the note port — no local visibility rules."""
        return bool(self._notes.interactable(str(note_id or "")))

    def note_author(self, note_id: Any) -> str | None:
        return self._notes.author_id(str(note_id or ""))

    # ------------------------------------------------------------------ #
    # IF-interaction-actions: likes / favourites
    # ------------------------------------------------------------------ #
    def like(self, token: str, note_id: str) -> Counters:
        return self.like_result(token, note_id).unwrap()

    def unlike(self, token: str, note_id: str) -> Counters:
        return self.unlike_result(token, note_id).unwrap()

    def favorite(self, token: str, note_id: str) -> Counters:
        return self.favorite_result(token, note_id).unwrap()

    def unfavorite(self, token: str, note_id: str) -> Counters:
        return self.unfavorite_result(token, note_id).unwrap()

    def like_result(self, token: Any, note_id: Any) -> InteractionResult[Counters]:
        return self._attempt(self._relation, token, note_id, "like")

    def unlike_result(self, token: Any, note_id: Any) -> InteractionResult[Counters]:
        return self._attempt(self._relation, token, note_id, "unlike")

    def favorite_result(self, token: Any, note_id: Any) -> InteractionResult[Counters]:
        return self._attempt(self._relation, token, note_id, "favorite")

    def unfavorite_result(self, token: Any, note_id: Any) -> InteractionResult[Counters]:
        return self._attempt(self._relation, token, note_id, "unfavorite")

    def _relation(self, token: Any, note_id: Any, action: str) -> InteractionResult[Counters]:
        actor = self._authenticate(token)  # AUTH_REQUIRED, before any write
        note = self._require_interactable(note_id)
        transition = self._apply(action, actor.user_id, note)
        return InteractionResult(
            ok=True,
            value=transition.counters,
            actor_id=actor.user_id,
            counters=transition.counters,
            changed=transition.changed,
            entity_id=note,
            details={"note_id": note, "action": action, "actor_id": actor.user_id},
        )

    def _apply(self, action: str, user_id: str, note_id: str) -> Transition:
        stamp = self._stamp()
        handlers = {
            "like": lambda: self._store.add_like(user_id, note_id, created_at=stamp),
            "unlike": lambda: self._store.remove_like(user_id, note_id),
            "favorite": lambda: self._store.add_favorite(user_id, note_id, created_at=stamp),
            "unfavorite": lambda: self._store.remove_favorite(user_id, note_id),
        }
        handler = handlers.get(action)
        if handler is None:  # pragma: no cover - programmer error guard
            raise ContractError(
                ErrorCode.INTERNAL_ERROR, "unknown interaction action", {"action": [action]}
            )
        return handler()

    # ------------------------------------------------------------------ #
    # IF-interaction-actions: comments
    # ------------------------------------------------------------------ #
    def comment(
        self,
        token: str,
        note_id: str,
        parent_comment_id: str | None = None,
        body: str | None = None,
    ) -> CommentView:
        return self.comment_result(token, note_id, parent_comment_id, body).unwrap()

    def comment_result(
        self,
        token: Any,
        note_id: Any,
        parent_comment_id: Any = None,
        body: Any = None,
    ) -> InteractionResult[CommentView]:
        return self._attempt(self._comment, token, note_id, parent_comment_id, body)

    def _comment(
        self, token: Any, note_id: Any, parent_comment_id: Any, body: Any
    ) -> InteractionResult[CommentView]:
        actor = self._authenticate(token)  # AUTH_REQUIRED writes nothing (R4-AC3)
        # 1. contract validation (VALIDATION_ERROR, collected field errors)
        check_fields(
            {"body": body, "parent_comment_id": parent_comment_id}, COMMENT_INPUT_SPEC
        )
        clean_body = clean_text(body)
        parent_id = str(parent_comment_id).strip() if parent_comment_id is not None else None
        # 2. the note port owns existence + visibility (NOTE_NOT_FOUND)
        note = self._require_interactable(note_id)
        # 3. threaded replies: same note, visible parent (COMMENT_NOT_FOUND)
        parent = self._resolve_parent(note, parent_id)
        record = CommentRecord(
            id=new_id(),
            note_id=note,
            author_id=actor.user_id,
            body=clean_body,
            parent_comment_id=parent.id if parent is not None else None,
            status=ContentStatus.PUBLISHED,
            moderation=ModerationState.VISIBLE,
            created_at=self._stamp(),
        )
        transition = self._store.append_comment(record)
        return InteractionResult(
            ok=True,
            value=record.to_view(self._nickname(record.author_id)),
            actor_id=actor.user_id,
            counters=transition.counters,
            changed=transition.changed,
            entity_id=record.id,
            details={
                "note_id": note,
                "parent_comment_id": record.parent_comment_id,
                "thread_root_id": parent.id if parent is not None else record.id,
                "body_length": len(record.body),
            },
        )

    def list_comments(self, note_id: str) -> list[CommentThread]:
        return self.comment_threads(note_id)

    def delete_comment(
        self, token: Any, comment_id: Any, *, hard: bool = False
    ) -> Counters:
        """Caller removes *their own* comment. Returns the counter state.

        Internal seam above ``comment``: the frozen surface does not define a
        deletion entry point, but leaving a comment up forever is not a design.
        """
        return self.delete_comment_result(token, comment_id, hard=hard).unwrap()

    def delete_comment_result(
        self, token: Any, comment_id: Any, *, hard: bool = False
    ) -> InteractionResult[Counters]:
        """Non-raising view of :meth:`delete_comment`."""
        return self._attempt(self._delete, token, comment_id, hard=hard)

    def _delete(
        self, token: Any, comment_id: Any, *, hard: bool = False
    ) -> InteractionResult[Counters]:
        """Author-only removal, hidden (default) or purged (``hard=True``).

        ``R4-AC3`` ordering, again: authenticate first.  A comment that does not
        exist and a comment belonging to somebody else answer identically —
        ``COMMENT_NOT_FOUND`` — so this seam cannot be used to probe ids.
        Moderation does *not* go through here; it flags rows via
        :meth:`~blacknote.interaction.repository.InteractionStore.save_comment`,
        and flags are not deletions.
        """
        actor = self._authenticate(token)
        # A malformed id is answered like a missing one (see ``checked_note_id``):
        # this seam must not become a probe for the shape of other people's rows.
        try:
            wanted = normalize_id(comment_id)
        except ContractError:
            wanted = ""
        record = self._store.get_comment(wanted) if wanted else None
        if record is None or record.author_id != actor.user_id:
            raise ContractError(
                ErrorCode.COMMENT_NOT_FOUND,
                None,
                {"comment_id": [str(comment_id)]},
            )
        transition = self._store.delete_comment(record.id, hard=hard)
        return InteractionResult(
            ok=True,
            value=transition.counters,
            actor_id=actor.user_id,
            counters=transition.counters,
            changed=transition.changed,
            entity_id=record.id,
            details={"note_id": record.note_id, "hard": hard},
        )

    def comment_threads(
        self, note_id: Any, *, viewer: Any = None, include_hidden: bool = False
    ) -> list[CommentThread]:
        """Chronological threads with blocked subtrees pruned.

        ``viewer`` is an optional user id: the shared predicate lets an author
        keep seeing their own comment.  ``include_hidden`` (moderation back
        office) returns the raw tree without filtering.  The frozen
        ``list_comments(note_id)`` entry point stays anonymous.
        """
        note = self._require_interactable(note_id)
        rows = self._store.comments_for_note(note)
        threads = assemble_threads(rows, viewer=viewer, prune=not include_hidden)
        return self._project(threads)

    def get_comment(self, comment_id: Any) -> CommentRecord | None:
        """Raw stored comment (adapter seam for ``mod-moderation``)."""
        if not is_valid_id(comment_id):
            return None
        return self._store.get_comment(str(comment_id))

    def comment_note_id(self, comment_id: Any) -> str | None:
        record = self.get_comment(comment_id)
        return record.note_id if record is not None else None

    def thread_of(self, comment_id: Any, *, viewer: Any = None) -> CommentThread | None:
        """The visible thread (root + replies) that contains *comment_id*."""
        record = self.get_comment(comment_id)
        if record is None:
            raise ContractError(
                ErrorCode.COMMENT_NOT_FOUND, None, {"comment_id": [str(comment_id)]}
            )
        rows = self._store.comments_for_note(record.note_id)
        for root, branch in assemble_threads(rows, viewer=viewer):
            if root.id == record.id or any(child.id == record.id for child in branch):
                return self._project([(root, branch)])[0]
        return None

    def comment_count(self, note_id: Any) -> int:
        return self.counters(note_id).comments

    # ------------------------------------------------------------------ #
    # IF-interaction-actions: favourites
    # ------------------------------------------------------------------ #
    def list_favorites(
        self, token: str, query: PageQuery | Mapping[str, Any] | None = None
    ) -> Page[NoteSummary]:
        return self.favorites(token, query)

    def favorites(
        self, token: Any, query: PageQuery | Mapping[str, Any] | None = None
    ) -> Page[NoteSummary]:
        actor = self._authenticate(token)
        page_query = coerce_page_query(query)
        limit = clamp_limit(page_query.limit, default=self._page_size)
        cursor = decode_cursor(page_query.cursor)
        window = self._store.favorite_page(actor.user_id, limit=limit, cursor=cursor)
        items: list[NoteSummary] = []
        for row in window.rows:
            summary = self._notes.summary(row.note_id)
            if summary is None:
                continue
            # counters come from this module's cache, inside the same read, so
            # the note view and the interaction view cannot disagree
            items.append(with_counters(summary, self._store.counters(row.note_id)))
        next_cursor = (
            encode_cursor(window.rows[-1].created_at, window.rows[-1].id)
            if window.has_more and window.rows
            else None
        )
        return page_of(items, limit=limit, cursor=next_cursor, total=window.total)

    def favorite_note_ids(self, token: Any) -> tuple[str, ...]:
        """All favourited note ids of the caller, newest first (no projection)."""
        actor = self._authenticate(token)
        out: list[str] = []
        cursor: tuple[str, str] | None = None
        for _page in range(_MAX_PAGES):
            window: FavoriteWindow = self._store.favorite_page(
                actor.user_id, limit=self._page_size, cursor=cursor
            )
            out.extend(row.note_id for row in window.rows)
            if not window.has_more or not window.rows:
                return tuple(out)
            cursor = (window.rows[-1].created_at, window.rows[-1].id)
        raise ContractError(  # pragma: no cover - runaway guard
            ErrorCode.INTERNAL_ERROR,
            "favourite collection exceeded the pagination safety limit",
            {"user_id": [actor.user_id]},
        )

    def liked_notes(self, user_id: Any) -> tuple[str, ...]:
        return self._store.liked_note_ids(str(user_id or ""))

    def is_liked(self, token: Any, note_id: Any) -> bool:
        actor = self._authenticate(token)
        return self._store.has_like(actor.user_id, self._require_interactable(note_id))

    def is_favorited(self, token: Any, note_id: Any) -> bool:
        actor = self._authenticate(token)
        return self._store.has_favorite(actor.user_id, self._require_interactable(note_id))

    def counters(self, note_id: Any) -> Counters:
        """The counter cache — the value any note view must display."""
        return self._store.counters(self._require_interactable(note_id))

    def derived_counters(self, note_id: Any) -> Counters:
        """Truth recomputed from the relation tables (drift detector)."""
        return self._store.derive_counters(self._require_interactable(note_id))

    def sync_counters(self, note_id: Any) -> Counters:
        """Repair the cache from the relations (ops tool, never implicit)."""
        return self._store.sync_counters(self._require_interactable(note_id))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _authenticate(self, token: Any) -> Principal:
        return authenticate_token(token, self._sessions)

    def _attempt(
        self, action: Callable[..., InteractionResult[R]], *args: Any, **kwargs: Any
    ) -> InteractionResult[R]:
        """Run a use case, folding a business failure into the result envelope.

        The ``*_result`` methods are the layer's non-raising view: they hand the
        composition root the very same ``ErrorEnvelope`` the frozen methods
        raise, so a handler can render ``result.to_dict()`` without try/except.
        Anything that is *not* a contract error keeps propagating — a broken
        note dependency is a server bug, not a user error, and must never be
        turned into a ``400``.
        """
        try:
            return action(*args, **kwargs)
        except ContractError as exc:
            return InteractionResult.from_error(exc)

    def _require_interactable(self, note_id: Any) -> str:
        note = checked_note_id(note_id)
        if not self._notes.interactable(note):
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note})
        return note

    def _resolve_parent(self, note_id: str, parent_comment_id: str | None) -> CommentRecord | None:
        if parent_comment_id is None:
            return None
        parent = self._store.get_comment(parent_comment_id)
        if parent is None or parent.note_id != note_id or not is_visible_to(None, parent):
            raise ContractError(
                ErrorCode.COMMENT_NOT_FOUND, None, {"parent_comment_id": [parent_comment_id]}
            )
        return parent

    def _nickname(self, user_id: str) -> str:
        return self._identity.nickname_of(str(user_id or ""))

    def _stamp(self) -> str:
        value = self._clock()
        if isinstance(value, str):
            # canonicalise custom clocks ("2026-01-01T00:00:00" → "...000000Z")
            return to_iso(from_iso(value, field_name="clock", assume_utc=True))
        return to_iso(value)

    def _project(
        self, threads: Iterable[tuple[CommentRecord, list[CommentRecord]]]
    ) -> list[CommentThread]:
        def view(record: CommentRecord) -> CommentView:
            return record.to_view(self._nickname(record.author_id))

        return [
            CommentThread(comment=view(root), replies=tuple(view(child) for child in branch))
            for root, branch in threads
        ]
