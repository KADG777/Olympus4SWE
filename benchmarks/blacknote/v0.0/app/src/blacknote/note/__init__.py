"""``mod-note`` — the note content layer of BlackNote.

Public surface (all signatures frozen in :mod:`blacknote.contracts.interfaces`):

* :class:`NoteService` — ``IF-note-crud`` + ``IF-note-feed`` +
  ``IF-note-exists-for-interaction``.
* :class:`~blacknote.note.models.NoteRecord` — the persisted row shape
  (``status`` ∈ ``draft`` / ``published`` / ``deleted``, where ``deleted`` is a
  soft-delete tombstone).
* :class:`~blacknote.note.repository.InMemoryNoteRepository` /
  :class:`~blacknote.note.repository.SQLiteNoteRepository` — storage ports.
* :class:`~blacknote.note.identity.InMemoryAccountDirectory` and
  :func:`~blacknote.note.identity.adapt_session_resolver` — the two identity
  ports the note layer needs from ``mod-account`` (duck typed on purpose: this
  package imports *only* ``blacknote.contracts`` and the standard library, never
  a sibling business module).

Errors are always :class:`~blacknote.contracts.errors.ContractError` instances
carrying the frozen ``ErrorEnvelope`` — see the module docstring of
:mod:`blacknote.note.service`.
"""

from __future__ import annotations

from .identity import (
    AccountDirectory,
    AccountServiceDirectory,
    InMemoryAccountDirectory,
    NullAccountDirectory,
    adapt_session_resolver,
    from_account_service,
)
from .models import ANONYMOUS_NICKNAME, NoteRecord
from .queries import (
    NoteFeedQuery,
    NoteFilters,
    coerce_feed_query,
    coerce_page_query,
    decode_cursor,
    encode_cursor,
)
from .repository import (
    InMemoryNoteRepository,
    NoteRepository,
    SQLiteNoteRepository,
    open_repository,
)
from .service import DEFAULT_SCAN_LIMIT, NoteService
from .validation import (
    MAX_IMAGES,
    MIN_TOPIC_TAGS,
    WRITABLE_STATUSES,
    check_image_urls,
    check_tags,
    require_topic,
    require_write_status,
    resolve_topic,
)

__all__ = [
    "NoteService",
    "NoteRecord",
    "NoteRepository",
    "InMemoryNoteRepository",
    "SQLiteNoteRepository",
    "open_repository",
    "NoteFeedQuery",
    "NoteFilters",
    "encode_cursor",
    "decode_cursor",
    "coerce_feed_query",
    "coerce_page_query",
    "AccountDirectory",
    "InMemoryAccountDirectory",
    "NullAccountDirectory",
    "AccountServiceDirectory",
    "from_account_service",
    "adapt_session_resolver",
    "ANONYMOUS_NICKNAME",
    "MAX_IMAGES",
    "MIN_TOPIC_TAGS",
    "WRITABLE_STATUSES",
    "check_tags",
    "check_image_urls",
    "require_topic",
    "require_write_status",
    "resolve_topic",
    "DEFAULT_SCAN_LIMIT",
]
