"""``mod-interaction`` — the interaction / discussion layer (likes, favourites, comments).

Public surface
--------------
:class:`InteractionService`
    the frozen ``IF-interaction-actions`` implementation (``like``, ``unlike``,
    ``favorite``, ``unfavorite``, ``comment``, ``list_comments``,
    ``list_favorites``). Business failures are raised as the platform
    :class:`~.contracts.errors.ContractError`, exactly like ``mod-note`` and
    ``mod-account``; the ``*_result`` variants hand internal callers an
    :class:`~.result.InteractionResult` that carries the actor, the idempotency
    flag and the post-write counters (and renders the very same envelope via
    ``result.envelope`` / ``result.to_dict()`` for the web layer).
:class:`~.repository.InMemoryInteractionStore` / :class:`~.repository.SQLiteInteractionStore`
    the two storage engines; both keep the relation write and the note counter
    update inside one transaction (``INV-concurrency-counters``).
:func:`~.repository.open_store`
    composition-root helper (``prefer="memory"`` or an explicit ``sqlite://``
    location; there is no implicit ``data/`` default here).
:class:`~.ports.InMemoryNoteDirectory`
    a self-contained note *fake* (existence port + summary projection) used by
    dev wiring and by this layer's own tests, so nothing has to import
    ``mod-note``.

Wiring example (what ``mod-web`` composes at start-up)
------------------------------------------------------
.. code-block:: python

    from blacknote.interaction import InteractionService, InMemoryNoteDirectory

    directory = InMemoryNoteDirectory([{"id": note_id, "author_id": author_id}])
    service = InteractionService(directory, sessions={token: principal})
    service.like(token, note_id).likes          # -> 1, again 1 on repeat

Dependency direction (``IF-deps``): interaction → contracts + note query port.
This package never imports ``blacknote.note`` / ``blacknote.account`` /
``blacknote.moderation``; the assertion is exercised in ``tests/test_interaction.py``.
"""

from __future__ import annotations

from .models import (
    ANONYMOUS_NICKNAME,
    CommentRecord,
    FavoriteRecord,
    LikeRecord,
    relation_key,
)
from .ports import (
    DirectoryNote,
    IdentityDirectory,
    InMemoryIdentityDirectory,
    InMemoryNoteDirectory,
    NoteDirectory,
    NoteExistencePort,
    NullIdentityDirectory,
    NullNoteDirectory,
    adapt_directory,
    adapt_note_port,
    adapt_sessions,
    authenticate,
    fixture_token,
)
from .queries import (
    assemble_threads,
    decode_cursor,
    encode_cursor,
    page_of,
    thread_of,
    visible_rows,
)
from .repository import (
    CounterSink,
    DictCounterSink,
    FavoriteWindow,
    InMemoryInteractionStore,
    InteractionStore,
    NoteRecordCounterSink,
    SQLiteInteractionStore,
    SqlCounterTableSink,
    SqlNoteColumnSink,
    Transition,
    open_store,
)
from .result import InteractionResult
from .service import (
    COMMENT_INPUT_SPEC,
    MAX_COMMENT_BODY_LENGTH,
    InteractionService,
)

__all__ = [
    # service
    "InteractionService",
    "COMMENT_INPUT_SPEC",
    "MAX_COMMENT_BODY_LENGTH",
    # results
    "InteractionResult",
    # models
    "ANONYMOUS_NICKNAME",
    "CommentRecord",
    "FavoriteRecord",
    "LikeRecord",
    "relation_key",
    # ports
    "NoteExistencePort",
    "NoteDirectory",
    "DirectoryNote",
    "InMemoryNoteDirectory",
    "NullNoteDirectory",
    "IdentityDirectory",
    "InMemoryIdentityDirectory",
    "NullIdentityDirectory",
    "adapt_note_port",
    "adapt_directory",
    "adapt_sessions",
    "authenticate",
    "fixture_token",
    # storage
    "InteractionStore",
    "InMemoryInteractionStore",
    "SQLiteInteractionStore",
    "CounterSink",
    "DictCounterSink",
    "NoteRecordCounterSink",
    "SqlNoteColumnSink",
    "SqlCounterTableSink",
    "Transition",
    "FavoriteWindow",
    "open_store",
    # read shaping
    "encode_cursor",
    "decode_cursor",
    "page_of",
    "visible_rows",
    "thread_of",
    "assemble_threads",
]

#: Layer identity, mirroring ``blacknote.note.__all__`` style metadata.
MODULE_ID = "mod-interaction"
