"""Read-side shaping: cursors, thread assembly, pagination envelopes.

Visibility filtering lives here on purpose: ``list_comments`` is a read path of
``mod-interaction``, so the comment *subtree* of a blocked comment is pruned
here, while ``mod-moderation`` only ever flips the ``moderation`` flag (no
cascade updates, per the outbox rule in ``docs/ARCHITECTURE.md``).

The pruning rule is expressed with the frozen platform predicate
:func:`blacknote.contracts.visibility.is_visible_to` — this module contains no
private notion of "what is blocked".
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Final, TypeVar

from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id, normalize_id
from blacknote.contracts.interfaces import (
    Counters,
    NoteSummary,
    Page,
    PageQuery,
)
from blacknote.contracts.validation import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, limit_for
from blacknote.contracts.visibility import is_visible_to

from .models import CommentRecord

#: Lower page-size bound, read from the contract limit table.
MIN_PAGE_SIZE: Final[int] = int(limit_for("page_size")[0] or 1)

__all__ = [
    "CURSOR_SEPARATOR",
    "encode_cursor",
    "decode_cursor",
    "coerce_page_query",
    "page_of",
    "visible_rows",
    "thread_of",
    "assemble_threads",
    "summary_from_fields",
    "summary_from_mapping",
    "summary_from_note",
    "clamp_limit",
    "with_counters",
    "checked_note_id",
]

#: ``created_at|entity_id`` base64url — the same cursor shape ``mod-note`` uses.
CURSOR_SEPARATOR: Final[str] = "|"

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def encode_cursor(created_at: str, entity_id: str) -> str:
    raw = f"{created_at}{CURSOR_SEPARATOR}{entity_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: Any) -> tuple[str, str] | None:
    """Return ``(created_at, id)``; ``None`` for an absent cursor.

    A cursor the client tampered with is a client error, never a silent reset:
    the platform error vocabulary freezes ``VALIDATION_ERROR`` for this module.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        parts = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            padded = text + "=" * (-len(text) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ContractError(
                ErrorCode.VALIDATION_ERROR, "invalid cursor", {"cursor": ["invalid_cursor"]}
            ) from exc
        parts = decoded.split(CURSOR_SEPARATOR, maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ContractError(ErrorCode.VALIDATION_ERROR, "invalid cursor",
                            {"cursor": ["invalid_cursor"]})
    return parts[0], parts[1]


def _validated_cursor(value: Any) -> str | None:
    """Normalise a query cursor, rejecting tampered values early."""
    text = None if value is None else (str(value).strip() or None)
    if text is not None:
        decode_cursor(text)
    return text


def coerce_page_query(query: PageQuery | Mapping[str, Any] | None) -> PageQuery:
    if query is None:
        return PageQuery()
    if isinstance(query, PageQuery):
        return query
    if isinstance(query, Mapping):
        payload = dict(query)
        cursor = payload.get("cursor")
        return PageQuery(
            limit=int(payload.get("limit") or MIN_PAGE_SIZE),
            cursor=_validated_cursor(cursor),
        )
    raise ContractError(
        ErrorCode.VALIDATION_ERROR, "query must be a PageQuery or a mapping",
        {"query": ["invalid_query"]},
    )


def page_of(
    items: Iterable[T], *, limit: int, cursor: str | None, total: int | None = None
) -> Page[T]:
    materialised = tuple(items)
    safe_limit = min(max(int(limit), MIN_PAGE_SIZE), MAX_PAGE_SIZE)
    return Page(
        items=materialised,
        next_cursor=cursor,
        total=len(materialised) if total is None else max(0, int(total)),
        limit=safe_limit,
    )


# --------------------------------------------------------------------------- #
# comment threads
# --------------------------------------------------------------------------- #
def visible_rows(
    rows: Iterable[CommentRecord], *, viewer: str | None = None
) -> dict[str, CommentRecord]:
    """Keep the rows the *viewer* may see (shared predicate, no local rules)."""
    return {row.id: row for row in rows if is_visible_to(viewer, row)}


def thread_of(
    row: CommentRecord, by_id: Mapping[str, CommentRecord], visible: Mapping[str, CommentRecord]
) -> CommentRecord | None:
    """Topmost visible ancestor of *row*, or ``None`` when the row is hidden.

    Walking *up* is what prunes subtrees: if any ancestor is invisible (blocked
    or deleted), the whole branch disappears for every viewer — including the
    author, who keeps seeing their own comment only when its ancestors are fine.
    A parent that is absent from *by_id* (hard-deleted, or belonging to another
    note) demotes the child to a thread root instead of hiding it.
    """
    current = row
    seen: set[str] = {row.id}
    while True:
        parent_id = current.parent_comment_id
        if parent_id is None:
            return current
        if parent_id in seen:
            return None  # cyclic ancestry is never rendered
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return current
        if parent.id not in visible:
            return None
        current = parent


def assemble_threads(
    rows: Iterable[CommentRecord], *, viewer: str | None = None, prune: bool = True
) -> list[tuple[CommentRecord, list[CommentRecord]]]:
    """Chronological threads: roots ``created_at ASC``, replies depth-first.

    ``prune=False`` disables the visibility filter (moderation back-office
    views); otherwise invisible rows and their whole subtrees are dropped.
    Return shape is ``(root, replies)`` with ``replies`` in pre-order, so
    ancestry always renders before its descendants.
    """
    materialised = list(rows)
    by_id = {row.id: row for row in materialised}
    visible = visible_rows(materialised, viewer=viewer) if prune else {row.id: row for row in materialised}
    children: dict[str, list[CommentRecord]] = {}
    roots: dict[str, list[CommentRecord]] = {}
    for row in visible.values():
        root = thread_of(row, by_id, visible)
        if root is None:
            continue
        if root.id == row.id:
            roots.setdefault(row.id, []).append(row)
        else:
            children.setdefault(root.id, []).append(row)

    def key(record: CommentRecord) -> tuple[str, str]:
        return record.sort_key

    ordered_roots = sorted({row.id: row for row in _flatten(roots)}.values(), key=key)
    threads: list[tuple[CommentRecord, list[CommentRecord]]] = []
    visited: set[str] = set()

    for root in ordered_roots:
        branch: list[CommentRecord] = []

        def visit(node: CommentRecord) -> None:
            for child in sorted(children.get(node.id, ()), key=key):
                if child.id in visited:
                    continue  # a row can only belong to one rendered branch
                visited.add(child.id)
                branch.append(child)
                visit(child)

        visit(root)
        threads.append((root, branch))
    return threads


def _flatten(buckets: Mapping[str, list[CommentRecord]]) -> list[CommentRecord]:
    out: list[CommentRecord] = []
    for records in buckets.values():
        out.extend(records)
    return out


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def clamp_limit(value: Any, *, default: int = DEFAULT_PAGE_SIZE) -> int:
    """`limit` handling in the note-service shape (``mod-note`` parity)."""
    if value is None:
        return default
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ContractError(
            ErrorCode.VALIDATION_ERROR, "limit must be an integer", {"limit": ["invalid_limit"]}
        ) from None
    if limit < MIN_PAGE_SIZE:
        raise ContractError(
            ErrorCode.VALIDATION_ERROR, "limit must be >= 1", {"limit": ["invalid_limit"]}
        )
    return min(limit, MAX_PAGE_SIZE)


def _coerce_counters(value: Any) -> Counters | None:
    """Accept the counter shapes note dependencies actually expose."""
    if value is None:
        return None
    if isinstance(value, Counters):
        return value
    if isinstance(value, Mapping):
        return Counters(
            **{key: value[key] for key in ("likes", "favorites", "comments") if key in value}
        )
    return None


def with_counters(summary: NoteSummary, counters: Any) -> NoteSummary:
    """Overlay the interaction counter cache on a note summary (one snapshot).

    ``counters`` accepts a :class:`Counters` or the flat ``*_count`` column
    shape a note row may carry; the value shown by the favourites list is
    therefore literally the value the interaction layer just read.
    """
    payload = counters if isinstance(counters, Counters) else _coerce_counters(counters)
    if payload is None:
        raise ContractError(
            ErrorCode.INTERNAL_ERROR, "unsupported counters payload",
            {"counters": [type(counters).__name__]},
        )
    return replace(summary, counters=payload)


def checked_note_id(note_id: Any) -> str:
    """Malformed note ids are reported as absent (existence is not leaked)."""
    if not is_valid_id(note_id):
        raise ContractError(
            ErrorCode.NOTE_NOT_FOUND, None, {"note_id": ["invalid_or_missing_note_id"]}
        )
    return normalize_id(note_id)
