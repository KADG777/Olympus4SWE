"""Ports and adapters of ``mod-interaction``.

The interaction layer owns *no* note table.  Notes live in ``mod-note``, so the
only thing this layer may ask about a note is the frozen
``IF-note-exists-for-interaction`` pair:

``is_publicly_interactable(note_id) -> bool``
``get_note_author_id(note_id) -> str | None``

That is a strict one-way dependency (``mod-interaction`` → ``mod-note``); this
layer never calls a note *write* API and never imports a sibling business
module — the same discipline ``mod-note`` applies towards ``mod-account`` so
the Wave-1 modules can be built in parallel.  Everything is duck-typed, so
:func:`adapt_note_port` accepts:

* the note service itself (both frozen methods, optionally ``get_note`` /
  ``inspect_note_record`` for the favorite listing projection);
* a note repository exposing ``get(note_id)`` (rows are judged by the shared
  contract predicate, never by a local rule);
* a plain ``{note_id: row}`` mapping (fixtures, dev wiring).

Two further ports are injected at start-up by ``mod-web``:

:class:`IdentityDirectory`
    ``nickname_of(user_id)`` for ``CommentView.author_nickname``.
Session resolver adaptation
    anything able to turn a bearer token into a
    :class:`~blacknote.contracts.auth_guard.Principal`; the actual gating is
    performed by :func:`blacknote.contracts.auth_guard.require_auth`
    (``INV-auth-session``), this module only adapts the shape of the dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from blacknote.contracts.auth_guard import Principal, require_auth
from blacknote.contracts.errors import ContractError, ErrorCode, ErrorEnvelope, is_error
from blacknote.contracts.ids import is_valid_id, new_id
from blacknote.contracts.interfaces import NoteSummary, NoteView
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.validation import clean_text
from blacknote.contracts.visibility import (
    ContentStatus,
    ModerationState,
    Visibility,
    VisibilityScope,
    is_interactable,
    is_visible_to,
)

__all__ = [
    "NoteExistencePort",
    "NoteDirectory",
    "NullNoteDirectory",
    "InMemoryNoteDirectory",
    "adapt_note_port",
    "IdentityDirectory",
    "NullIdentityDirectory",
    "InMemoryIdentityDirectory",
    "adapt_directory",
    "adapt_sessions",
    "ANONYMOUS_NICKNAME",
]

ANONYMOUS_NICKNAME = "Anonymous"

_SUMMARY_FIELDS = (
    "author_nickname",
    "title",
    "body",
    "tags",
    "image_urls",
    "visibility",
    "status",
    "moderation",
    "created_at",
    "updated_at",
)


# --------------------------------------------------------------------------- #
# IF-note-exists-for-interaction
# --------------------------------------------------------------------------- #
@runtime_checkable
class NoteExistencePort(Protocol):
    """The frozen existence/visibility port (the *only* mandatory note seam)."""

    def is_publicly_interactable(self, note_id: str) -> bool:  # pragma: no cover - protocol
        ...

    def get_note_author_id(self, note_id: str) -> str | None:  # pragma: no cover - protocol
        ...


class NoteDirectory:
    """Normalised view of the note port used by :class:`InteractionService`.

    ``interactable`` / ``author_id`` implement ``IF-note-exists-for-interaction``
    and ``summary`` is the optional read seam needed to project favorites onto
    :class:`~blacknote.contracts.interfaces.NoteSummary`.  A port that only
    provides the two frozen methods simply yields ``None`` summaries: the
    favorite listing then degrades to "no cards" instead of inventing content.
    """

    def __init__(
        self,
        *,
        interactable: Callable[[str], bool],
        author_id: Callable[[str], str | None],
        summarise: Callable[[str], NoteSummary | None] | None = None,
    ) -> None:
        self._interactable = interactable
        self._author_id = author_id
        self._summarise = summarise

    def interactable(self, note_id: str) -> bool:
        return bool(self._guard(self._interactable, note_id, default=False))

    def author_id(self, note_id: str) -> str | None:
        value = self._guard(self._author_id, note_id, default=None)
        return value if isinstance(value, str) and is_valid_id(value) else None

    def summary(self, note_id: str) -> NoteSummary | None:
        if self._summarise is None:
            return None
        value = self._guard(self._summarise, note_id, default=None)
        return value if isinstance(value, NoteSummary) else None

    # -- frozen port shape (so a directory *is* a NoteExistenceApi) ----------
    def is_publicly_interactable(self, note_id: str) -> bool:
        """Alias of :meth:`interactable` matching ``IF-note-exists-for-interaction``."""
        return self.interactable(note_id)

    def get_note_author_id(self, note_id: str) -> str | None:
        """Alias of :meth:`author_id` matching ``IF-note-exists-for-interaction``."""
        return self.author_id(note_id)

    @staticmethod
    def _guard(call: Callable[[str], Any], note_id: str, *, default: Any) -> Any:
        """The existence port fails closed: never let a probe raise at a caller."""
        if not is_valid_id(note_id):
            return default
        try:
            return call(note_id)
        except (ContractError, LookupError, TypeError, ValueError):
            return default


class NullNoteDirectory(NoteDirectory):
    """Fail-closed default: no note is interactable, so every write 404s."""

    def __init__(self) -> None:
        super().__init__(interactable=lambda _note_id: False,
                         author_id=lambda _note_id: None)


# --------------------------------------------------------------------------- #
# Dev / test fixture note directory (self-contained: never imports mod-note)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectoryNote:
    """Minimal note row understood by :class:`InMemoryNoteDirectory`."""

    id: str
    author_id: str
    author_nickname: str = ""
    title: str = "note"
    body: str = ""
    tags: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    visibility: Visibility = Visibility.PUBLIC
    status: ContentStatus = ContentStatus.PUBLISHED
    moderation: ModerationState = ModerationState.VISIBLE
    created_at: str = ""
    updated_at: str = ""
    counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "author_id", str(self.author_id))
        object.__setattr__(self, "visibility", Visibility(str(self.visibility)))
        object.__setattr__(self, "status", ContentStatus(str(self.status)))
        object.__setattr__(self, "moderation", ModerationState(str(self.moderation)))
        object.__setattr__(self, "tags", tuple(self.tags or ()))
        object.__setattr__(self, "image_urls", tuple(self.image_urls or ()))
        stamp = self.created_at or now_iso()
        object.__setattr__(self, "created_at", stamp)
        object.__setattr__(self, "updated_at", self.updated_at or stamp)


class InMemoryNoteDirectory(NoteDirectory):
    """A note *fake*: the two frozen methods plus a summary projection.

    It exists so that dev wiring and the layer's own tests have something to
    inject without importing ``mod-note``; visibility decisions are delegated to
    the shared contract predicate exactly like the real service does.
    """

    def __init__(self, notes: Any = None) -> None:
        self.rows: dict[str, DirectoryNote] = {}
        super().__init__(interactable=self._interactable, author_id=self._author,
                         summarise=self._summary)
        for row in notes or ():
            self.add(row)

    # -- fixture API (dev wiring only) ---------------------------------------
    def add(self, row: DirectoryNote | Mapping[str, Any] | Any) -> DirectoryNote:
        note = row if isinstance(row, DirectoryNote) else DirectoryNote(**dict(row))
        self.rows[note.id] = note
        return note

    def publish(self, note_id: str, *, status: ContentStatus | str,
                moderation: ModerationState | str | None = None) -> None:
        """Flip lifecycle flags (used by fixtures simulating delete / takedown)."""
        note = self.rows.get(note_id)
        if note is None:
            return
        changes: dict[str, Any] = {"status": status}
        if moderation is not None:
            changes["moderation"] = moderation
        self.rows[note_id] = _replace(note, **changes)

    def set_counter(self, note_id: str, **counters: int) -> None:
        """Seed a cached counter value (the note-side cache column)."""
        note = self.rows.get(note_id)
        if note is not None:
            merged = dict(note.counters)
            merged.update(counters)
            self.rows[note_id] = _replace(note, counters=merged)

    def counters_of(self, note_id: str) -> Mapping[str, int]:
        note = self.rows.get(note_id)
        return dict(note.counters) if note else {}

    # -- port implementation --------------------------------------------------
    def _interactable(self, note_id: str) -> bool:
        note = self.rows.get(note_id)
        return note is not None and is_interactable(None, note)

    def _author(self, note_id: str) -> str | None:
        note = self.rows.get(note_id)
        if note is None or not is_visible_to(None, note, scope=VisibilityScope.PUBLIC):
            return None
        return note.author_id

    def _summary(self, note_id: str) -> NoteSummary | None:
        note = self.rows.get(note_id)
        if note is None or not is_visible_to(None, note, scope=VisibilityScope.PUBLIC):
            return None
        return _summary_of(note)


def _replace(row: DirectoryNote, **changes: Any) -> DirectoryNote:
    from dataclasses import replace as _dc_replace

    return _dc_replace(row, **changes)


def _nickname_of(value: Any) -> str:
    return clean_text(getattr(value, "author_nickname", "") or "") or ANONYMOUS_NICKNAME


def _summary_of(row: Any) -> NoteSummary:
    """Project any note-row-ish object onto ``NoteSummary`` (contract types only)."""
    view = _view_of(row)
    return view.summary()


def _view_of(row: Any) -> NoteView:
    to_view = getattr(row, "to_view", None)
    if callable(to_view):
        view = to_view()
        if isinstance(view, NoteView):
            return view
    if isinstance(row, NoteView):
        return row
    data = dict(row) if isinstance(row, Mapping) else {
        name: getattr(row, name, None) for name in _SUMMARY_FIELDS + ("id", "author_id")
    }
    counters = data.get("counters") or {}
    if not hasattr(counters, "likes"):
        counters = _counters_from(counters)
    return NoteView(
        id=data["id"],
        author_id=data["author_id"],
        author_nickname=data.get("author_nickname") or ANONYMOUS_NICKNAME,
        title=data.get("title") or "",
        body=data.get("body") or "",
        tags=tuple(data.get("tags") or ()),
        image_urls=tuple(data.get("image_urls") or ()),
        visibility=data.get("visibility", Visibility.PUBLIC),
        status=data.get("status", ContentStatus.PUBLISHED),
        moderation=data.get("moderation", ModerationState.VISIBLE),
        counters=counters,  # type: ignore[arg-type]
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def _counters_from(payload: Mapping[str, Any]):
    from blacknote.contracts.interfaces import Counters

    return Counters(
        likes=int(payload.get("likes", payload.get("like_count", 0)) or 0),
        favorites=int(payload.get("favorites", payload.get("favorite_count", 0)) or 0),
        comments=int(payload.get("comments", payload.get("comment_count", 0)) or 0),
    )


# --------------------------------------------------------------------------- #
# Adapter over an arbitrary note port
# --------------------------------------------------------------------------- #
_SEAM_SUMMARISERS: tuple[str, ...] = ("note_summary", "get_summary", "summary_of")


def adapt_note_port(target: Any) -> NoteDirectory:
    """Normalise any plausible note dependency into a :class:`NoteDirectory`.

    ``None`` is accepted and produces the fail-closed :class:`NullNoteDirectory`
    — an un-wired note port must never make a note look interactable.
    """
    if target is None:
        return NullNoteDirectory()
    if isinstance(target, NoteDirectory):
        return target

    if all(callable(getattr(target, name, None))
           for name in ("is_publicly_interactable", "get_note_author_id")):
        return NoteDirectory(
            interactable=target.is_publicly_interactable,
            author_id=target.get_note_author_id,
            summarise=_summariser_from(target),
        )

    row_reader = _row_reader(target)
    if row_reader is not None:
        def _row(note_id: str) -> Any:
            try:
                return row_reader(note_id)
            except (ContractError, LookupError):
                return None

        def _interactable(note_id: str) -> bool:
            row = _row(note_id)
            return row is not None and is_interactable(None, row)

        def _author(note_id: str) -> str | None:
            row = _row(note_id)
            if row is None or not is_visible_to(None, row, scope=VisibilityScope.PUBLIC):
                return None
            return getattr(row, "author_id", None) if not isinstance(row, Mapping) \
                else row.get("author_id")

        def _summary(note_id: str) -> NoteSummary | None:
            row = _row(note_id)
            if row is None or not is_visible_to(None, row, scope=VisibilityScope.PUBLIC):
                return None
            return _summary_of(row)

        return NoteDirectory(interactable=_interactable, author_id=_author,
                             summarise=_summary)

    raise ContractError(
        ErrorCode.CONFIG_ERROR,
        "the interaction layer needs a note port exposing is_publicly_interactable() "
        "and get_note_author_id() (IF-note-exists-for-interaction), or a note "
        "repository exposing get(note_id)",
        {"got": type(target).__name__},
    )


def _row_reader(target: Any) -> Callable[[str], Any] | None:
    for name in ("get", "load", "get_note_record", "inspect_note_record", "find"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    repository = getattr(target, "repository", None)
    getter = getattr(repository, "get", None)
    return getter if callable(getter) else None


def _summariser_from(target: Any) -> Callable[[str], NoteSummary | None] | None:
    """Pick the best available *public* note read seam for favorites listing.

    Order of preference: an explicit summary seam, then ``get_note`` (the
    frozen ``IF-note-crud`` read, filtered by the shared predicate on the note
    side), then a storage probe judged here through that very same predicate.
    """
    for name in _SEAM_SUMMARISERS:
        seam = getattr(target, name, None)
        if callable(seam):
            return _coerce_summary(seam)

    get_note = getattr(target, "get_note", None)
    if callable(get_note):
        def _via_get_note(note_id: str) -> NoteSummary | None:
            # Anonymous viewer: favorites cards must not leak the viewer's
            # private surface, and the note layer owns the visibility decision.
            try:
                view = get_note(None, note_id)
            except ContractError:
                return None
            if is_error(view) or view is None:
                return None
            return view.summary() if isinstance(view, NoteView) else _summary_of(view)

        return _via_get_note

    probe = getattr(target, "inspect_note_record", None)
    if callable(probe):
        def _via_probe(note_id: str) -> NoteSummary | None:
            try:
                row = probe(note_id)
            except ContractError:
                return None
            if row is None or not is_visible_to(None, row, scope=VisibilityScope.PUBLIC):
                return None
            return _summary_of(row)

        return _via_probe

    repository = getattr(target, "repository", None)
    getter = getattr(repository, "get", None)
    if callable(getter):
        def _via_repository(note_id: str) -> NoteSummary | None:
            try:
                row = getter(note_id)
            except ContractError:
                return None
            if row is None or not is_visible_to(None, row, scope=VisibilityScope.PUBLIC):
                return None
            return _summary_of(row)

        return _via_repository

    return None


def _coerce_summary(seam: Callable[[str], Any]) -> Callable[[str], NoteSummary | None]:
    def _call(note_id: str) -> NoteSummary | None:
        try:
            value = seam(note_id)
        except ContractError:
            return None
        if value is None or is_error(value):
            return None
        if isinstance(value, NoteSummary):
            return value
        if isinstance(value, NoteView):
            return value.summary()
        return _summary_of(value)

    return _call


# --------------------------------------------------------------------------- #
# Identity directory (nicknames for CommentView)
# --------------------------------------------------------------------------- #
@runtime_checkable
class IdentityDirectory(Protocol):
    """Read-only identity port (nickname projection), satisfied by ``mod-account``."""

    def nickname_of(self, user_id: str) -> str:  # pragma: no cover - protocol
        ...


class NullIdentityDirectory:
    """No identity data: every comment is attributed to ``Anonymous``."""

    def nickname_of(self, user_id: str) -> str:  # noqa: ARG002 - interface parity
        return ANONYMOUS_NICKNAME


class InMemoryIdentityDirectory:
    """Fixture directory: ``{user_id: nickname}`` plus token → principal data."""

    def __init__(self, nicknames: Mapping[str, str] | None = None) -> None:
        self.nicknames: dict[str, str] = dict(nicknames or {})

    def set_nickname(self, user_id: str, nickname: str) -> None:
        self.nicknames[user_id] = clean_text(nickname) or ANONYMOUS_NICKNAME

    def nickname_of(self, user_id: str) -> str:
        return self.nicknames.get(user_id, ANONYMOUS_NICKNAME)


def adapt_directory(target: Any) -> IdentityDirectory:
    """Accept a directory, an account facade, a mapping or ``None``."""
    if target is None:
        return NullIdentityDirectory()
    if callable(getattr(target, "nickname_of", None)):
        return target  # type: ignore[return-value]
    if callable(getattr(target, "get_profile", None)):
        return _AccountProfileDirectory(target)
    if isinstance(target, Mapping):
        return InMemoryIdentityDirectory(target)
    raise ContractError(
        ErrorCode.CONFIG_ERROR,
        "identity directory must expose nickname_of() or get_profile()",
        {"got": type(target).__name__},
    )


class _AccountProfileDirectory:
    """Duck-typed adapter over an account facade (never imports ``mod-account``)."""

    def __init__(self, account: Any) -> None:
        self._account = account

    def nickname_of(self, user_id: str) -> str:
        getter = getattr(self._account, "get_profile", None)
        if getter is None or not is_valid_id(user_id):
            return ANONYMOUS_NICKNAME
        try:
            profile = getter(user_id)
        except (ContractError, TypeError):
            return ANONYMOUS_NICKNAME
        if profile is None or isinstance(profile, ErrorEnvelope):
            return ANONYMOUS_NICKNAME
        return clean_text(getattr(profile, "nickname", "") or "") or ANONYMOUS_NICKNAME


# --------------------------------------------------------------------------- #
# Session resolver adaptation (INV-auth-session)
# --------------------------------------------------------------------------- #
def adapt_sessions(target: Any) -> Callable[[str], Principal | None] | None:
    """Turn any plausible session port into a ``require_auth`` resolver.

    Accepts ``None`` (no wiring: the contract guard then fails closed), an
    object with ``resolve(token)``, an account facade with
    ``current_user(token)``, a ``token → principal`` mapping (test fixtures), a
    fixed :class:`Principal`, or a plain callable.  The returned callable always
    produces a :class:`Principal` or ``None``; the AUTH_REQUIRED /
    SESSION_EXPIRED decision stays inside ``contracts.auth_guard``.
    """
    if target is None:
        return None
    if isinstance(target, Principal):
        fixed = target
        return lambda _token: fixed

    if hasattr(target, "resolve"):
        raw_resolve = target.resolve
    elif hasattr(target, "current_user"):
        raw_resolve = target.current_user
    elif isinstance(target, Mapping):
        table = dict(target)

        def raw_resolve(token: str, _table: dict = table) -> Any:  # type: ignore[misc]
            return _table.get(token)

    elif callable(target):
        raw_resolve = target
    else:
        raise ContractError(
            ErrorCode.CONFIG_ERROR,
            "session resolver must expose resolve()/current_user() or be callable",
            {"got": type(target).__name__},
        )

    def _resolve(token: str) -> Principal | None:
        return _to_principal(raw_resolve(token), token)

    return _resolve


def _to_principal(candidate: Any, token: str) -> Principal | None:
    if candidate is None:
        return None
    if isinstance(candidate, Principal):
        return candidate
    if is_error(candidate):
        return None
    if isinstance(candidate, Mapping):
        payload = dict(candidate)
        payload.setdefault("user_id", payload.pop("id", None))
        payload.pop("token", None)
        try:
            return Principal(token=token, **payload)
        except (ContractError, TypeError, ValueError):
            return None
    user_id = getattr(candidate, "user_id", None)
    if user_id is None:
        return None
    try:
        return Principal(
            user_id=user_id,
            token=token,
            role=getattr(candidate, "role", "user"),
            issued_at=getattr(candidate, "issued_at", None),
            expires_at=getattr(candidate, "expires_at", None),
            nickname=getattr(candidate, "nickname", None),
        )
    except (ContractError, TypeError, ValueError):
        return None


def authenticate(token: str, resolver: Callable[[str], Principal | None] | None) -> Principal:
    """Single write-gate helper: ``AUTH_REQUIRED`` (never a partial write)."""
    return require_auth(token, resolver)


def fixture_token() -> str:
    """Convenience for fixtures: a syntactically valid bearer token."""
    return new_id()
