"""Service level tests for ``mod-note`` (acceptance T1-C1 / T1-C2).

The note layer is exercised through its frozen contracts only:

* ``IF-note-crud``  : create / update / delete / get
* ``IF-note-feed``  : list_feed / search_notes
* ``IF-note-exists-for-interaction`` : is_publicly_interactable / get_note_author_id

No sibling business module is imported: the account dependencies are the two
ports of :mod:`blacknote.note.identity` (a stub session resolver + an in-memory
directory) and storage is the note layer's own repository implementations.

Error convention under test: the note service signals failures by raising
``ContractError`` carrying the frozen ``ErrorEnvelope`` (the same shape the web
layer serialises), so assertions read ``excinfo.value.envelope.code``.
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pytest

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id, new_id
from blacknote.contracts.interfaces import (
    Counters,
    NoteCrudApi,
    NoteExistenceApi,
    NoteFeedApi,
    NoteInput,
    NoteSummary,
    NoteView,
    Page,
    PageQuery,
    FeedQuery,
)
from blacknote.contracts.roles import Role
from blacknote.contracts.timeutil import ISO8601_UTC_PATTERN, to_iso
from blacknote.contracts.validation import FIXED_TOPIC_TAGS, LIMITS
from blacknote.contracts.visibility import ContentStatus, ModerationState, Visibility
from blacknote.note import (
    InMemoryAccountDirectory,
    InMemoryNoteRepository,
    NoteFeedQuery,
    NoteService,
    SQLiteNoteRepository,
    encode_cursor,
)

NOTE_DIR = Path(__file__).resolve().parents[1] / "src" / "blacknote" / "note"

BASE_MOMENT = datetime(2026, 2, 14, 9, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
class StubSessions:
    """Minimal stand-in for ``mod-account``'s session resolver port."""

    def __init__(self) -> None:
        self._by_token: dict[str, Principal] = {}

    def login(self, user_id: str, *, role: Role = Role.USER, nickname: str | None = None) -> str:
        token = new_id()
        self._by_token[token] = Principal(
            user_id=user_id, token=token, role=role, nickname=nickname
        )
        return token

    def resolve(self, token: str) -> Principal | None:
        return self._by_token.get(token)

    def expire(self, token: str) -> None:
        principal = self._by_token[token]
        self._by_token[token] = Principal(
            user_id=principal.user_id,
            token=token,
            role=principal.role,
            expires_at=to_iso(BASE_MOMENT - timedelta(seconds=1)),
        )


@dataclass
class Harness:
    settings: Settings = field(default_factory=lambda: Settings(environment="test",
                                                                database_url="sqlite:///:memory:"))
    repository: InMemoryNoteRepository = field(default_factory=InMemoryNoteRepository)
    sessions: StubSessions = field(default_factory=StubSessions)
    directory: InMemoryAccountDirectory = field(default_factory=InMemoryAccountDirectory)
    service: NoteService = field(init=False)
    stamps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.service = NoteService(
            self.repository,
            sessions=self.sessions,
            directory=self.directory,
            settings=self.settings,
            clock=self._clock,
        )

    def _clock(self) -> str:
        stamp = to_iso(BASE_MOMENT + timedelta(seconds=len(self.stamps) + 1))
        self.stamps.append(stamp)
        return stamp

    # -- actors ---------------------------------------------------------------
    def make_author(self, nickname: str = "Kai") -> tuple[str, str]:
        user_id = new_id()
        self.directory.set_nickname(user_id, nickname)
        return user_id, self.sessions.login(user_id, nickname=nickname)

    def make_admin(self, nickname: str = "Root") -> tuple[str, str]:
        user_id = new_id()
        self.directory.set_nickname(user_id, nickname)
        return user_id, self.sessions.login(user_id, role=Role.ADMIN, nickname=nickname)

    # -- content helpers ------------------------------------------------------
    def input(self, **changes: Any) -> NoteInput:
        payload: dict[str, Any] = {
            "title": "LangGraph checkpointing",
            "body": "## why\n**snapshot** every superstep",
            "tags": ["AI工具", "LangGraph"],
            "image_urls": ["https://cdn.example/a.png"],
        }
        payload.update(changes)
        return NoteInput(
            title=payload["title"],
            body=payload["body"],
            tags=tuple(payload["tags"]),
            image_urls=tuple(payload["image_urls"]),
            visibility=payload.get("visibility", Visibility.PUBLIC),
        )

    def create(self, token: str, **changes: Any) -> NoteView:
        return self.service.create_note(token, self.input(**changes))

    def publish_many(self, token: str, count: int, **changes: Any) -> list[NoteView]:
        return [self.create(token, **changes) for _ in range(count)]

    @staticmethod
    def titles(page: Page[NoteSummary]) -> list[str]:
        return [item.title for item in page.items]

    @staticmethod
    def ids(page: Page[NoteSummary]) -> list[str]:
        return [item.id for item in page.items]


@pytest.fixture()
def h() -> Harness:
    return Harness()


def envelope_of(exc: BaseException) -> Any:
    assert isinstance(exc, ContractError), (
        f"expected ContractError, got {type(exc).__name__}: {exc}"
    )
    return exc.envelope


def codes_of(env: Any) -> dict[str, str]:
    return {item["field"]: item["code"] for item in env.details.get("fields", [])}


# --------------------------------------------------------------------------- #
# create_note
# --------------------------------------------------------------------------- #
class TestCreateNote:
    def test_valid_note_is_created_and_appears_in_the_feed(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.create(token)

        assert isinstance(note, NoteView)
        assert is_valid_id(note.id) and note.author_id == author_id
        assert note.author_nickname == "Kai"
        assert note.status is ContentStatus.PUBLISHED
        assert note.moderation is ModerationState.VISIBLE
        assert note.visibility is Visibility.PUBLIC

        feed = h.service.list_feed(FeedQuery())
        assert [item.id for item in feed.items] == [note.id]
        assert feed.total == 1

        profile = h.service.list_feed(FeedQuery(author_id=author_id))
        assert [item.id for item in profile.items] == [note.id]

    def test_timestamps_are_canonical_utc_iso8601(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        for value in (note.created_at, note.updated_at):
            assert value.endswith("Z")
            assert ISO8601_UTC_PATTERN.match(value)
        payload = note.to_dict()
        assert json.loads(json.dumps(payload))["created_at"].endswith("Z")

    def test_interaction_counters_are_initialised_to_zero(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        assert note.counters.to_dict() == {"likes": 0, "favorites": 0, "comments": 0}
        record = h.service.inspect_note_record(note.id)
        assert record is not None and record.counters.to_dict() == note.counters.to_dict()

    def test_topic_tag_aliases_resolve_to_the_fixed_topics(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token, tags=["ai-products", "rag"])
        assert note.tags[0] == "AI产品"
        assert h.service.inspect_note_record(note.id).topic == "AI产品"

    def test_missing_fields_yield_validation_error_listing_them(self, h: Harness) -> None:
        _author, token = h.make_author()
        with pytest.raises(ContractError) as excinfo:
            h.service.create_note(token, {"body": "   "})
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert env.http_status == 400
        fields = {item["field"] for item in env.details["fields"]}
        assert {"title", "body"} <= fields
        assert codes_of(env)["title"] == "required"
        assert codes_of(env)["body"] == "blank"
        assert h.repository.count() == 0  # nothing was written

    def test_oversized_fields_are_rejected_with_the_contract_limits(
        self, h: Harness
    ) -> None:
        _author, token = h.make_author()
        with pytest.raises(ContractError) as excinfo:
            h.create(token, title="x" * (LIMITS["title"]["max"] + 1),
                     body="y" * (LIMITS["body"]["max"] + 1))
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        codes = codes_of(env)
        assert codes["title"] == codes["body"] == "too_long"
        assert h.repository.count() == 0

    def test_more_than_nine_images_is_rejected(self, h: Harness) -> None:
        _author, token = h.make_author()
        too_many = [f"https://cdn.example/{index}.png"
                    for index in range(LIMITS["images"]["max"] + 1)]
        with pytest.raises(ContractError) as excinfo:
            h.create(token, image_urls=too_many)
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert any(item["field"].startswith("image_urls") for item in env.details["fields"])

    def test_nine_images_are_accepted(self, h: Harness) -> None:
        _author, token = h.make_author()
        urls = [f"https://cdn.example/{index}.png" for index in range(LIMITS["images"]["max"])]
        note = h.create(token, image_urls=urls)
        assert note.image_urls == tuple(urls)

    def test_inline_image_beyond_the_size_budget_is_rejected(self, h: Harness) -> None:
        import base64

        h.settings = Settings(environment="test", database_url="sqlite:///:memory:",
                              max_image_bytes=64)
        h.service = NoteService(h.repository, sessions=h.sessions, directory=h.directory,
                                settings=h.settings, clock=h._clock)
        _author, token = h.make_author()
        payload = base64.b64encode(b"x" * 200).decode("ascii")
        with pytest.raises(ContractError) as excinfo:
            h.create(token, image_urls=[f"data:image/png;base64,{payload}"])
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["image_urls[0]"] == "image_too_large"

    def test_single_image_budget_comes_from_config(self) -> None:
        assert Settings().max_image_bytes == 5 * 1024 * 1024  # 5 MB per image

    def test_at_least_one_topic_tag_is_required(self, h: Harness) -> None:
        _author, token = h.make_author()
        with pytest.raises(ContractError) as excinfo:
            h.create(token, tags=[])
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["tags"] == "required"

        with pytest.raises(ContractError) as excinfo:
            h.create(token, tags=["LangGraph"])  # custom tags only, no fixed topic
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["tags"] == "invalid_topic"
        assert h.repository.count() == 0

    def test_status_vocabulary_is_draft_or_published(self, h: Harness) -> None:
        _author, token = h.make_author()
        draft = h.service.create_note(token, h.input(), status="draft")
        assert draft.status is ContentStatus.DRAFT
        for bad in ("deleted", "archived", 7):
            with pytest.raises(ContractError) as excinfo:
                h.service.create_note(token, h.input(), status=bad)
            assert envelope_of(excinfo.value).code is ErrorCode.VALIDATION_ERROR
        # a blank / omitted status keeps the documented default: publish now
        assert h.service.create_note(token, h.input(), status="").status \
            is ContentStatus.PUBLISHED

    def test_writes_require_a_session(self, h: Harness) -> None:
        author_id, _ignored = h.make_author()
        for bad_token in (None, "", "   ", "not-a-token", new_id(), new_id().upper(), 42):
            with pytest.raises(ContractError) as excinfo:
                h.service.create_note(bad_token, h.input())  # type: ignore[arg-type]
            assert envelope_of(excinfo.value).code is ErrorCode.AUTH_REQUIRED
        assert h.repository.count() == 0
        assert h.service.list_feed(FeedQuery(author_id=author_id)).total == 0

    def test_expired_session_cannot_write(self, h: Harness) -> None:
        _author, token = h.make_author()
        h.sessions.expire(token)
        with pytest.raises(ContractError) as excinfo:
            h.service.create_note(token, h.input())
        assert envelope_of(excinfo.value).code is ErrorCode.SESSION_EXPIRED
        assert h.repository.count() == 0


# --------------------------------------------------------------------------- #
# update_note / delete_note — ownership and reflection
# --------------------------------------------------------------------------- #
class TestAuthoringAndOwnership:
    def test_author_can_edit_and_the_change_is_reflected_everywhere(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        edited = h.service.update_note(token, note.id,
                                       h.input(title="Edited title", body="edited **body**"))

        assert edited.id == note.id
        assert edited.created_at == note.created_at
        assert edited.updated_at > note.updated_at
        assert h.service.get_note(token, note.id).title == "Edited title"
        feed_titles = [item.title for item in h.service.list_feed(FeedQuery()).items]
        assert feed_titles == ["Edited title"]
        assert h.service.search_notes("edited body", PageQuery()).total == 1
        assert "edited" in h.service.search_notes("edited body", PageQuery()).items[0].excerpt

    def test_edit_preserves_counters_and_ownership(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.create(token)
        # the interaction layer owns the counters; simulate its write
        record = h.service.inspect_note_record(note.id).changed(
            counters=h.service.inspect_note_record(note.id).counters.delta(likes=4, comments=2)
        )
        h.repository.save(record)

        edited = h.service.update_note(token, note.id, h.input(title="Rename only"))
        assert edited.counters.to_dict() == {"likes": 4, "favorites": 0, "comments": 2}
        assert edited.author_id == author_id
        assert h.service.get_note(None, note.id).counters.to_dict() == edited.counters.to_dict()

    def test_author_can_publish_a_draft_then_it_enters_the_public_feed(
        self, h: Harness
    ) -> None:
        _author, token = h.make_author()
        note = h.service.create_note(token, h.input(), status="draft")
        assert h.service.list_feed(FeedQuery()).total == 0
        published = h.service.update_note(token, note.id, h.input(), status="published")
        assert published.status is ContentStatus.PUBLISHED
        assert [item.id for item in h.service.list_feed(FeedQuery()).items] == [note.id]

    def test_non_author_update_and_delete_are_denied(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        stranger_id, stranger_token = h.make_author(nickname="Ana")

        for action in (
            lambda: h.service.update_note(stranger_token, note.id, h.input(title="hijack")),
            lambda: h.service.delete_note(stranger_token, note.id),
        ):
            with pytest.raises(ContractError) as excinfo:
                action()
            env = envelope_of(excinfo.value)
            assert env.code is ErrorCode.PERMISSION_DENIED
            assert env.http_status == 403

        assert h.service.get_note(token, note.id).title == "LangGraph checkpointing"
        assert h.service.is_publicly_interactable(note.id) is True

    def test_admins_do_not_edit_or_delete_notes_directly(self, h: Harness) -> None:
        """Admin removal goes through the moderation flag, not the author API."""
        _author, token = h.make_author()
        note = h.create(token)
        _admin_id, admin_token = h.make_admin()

        with pytest.raises(ContractError) as excinfo:
            h.service.update_note(admin_token, note.id, h.input(title="censorship"))
        assert envelope_of(excinfo.value).code is ErrorCode.PERMISSION_DENIED
        with pytest.raises(ContractError) as excinfo:
            h.service.delete_note(admin_token, note.id)
        assert envelope_of(excinfo.value).code is ErrorCode.PERMISSION_DENIED

        unlisted = h.service.apply_moderation_state(note.id, ModerationState.UNLISTED)
        assert unlisted.moderation is ModerationState.UNLISTED
        assert h.service.is_publicly_interactable(note.id) is False

    def test_anonymous_writes_are_denied_before_ownership(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        with pytest.raises(ContractError) as excinfo:
            h.service.delete_note(None, note.id)
        assert envelope_of(excinfo.value).code is ErrorCode.AUTH_REQUIRED

    def test_unknown_or_malformed_note_id(self, h: Harness) -> None:
        _author, token = h.make_author()
        with pytest.raises(ContractError) as excinfo:
            h.service.update_note(token, new_id(), h.input())
        assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
        with pytest.raises(ContractError) as excinfo:
            h.service.delete_note(token, "1")
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["note_id"] == "invalid_id"

    def test_delete_is_a_soft_delete_tombstone(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        assert h.service.delete_note(token, note.id) is None

        record = h.service.inspect_note_record(note.id)
        assert record is not None, "the row must survive (soft delete)"
        assert record.status is ContentStatus.DELETED
        assert h.repository.count() == 1

        with pytest.raises(ContractError) as excinfo:
            h.service.get_note(token, note.id)
        assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
        assert h.service.list_feed(FeedQuery()).total == 0
        assert h.service.search_notes("LangGraph", PageQuery()).total == 0
        # a second delete behaves like a missing note
        with pytest.raises(ContractError) as excinfo:
            h.service.delete_note(token, note.id)
        assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND


# --------------------------------------------------------------------------- #
# INV-visibility-predicate: reads must funnel through contracts.visibility
# --------------------------------------------------------------------------- #
class TestVisibilityPredicateIsTheSingleSource:
    def test_note_layer_does_not_reimplement_the_predicate(self) -> None:
        import blacknote.contracts.visibility as contract_visibility
        import blacknote.note.service as note_service

        # the service calls the contract function object itself, no local copy
        assert note_service.is_visible_to is contract_visibility.is_visible_to
        assert note_service.is_interactable is contract_visibility.is_interactable

        offenders: list[str] = []
        for path in sorted(NOTE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "is_visible_to":
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], "visibility must not be redefined inside mod-note"

    def test_every_read_path_consults_the_shared_predicate(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import blacknote.note.service as note_service

        _author, token = h.make_author()
        note = h.create(token)
        calls: list[tuple[str, str]] = []
        real = note_service.is_visible_to

        def spy(actor: Any, content: Any, *, scope: Any = "public", **kw: Any) -> bool:
            calls.append((getattr(content, "id", None) or "?", str(scope)))
            return real(actor, content, scope=scope, **kw)

        monkeypatch.setattr(note_service, "is_visible_to", spy)
        monkeypatch.setattr(
            note_service, "is_interactable",
            lambda actor, content, **kw: spy(actor, content, scope="public", **kw),
        )

        h.service.get_note(None, note.id)
        h.service.list_feed(FeedQuery())
        h.service.search_notes("LangGraph", PageQuery())
        h.service.is_publicly_interactable(note.id)
        h.service.get_note_author_id(note.id)

        assert len(calls) >= 6, calls
        assert {note.id} <= {item[0] for item in calls}
        assert "public" in {item[1] for item in calls}

    def test_reads_break_loudly_if_the_predicate_says_no(
        self, h: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behavioural proof that no fallback visibility logic exists locally."""
        import blacknote.note.service as note_service

        _author, token = h.make_author()
        note = h.create(token)
        monkeypatch.setattr(note_service, "is_visible_to", lambda *a, **k: False)
        monkeypatch.setattr(note_service, "is_interactable", lambda *a, **k: False)

        with pytest.raises(ContractError) as excinfo:
            h.service.get_note(token, note.id)
        assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
        assert h.service.list_feed(FeedQuery()).total == 0
        assert h.service.search_notes("LangGraph", PageQuery()).total == 0
        assert h.service.is_publicly_interactable(note.id) is False
        assert h.service.get_note_author_id(note.id) is None

    def test_draft_is_visible_to_the_author_only(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.service.create_note(token, h.input(), status="draft")

        # author detail + own management listing
        assert h.service.get_note(token, note.id).status is ContentStatus.DRAFT
        own = h.service.list_feed(
            FeedQuery(viewer_token=token, author_id=author_id)
        )
        assert [item.id for item in own.items] == [note.id]

        # everyone else, including the author's public feed, must not see it
        _other, other_token = h.make_author(nickname="Ana")
        for viewer in (None, other_token):
            with pytest.raises(ContractError) as excinfo:
                h.service.get_note(viewer, note.id)
            assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
            stranger_view = h.service.list_feed(FeedQuery(viewer_token=viewer,
                                                          author_id=author_id))
            assert stranger_view.total == 0
        assert h.service.list_feed(FeedQuery(viewer_token=token)).total == 0
        assert h.service.search_notes("checkpointing", PageQuery()).total == 0

    def test_moderated_note_disappears_from_public_surfaces(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.create(token)
        h.service.apply_moderation_state(note.id, ModerationState.UNLISTED)

        assert h.service.list_feed(FeedQuery()).total == 0
        assert h.service.search_notes("checkpointing", PageQuery()).total == 0
        _other, other_token = h.make_author(nickname="Ana")
        for viewer in (None, other_token):
            with pytest.raises(ContractError) as excinfo:
                h.service.get_note(viewer, note.id)
            assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
        # the author keeps a management view so the note can still be fixed
        assert [item.id for item in
                h.service.list_feed(FeedQuery(viewer_token=token,
                                              author_id=author_id)).items] == [note.id]

    def test_private_note_is_author_only(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.create(token, visibility=Visibility.PRIVATE)
        assert h.service.list_feed(FeedQuery()).total == 0
        assert h.service.is_publicly_interactable(note.id) is False
        assert h.service.get_note(token, note.id).id == note.id
        _other, other_token = h.make_author(nickname="Ana")
        with pytest.raises(ContractError):
            h.service.get_note(other_token, note.id)

    def test_invisible_and_missing_notes_produce_the_same_404(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.service.create_note(token, h.input(), status="draft")
        hidden = envelope_of(_raise(h.service.get_note, None, note.id))
        missing = envelope_of(_raise(h.service.get_note, None, new_id()))
        assert hidden.code is missing.code is ErrorCode.NOTE_NOT_FOUND
        assert hidden.http_status == missing.http_status == 404
        assert hidden.message == missing.message
        # existence is never disclosed: only the requested id differs
        assert set(hidden.details) == set(missing.details) == {"note_id"}


def _raise(func: Any, *args: Any, **kwargs: Any) -> ContractError:
    try:
        func(*args, **kwargs)
    except ContractError as exc:
        return exc
    raise AssertionError(f"{func!r} unexpectedly succeeded")


# --------------------------------------------------------------------------- #
# list_feed
# --------------------------------------------------------------------------- #
class TestFeed:
    def test_created_at_desc_ordering(self, h: Harness) -> None:
        _author, token = h.make_author()
        created = [h.create(token, title=f"note {index}") for index in range(5)]
        page = h.service.list_feed(FeedQuery(limit=10))
        assert [item.title for item in page.items] == [c.title for c in reversed(created)]
        stamps = [item.created_at for item in page.items]
        assert stamps == sorted(stamps, reverse=True)

    def test_cursor_pagination_covers_every_note_exactly_once(self, h: Harness) -> None:
        _author, token = h.make_author()
        created = [h.create(token, title=f"note {index}") for index in range(7)]
        expected = [c.id for c in reversed(created)]

        seen: list[str] = []
        cursor: str | None = None
        pages = 0
        while pages < 10:
            page = h.service.list_feed(FeedQuery(limit=3, cursor=cursor))
            pages += 1
            assert page.limit == 3
            assert len(page.items) <= 3
            seen.extend(item.id for item in page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert seen == expected
        assert pages == 3

    def test_total_counts_every_visible_note_from_the_cursor_on(self, h: Harness) -> None:
        _author, token = h.make_author()
        for index in range(5):
            h.create(token, title=f"note {index}")
        first = h.service.list_feed(FeedQuery(limit=2))
        assert first.total == 5 and first.has_more is True
        second = h.service.list_feed(FeedQuery(limit=2, cursor=first.next_cursor))
        assert second.total == 3 and second.has_more is True
        third = h.service.list_feed(FeedQuery(limit=2, cursor=second.next_cursor))
        assert third.total == 1 and third.has_more is False and len(third.items) == 1

    def test_pagination_defaults_to_at_least_ten_items(self, h: Harness) -> None:
        _author, token = h.make_author()
        for index in range(12):
            h.create(token, title=f"note {index}")
        page = h.service.list_feed(FeedQuery())
        assert page.limit >= 10
        assert len(page.items) == 12
        assert page.next_cursor is None

    def test_invalid_page_size_and_cursor_are_validation_errors(self, h: Harness) -> None:
        for bad_limit in (0, LIMITS["page_size"]["max"] + 1, "ten", 3.7):
            with pytest.raises(ContractError) as excinfo:
                h.service.list_feed(FeedQuery(limit=bad_limit))  # type: ignore[arg-type]
            assert envelope_of(excinfo.value).code is ErrorCode.VALIDATION_ERROR
        # the web handler path (raw query string mapping) must behave the same
        for bad_payload in ({"limit": "ten"}, {"limit": 0}, {"bogus": 1}):
            with pytest.raises(ContractError) as excinfo:
                h.service.list_feed(bad_payload)  # type: ignore[arg-type]
            assert envelope_of(excinfo.value).code is ErrorCode.VALIDATION_ERROR
        with pytest.raises(ContractError) as excinfo:
            h.service.list_feed(FeedQuery(cursor="bm90LWEtY3Vyc29y"))
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["cursor"] == "invalid_cursor"

    def test_topic_filter_is_exact(self, h: Harness) -> None:
        _author, token = h.make_author()
        topics = list(FIXED_TOPIC_TAGS)
        by_topic: dict[str, list[str]] = {topic: [] for topic in topics}
        for index, topic in enumerate(topics * 3):
            note = h.create(token, title=f"t{index}", tags=[topic, f"custom{index}"])
            by_topic[topic].append(note.id)
        mixed = h.create(token, title="mixed", tags=["AI工具", "AI新闻"])
        by_topic["AI工具"].append(mixed.id)
        # a note owns exactly one canonical topic: the first fixed tag wins
        assert h.service.inspect_note_record(mixed.id).topic == "AI工具"

        for topic, expected in by_topic.items():
            page = h.service.list_feed(FeedQuery(topic=topic,
                                                limit=LIMITS["page_size"]["max"]))
            assert {item.id for item in page.items} == set(expected), topic
            assert page.total == len(expected), topic
            assert all(topic in item.tags for item in page.items)
        # topic aliases reach the same canonical bucket
        alias = h.service.list_feed(FeedQuery(topic="ai-news", limit=50))
        assert {item.id for item in alias.items} == set(by_topic["AI新闻"])

    def test_tag_filter_is_exact(self, h: Harness) -> None:
        _author, token = h.make_author()
        wanted = h.create(token, title="graph", tags=["AI工具", "LangGraph"])
        h.create(token, title="other", tags=["AI工具", "Diffusers"])
        page = h.service.list_feed(NoteFeedQuery(tag="langgraph", limit=10))
        assert [item.id for item in page.items] == [wanted.id]
        assert page.total == 1

    def test_author_filter_is_exact(self, h: Harness) -> None:
        kai_id, kai = h.make_author(nickname="Kai")
        ana_id, ana = h.make_author(nickname="Ana")
        kai_note = h.create(kai, title="by kai")
        ana_note = h.create(ana, title="by ana")
        kais = h.service.list_feed(FeedQuery(author_id=kai_id))
        anas = h.service.list_feed(FeedQuery(author_id=ana_id))
        assert [item.id for item in kais.items] == [kai_note.id]
        assert [item.id for item in anas.items] == [ana_note.id]
        assert kais.total == 1 and anas.total == 1
        # every card carries the author identity
        assert kais.items[0].author_nickname == "Kai"
        assert anas.items[0].author_nickname == "Ana"

    def test_following_only_filter(self, h: Harness) -> None:
        kai_id, kai = h.make_author(nickname="Kai")
        ana_id, ana = h.make_author(nickname="Ana")
        ivy_id, ivy = h.make_author(nickname="Ivy")
        h.directory.follow(kai_id, ana_id)
        h.create(kai, title="mine")
        followed = h.create(ana, title="followed author")
        h.create(ivy, title="stranger")
        _ = ivy

        page = h.service.list_feed(NoteFeedQuery(viewer_token=kai, following_only=True, limit=10))
        assert [item.id for item in page.items] == [followed.id]
        assert page.total == 1
        # filters compose
        combined = h.service.list_feed(
            NoteFeedQuery(viewer_token=kai, following_only=True, author_id=ivy_id, limit=10)
        )
        assert combined.total == 0 and combined.next_cursor is None

    def test_following_only_requires_a_viewer(self, h: Harness) -> None:
        with pytest.raises(ContractError) as excinfo:
            h.service.list_feed(NoteFeedQuery(following_only=True))
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert codes_of(env)["following_only"] == "viewer_required"

    def test_empty_feed_reports_zero_total(self, h: Harness) -> None:
        page = h.service.list_feed(FeedQuery(topic="AI产品"))
        assert page.total == 0
        assert list(page.items) == []
        assert page.next_cursor is None
        assert page.to_dict()["total"] == 0

    def test_mapping_query_is_accepted_for_web_handlers(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token, tags=["AI新闻"])
        page = h.service.list_feed({"topic": "ai-news", "limit": 10})
        assert [item.id for item in page.items] == [note.id]
        with pytest.raises(ContractError) as excinfo:
            h.service.list_feed({"bogus": 1})
        assert envelope_of(excinfo.value).code is ErrorCode.VALIDATION_ERROR


# --------------------------------------------------------------------------- #
# search_notes
# --------------------------------------------------------------------------- #
class TestSearch:
    def test_keyword_hits_title_and_body(self, h: Harness) -> None:
        _author, token = h.make_author()
        in_title = h.create(token, title="vLLM throughput notes", body="nothing special")
        in_body = h.create(token, title="unrelated", body="explained: vLLM batching")
        h.create(token, title="different", body="cuda graphs")

        page = h.service.search_notes("vllm", PageQuery())
        assert {item.id for item in page.items} == {in_title.id, in_body.id}
        assert page.total == 2
        assert all("vllm" in (item.title + item.excerpt).lower() for item in page.items)

    def test_no_match_returns_zero_total(self, h: Harness) -> None:
        _author, token = h.make_author()
        h.create(token, title="present", body="present body")
        page = h.service.search_notes("absent-keyword", PageQuery())
        assert isinstance(page, Page)
        assert page.total == 0
        assert list(page.items) == []
        assert page.next_cursor is None
        assert json.dumps(page.to_dict(), default=str)

    def test_wildcards_are_treated_literally(self, h: Harness) -> None:
        _author, token = h.make_author()
        wanted = h.create(token, title="100% coverage", body="uses percent signs")
        h.create(token, title="nothing", body="unrelated body")
        page = h.service.search_notes("100%", PageQuery())
        assert [item.id for item in page.items] == [wanted.id]

    def test_search_uses_the_public_surface_only(self, h: Harness) -> None:
        author_id, token = h.make_author()
        draft = h.service.create_note(token, h.input(title="secret draft"), status="draft")
        live = h.create(token, title="secret live")
        h.service.apply_moderation_state(live.id, ModerationState.UNLISTED)
        found = h.service.search_notes("secret", PageQuery())
        assert found.total == 0 and list(found.items) == []
        # the public feed hides both as well ...
        assert h.service.list_feed(FeedQuery()).total == 0
        # ... while the author's management view keeps every one of them
        mine = h.service.list_feed(
            FeedQuery(viewer_token=token, author_id=author_id, limit=20)
        )
        assert [item.id for item in mine.items] == [live.id, draft.id]

    def test_search_pagination_is_cursor_based_and_newest_first(self, h: Harness) -> None:
        _author, token = h.make_author()
        created = [h.create(token, title=f"vector store {index}") for index in range(5)]
        first = h.service.search_notes("vector store", PageQuery(limit=2))
        assert [item.title for item in first.items] == ["vector store 4", "vector store 3"]
        assert first.total == 5 and first.next_cursor
        second = h.service.search_notes("vector store",
                                        PageQuery(limit=2, cursor=first.next_cursor))
        assert [item.title for item in second.items] == ["vector store 2", "vector store 1"]
        third = h.service.search_notes("vector store",
                                       PageQuery(limit=2, cursor=second.next_cursor))
        assert [item.title for item in third.items] == ["vector store 0"]
        assert third.next_cursor is None
        assert len({item.id for item in list(first) + list(second) + list(third)}) == 5
        assert created

    def test_invalid_keyword_is_a_validation_error(self, h: Harness) -> None:
        for bad in ("", "   ", "x" * (LIMITS["keyword"]["max"] + 1), None, 7):
            with pytest.raises(ContractError) as excinfo:
                h.service.search_notes(bad, PageQuery())  # type: ignore[arg-type]
            assert envelope_of(excinfo.value).code is ErrorCode.VALIDATION_ERROR


# --------------------------------------------------------------------------- #
# IF-note-exists-for-interaction
# --------------------------------------------------------------------------- #
class TestExistencePort:
    def test_published_note_is_interactable_and_reports_its_author(self, h: Harness) -> None:
        author_id, token = h.make_author()
        note = h.create(token)
        assert h.service.is_publicly_interactable(note.id) is True
        assert h.service.get_note_author_id(note.id) == author_id

    @pytest.mark.parametrize("mutation", ["draft", "private", "unlisted", "deleted"])
    def test_non_public_states_are_not_interactable(self, h: Harness, mutation: str) -> None:
        author_id, token = h.make_author()
        if mutation == "draft":
            note = h.service.create_note(token, h.input(), status="draft")
        elif mutation == "private":
            note = h.create(token, visibility=Visibility.PRIVATE)
        elif mutation == "unlisted":
            note = h.create(token)
            h.service.apply_moderation_state(note.id, ModerationState.UNLISTED)
        else:
            note = h.create(token)
            h.service.delete_note(token, note.id)

        assert h.service.is_publicly_interactable(note.id) is False
        assert h.service.get_note_author_id(note.id) is None

    def test_unknown_and_malformed_ids_fail_closed_without_raising(self, h: Harness) -> None:
        for bad in (new_id(), "1", "", None, 12345, new_id().upper()):
            assert h.service.is_publicly_interactable(bad) is False  # type: ignore[arg-type]
            assert h.service.get_note_author_id(bad) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# repositories
# --------------------------------------------------------------------------- #
class TestSqliteRepository:
    def test_round_trip_filters_and_persistence(self, h: Harness, tmp_path: Path) -> None:
        db = tmp_path / "notes.db"
        repo = SQLiteNoteRepository(f"sqlite:///{db}", settings=h.settings)
        sessions = h.sessions
        service = NoteService(repo, sessions=sessions, directory=h.directory,
                              settings=h.settings, clock=h._clock)
        author_id, token = h.make_author(nickname="Kai")
        graph = service.create_note(token, NoteInput(title="graph rag",
                                                    body="chunking", tags=["AI工具", "RAG"]))
        news = service.create_note(token, NoteInput(title="news", body="weekly digest",
                                                   tags=["AI新闻"]))
        assert isinstance(graph, NoteView) and isinstance(news, NoteView)

        assert repo.count() == 2
        tools = service.list_feed(FeedQuery(topic="ai-tools"))
        assert [item.id for item in tools.items] == [graph.id]
        rag = service.list_feed(NoteFeedQuery(tag="RAG"))
        assert [item.id for item in rag.items] == [graph.id]
        assert service.search_notes("digest", PageQuery()).total == 1

        service.update_note(token, news.id, NoteInput(title="news edited", body="edited body",
                                                     tags=["AI新闻"]))
        service.delete_note(token, graph.id)
        assert service.inspect_note_record(graph.id).status is ContentStatus.DELETED
        assert repo.count() == 2, "soft delete keeps the row"
        repo.close()

        # reopen: content survived the restart
        revived = NoteService(SQLiteNoteRepository(f"sqlite:///{db}", settings=h.settings),
                              sessions=sessions, directory=h.directory, settings=h.settings)
        assert revived.repository.count() == 2
        assert revived.get_note(None, news.id).title == "news edited"
        with pytest.raises(ContractError) as excinfo:
            revived.get_note(None, graph.id)
        assert envelope_of(excinfo.value).code is ErrorCode.NOTE_NOT_FOUND
        assert [item.id for item in revived.list_feed(FeedQuery(author_id=author_id)).items] == \
            [news.id]
        revived.repository.close()

    def test_in_memory_sqlite_url_is_supported(self, h: Harness) -> None:
        repo = SQLiteNoteRepository("sqlite:///:memory:", settings=h.settings)
        service = NoteService(repo, sessions=h.sessions, directory=h.directory,
                              settings=h.settings, clock=h._clock)
        _author, token = h.make_author()
        note = service.create_note(token, h.input())
        assert service.get_note(None, note.id).id == note.id
        assert isinstance(repo.stream().__iter__(), Iterable)
        repo.close()

    def test_in_memory_repository_orders_and_filters(self, h: Harness) -> None:
        repo = h.repository
        _author, token = h.make_author()
        notes = [h.create(token, title=f"n{index}") for index in range(3)]
        from blacknote.note import NoteFilters

        newest = max(note.created_at for note in notes)
        window = [row.id for row in repo.stream(NoteFilters(before_created_at=newest,
                                                            before_id=min(n.id for n in notes)))]
        assert len(window) == 2
        assert [row.created_at for row in repo.stream()] == sorted(
            [row.created_at for row in repo.stream()], reverse=True
        )


# --------------------------------------------------------------------------- #
# frozen contract conformance
# --------------------------------------------------------------------------- #
class TestContractConformance:
    def test_service_satisfies_all_three_note_protocols(self, h: Harness) -> None:
        from blacknote.contracts.interfaces import NoteService as NoteServiceProtocol

        assert isinstance(h.service, NoteCrudApi)
        assert isinstance(h.service, NoteFeedApi)
        assert isinstance(h.service, NoteExistenceApi)
        assert isinstance(h.service, NoteServiceProtocol)

    def test_frozen_parameter_names_are_preserved(self, h: Harness) -> None:
        expected = {
            "create_note": ["token", "input"],
            "update_note": ["token", "note_id", "input"],
            "delete_note": ["token", "note_id"],
            "get_note": ["token_or_null", "note_id"],
            "list_feed": ["query"],
            "search_notes": ["keyword", "query"],
            "is_publicly_interactable": ["note_id"],
            "get_note_author_id": ["note_id"],
        }
        for name, params in expected.items():
            signature = inspect.signature(getattr(h.service, name))
            positional = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY)
            ]
            assert positional == params, name

    def test_pages_are_serialisable_and_error_free(self, h: Harness) -> None:
        _author, token = h.make_author()
        h.create(token)
        payload = h.service.list_feed(FeedQuery()).to_dict()
        assert json.loads(json.dumps(payload))["items"][0]["counters"] == {
            "likes": 0, "favorites": 0, "comments": 0
        }
        assert payload["total"] == 1

    def test_cursor_round_trip(self) -> None:
        created_at, note_id = to_iso(BASE_MOMENT), new_id()
        token_value = encode_cursor(created_at, note_id)
        assert len(token_value) <= LIMITS["cursor"]["max"]
        assert _decode(token_value) == (created_at, note_id)


def assert_feed_protocol(service: Any) -> bool:  # pragma: no cover - kept for docs
    """Convenience helper retained for readers of the conformance class."""
    return isinstance(service, NoteFeedApi)


def _decode(value: str) -> tuple[str, str]:
    from blacknote.note import decode_cursor

    return decode_cursor(value)


# --------------------------------------------------------------------------- #
# INV-concurrency-counters: the note layer owns content, never the counters
# --------------------------------------------------------------------------- #
class TestCounterIntegrity:
    def test_note_layer_never_touches_interaction_counters(self, h: Harness) -> None:
        _author, token = h.make_author()
        note = h.create(token)
        assert note.counters.to_dict() == {"likes": 0, "favorites": 0, "comments": 0}

        # the interaction layer (wave 2) writes the cache; the note layer must
        # read it through and hand it back untouched
        boosted = h.service.inspect_note_record(note.id).changed(
            counters=Counters(likes=7, favorites=3, comments=11)
        )
        h.repository.save(boosted)
        expected = {"likes": 7, "favorites": 3, "comments": 11}

        h.service.update_note(token, note.id, h.input(title="renamed"))
        h.service.list_feed(FeedQuery())
        h.service.search_notes("renamed", PageQuery())
        h.service.get_note(token, note.id)
        h.service.is_publicly_interactable(note.id)
        h.service.get_note_author_id(note.id)
        assert h.service.inspect_note_record(note.id).counters.to_dict() == expected

        h.service.delete_note(token, note.id)
        assert h.service.inspect_note_record(note.id).counters.to_dict() == expected

    def test_no_counter_mutation_lives_in_the_note_package(self) -> None:
        """Static guard: no ``counters.likes = ...`` and no ``.delta(...)``."""
        mutated: list[str] = []
        for path in sorted(NOTE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr in {"likes", "favorites", "comments"}
                    and isinstance(node.ctx, ast.Store)
                ):
                    mutated.append(f"{path.name}:{node.lineno} assignment")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"delta", "bump", "increment", "decrement"}:
                        mutated.append(f"{path.name}:{node.lineno} counter arithmetic")
        assert mutated == [], mutated


# --------------------------------------------------------------------------- #
# SQL search / tag path (same semantics as the in-memory store)
# --------------------------------------------------------------------------- #
class TestSqliteSearchAndTags:
    def test_multiword_keyword_and_case_insensitive_tags(
        self, h: Harness, tmp_path: Path
    ) -> None:
        repo = SQLiteNoteRepository(f"sqlite:///{tmp_path / 'search.db'}", settings=h.settings)
        service = NoteService(repo, sessions=h.sessions, directory=h.directory,
                              settings=h.settings, clock=h._clock)
        _author, token = h.make_author()
        both = service.create_note(
            token, NoteInput(title="Edited title", body="edited **body** today",
                             tags=["AI工具", "LangGraph"])
        )
        title_only = service.create_note(
            token, NoteInput(title="edited notes", body="nothing to see",
                             tags=["AI工具"])
        )

        phrase = service.search_notes("edited body", PageQuery())
        assert [item.id for item in phrase.items] == [both.id]
        assert phrase.total == 1

        single = service.search_notes("EDITED", PageQuery())
        assert [item.id for item in single.items] == [title_only.id, both.id]

        assert [item.id for item in
                service.list_feed(NoteFeedQuery(tag="langgraph", limit=10)).items] == [both.id]
        assert [item.id for item in
                service.list_feed(NoteFeedQuery(tag="LangGraph", limit=10)).items] == [both.id]
        assert service.search_notes("nothing to see", PageQuery()).total == 1
        assert service.search_notes("missing keyword", PageQuery()).total == 0
        repo.close()


# --------------------------------------------------------------------------- #
# module boundary: mod-note imports the shared contracts, never a sibling module
# --------------------------------------------------------------------------- #
class TestLayerBoundaries:
    def test_served_reads_never_use_the_raw_storage_probe(self) -> None:
        """``inspect_note_record`` is a diagnostic seam, never a read surface."""
        import blacknote.note.service as note_service

        tree = ast.parse(inspect.getsource(note_service))
        served = {
            "get_note", "list_feed", "search_notes",
            "is_publicly_interactable", "get_note_author_id", "_page", "_readable",
        }
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in served:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Attribute) and inner.attr in {
                        "inspect_note_record", "get"
                    }:
                        offenders.append(f"{node.name}:{inner.lineno} -> {inner.attr}")
        assert offenders == [], offenders
        # ... and the probe exists under its diagnostic name only
        assert probe_is_diagnostic_only()

    def test_the_package_only_reaches_the_shared_contracts(self) -> None:
        """``mod-note`` must not import a sibling business module (wave 2+)."""
        allowed_prefixes = ("blacknote.contracts", "blacknote.note")
        offenders: list[str] = []
        for path in sorted(NOTE_DIR.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] if node.level == 0 else []
                else:
                    continue
                for name in names:
                    if name.startswith("blacknote.") and not name.startswith(allowed_prefixes):
                        offenders.append(f"{path.name}:{node.lineno} -> {name}")
        assert offenders == [], offenders


def probe_is_diagnostic_only() -> bool:
    from blacknote.note.service import NoteService

    return callable(NoteService.inspect_note_record) and not hasattr(
        NoteService, "get_note_record"
    )
