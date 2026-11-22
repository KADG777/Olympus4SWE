"""Contract layer self tests (Wave 0 / acceptance T0-C1).

Each test class pins one frozen invariant of ``blacknote.contracts``:

* ``INV-error-envelope``       → :class:`TestErrorEnvelope`
* ``INV-id-format``            → :class:`TestIds`
* ``INV-time-encoding``        → :class:`TestTimeUtil`
* ``INV-visibility-predicate`` → :class:`TestVisibilityPredicate`
* ``INV-auth-session``         → :class:`TestAuthGuard`
* ``INV-role-check``           → :class:`TestRoleCheck`
* ``INV-field-validation``     → :class:`TestFieldValidation`
* ``INV-config-db``            → :class:`TestConfig`
* interface signatures + purity → :class:`TestInterfaces`, :class:`TestContractPurity`
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import uuid
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blacknote.contracts import (
    DEFAULT_DB_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    INTERFACES,
    LIMITS,
    ROLE_HIERARCHY,
    TIMEOUTS,
    UUID4_PATTERN,
    ContractError,
    Counters,
    ContentStatus,
    ErrorCode,
    ErrorEnvelope,
    FeedQuery,
    FieldError,
    ModerationState,
    NoteInput,
    NoteView,
    Page,
    PageQuery,
    Principal,
    ProfilePatch,
    Role,
    SessionToken,
    Settings,
    UserSummary,
    Visibility,
    VisibilityScope,
    as_envelope,
    check_fields,
    filter_visible,
    is_admin,
    is_authenticated,
    is_error,
    is_expired,
    is_iso8601,
    is_valid_id,
    make_error,
    new_id,
    normalize_email,
    normalize_tag,
    now_iso,
    now_utc,
    parse_role,
    require_admin,
    require_auth,
    require_id,
    require_owner_or_admin,
    require_visible,
    status_for,
    to_iso,
    validate_fields,
)
from blacknote.contracts.interfaces import (
    AccountAuthApi,
    AccountProfileApi,
    CommentThread,
    CommentView,
    InteractionApi,
    ModerationApi,
    NoteCrudApi,
    NoteExistenceApi,
    NoteFeedApi,
    NoteSummary,
    ProfileView,
)
from blacknote.contracts.visibility import is_visible_to, visibility_of

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "src" / "blacknote" / "contracts"

AUTHOR_ID = new_id()
OTHER_ID = new_id()
ADMIN_ID = new_id()
NOTE_ID = new_id()
COMMENT_ID = new_id()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_principal(
    user_id: str = AUTHOR_ID,
    role: Role | str = Role.USER,
    token: str | None = None,
    expires_at: str | None = None,
) -> Principal:
    return Principal(
        user_id=user_id,
        token=token if token is not None else new_id(),
        role=role,
        expires_at=expires_at,
    )


def content(
    *,
    author_id: str | None = AUTHOR_ID,
    status: str = "published",
    moderation: str | None = "visible",
    visibility: str = "public",
    **extra: object,
) -> SimpleNamespace:
    payload = {
        "author_id": author_id,
        "status": status,
        "moderation": moderation,
        "visibility": visibility,
        **extra,
    }
    return SimpleNamespace(**payload)


def envelope_of(exc: BaseException) -> ErrorEnvelope:
    assert isinstance(exc, ContractError)
    return exc.envelope


# --------------------------------------------------------------------------- #
# INV-error-envelope
# --------------------------------------------------------------------------- #
class TestErrorEnvelope:
    def test_every_business_code_from_the_design_spec_exists(self) -> None:
        required = {
            "EMAIL_TAKEN",
            "VALIDATION_ERROR",
            "INVALID_CREDENTIALS",
            "AUTH_REQUIRED",
            "PERMISSION_DENIED",
            "USER_NOT_FOUND",
            "NOTE_NOT_FOUND",
            "COMMENT_NOT_FOUND",
        }
        assert required <= {str(code) for code in ErrorCode}

    def test_code_is_string_enum_and_http_mapping_is_deterministic(self) -> None:
        assert ErrorCode.AUTH_REQUIRED == "AUTH_REQUIRED"
        assert str(ErrorCode.NOTE_NOT_FOUND) == "NOTE_NOT_FOUND"
        assert status_for(ErrorCode.AUTH_REQUIRED) == 401
        assert status_for("AUTH_REQUIRED") == 401
        assert status_for(ErrorCode.PERMISSION_DENIED) == 403
        assert status_for(ErrorCode.VALIDATION_ERROR) == 400
        for code in (ErrorCode.NOTE_NOT_FOUND, ErrorCode.COMMENT_NOT_FOUND,
                     ErrorCode.USER_NOT_FOUND):
            assert status_for(code) == 404
        assert status_for(ErrorCode.EMAIL_TAKEN) == 409
        assert status_for("NOT_A_REAL_CODE") == 500  # fail closed

    def test_envelope_shape_is_stable_and_json_safe(self) -> None:
        env = make_error(
            ErrorCode.VALIDATION_ERROR,
            details={"fields": [{"field": "email", "code": "invalid_email", "message": "bad"}]},
        )
        payload = env.to_dict()
        assert set(payload) == {"code", "message", "details", "request_id"} or set(payload) == {
            "code",
            "message",
            "details",
        }
        assert payload["code"] == "VALIDATION_ERROR"
        assert payload["message"]  # never empty
        assert json.loads(json.dumps(payload)) == payload
        assert env.http_status == 400

    def test_default_message_and_round_trip(self) -> None:
        env = make_error(ErrorCode.NOTE_NOT_FOUND)
        assert env.message == ErrorEnvelope.from_dict(env.to_dict()).message
        assert ErrorEnvelope.from_dict(env.to_dict()) == env

    def test_field_errors_are_collected_into_details(self) -> None:
        env = make_error(ErrorCode.VALIDATION_ERROR).with_field_errors(
            [FieldError("title", "too_long", "max 80"), FieldError("body", "required", "missing")]
        )
        assert [fe.field for fe in env.field_errors] == ["title", "body"]
        assert env.details["fields"][0]["code"] == "too_long"

    def test_contract_error_always_carries_an_envelope(self) -> None:
        with pytest.raises(ContractError) as excinfo:
            raise ContractError.validation([FieldError("email", "required", "Email is required")])
        env = envelope_of(excinfo.value)
        assert env.code is ErrorCode.VALIDATION_ERROR
        assert env.details["fields"] == [
            {"field": "email", "code": "required", "message": "Email is required"}
        ]
        assert excinfo.value.http_status == 400
        assert as_envelope(excinfo.value) == env

    def test_envelope_never_leaks_exception_internals(self) -> None:
        env = make_error(ErrorCode.DATABASE_ERROR, "psycopg connection failed\nat frame 0x1")
        assert "Traceback" not in env.to_json()
        assert env.code == "DATABASE_ERROR"


# --------------------------------------------------------------------------- #
# INV-id-format
# --------------------------------------------------------------------------- #
class TestIds:
    def test_new_id_is_canonical_uuid4(self) -> None:
        value = new_id()
        assert isinstance(value, str)
        assert UUID4_PATTERN.match(value)
        assert uuid.UUID(value).version == 4
        assert is_valid_id(value)

    def test_generated_ids_are_unique(self) -> None:
        assert len({new_id() for _ in range(500)}) == 500

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            "",
            "   ",
            12345,
            1.5,
            "not-an-id",
            "00000000-0000-0000-0000-000000000000",  # nil
            str(uuid.uuid1()),                      # wrong version
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "x")),
            new_id().upper(),                       # must be lowercase
            "{" + new_id() + "}",                   # braces
            "urn:uuid:" + new_id(),                 # urn form
            new_id().replace("-", ""),              # no hyphens
            new_id()[:-1],                          # truncated
            new_id() + "extra",
        ],
    )
    def test_rejects_non_canonical_uuid4(self, bad: object) -> None:
        assert is_valid_id(bad) is False

    def test_accepts_canonical_uuid4(self) -> None:
        assert is_valid_id(str(uuid.UUID("6ba7b810-9dad-41d1-80b4-00c04fd430c8"))) is True

    def test_require_id_raises_validation_envelope(self) -> None:
        with pytest.raises(ContractError) as excinfo:
            require_id("nope", field_name="note_id")
        env = envelope_of(excinfo.value)
        assert env.code == "VALIDATION_ERROR"
        assert env.details["fields"][0]["field"] == "note_id"
        assert env.details["fields"][0]["code"] == "invalid_id"

    def test_require_id_returns_input_unchanged(self) -> None:
        value = new_id()
        assert require_id(value, "note_id") == value

    def test_normalize_id_accepts_wrapped_uuid4(self) -> None:
        from blacknote.contracts.ids import normalize_id

        value = new_id()
        assert normalize_id(f"  {value.upper()} ") == value
        assert normalize_id("{" + value + "}") == value
        assert normalize_id("urn:uuid:" + value) == value
        with pytest.raises(ContractError):
            normalize_id("nope")


# --------------------------------------------------------------------------- #
# INV-time-encoding
# --------------------------------------------------------------------------- #
class TestTimeUtil:
    def test_now_iso_is_canonical_utc(self) -> None:
        value = now_iso()
        assert is_iso8601(value, canonical_only=True)
        assert value.endswith("Z")
        assert len(value) == len("2026-01-01T00:00:00.000000Z")

    def test_now_utc_is_aware(self) -> None:
        assert now_utc().tzinfo is not None
        assert now_utc().utcoffset() == timedelta(0)

    def test_to_iso_converts_offsets_and_rejects_naive(self) -> None:
        plus2 = datetime(2026, 2, 14, 11, 30, 0, 123456, tzinfo=timezone(timedelta(hours=2)))
        assert to_iso(plus2) == "2026-02-14T09:30:00.123456Z"
        with pytest.raises(ContractError) as excinfo:
            to_iso(datetime(2026, 2, 14, 9, 30))
        assert envelope_of(excinfo.value).details["fields"][0]["code"] == "naive_datetime"

    def test_from_iso_accepts_z_and_offsets_rejects_naive(self) -> None:
        from blacknote.contracts.timeutil import from_iso

        assert to_iso(from_iso("2026-02-14T09:30:00.000000Z")) == "2026-02-14T09:30:00.000000Z"
        shifted = from_iso("2026-02-14T17:30:00+08:00")
        assert to_iso(shifted) == "2026-02-14T09:30:00.000000Z"
        with pytest.raises(ContractError):
            from_iso("2026-02-14T09:30:00")
        assert to_iso(from_iso("2026-02-14T09:30:00", assume_utc=True)) == (
            "2026-02-14T09:30:00.000000Z"
        )
        with pytest.raises(ContractError):
            from_iso("14/02/2026 09:30")

    def test_round_trip_is_stable(self) -> None:
        from blacknote.contracts.timeutil import from_iso

        moment = now_utc()
        assert from_iso(to_iso(moment)) == moment

    def test_is_expired_semantics(self) -> None:
        past = to_iso(now_utc() - timedelta(seconds=1))
        future = to_iso(now_utc() + timedelta(hours=1))
        assert is_expired(past) is True
        assert is_expired(future) is False
        assert is_expired(None) is False
        assert is_expired("garbage") is True  # fails closed

    def test_lexicographic_order_matches_chronology(self) -> None:
        earlier = to_iso(datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=timezone.utc))
        later = to_iso(datetime(2026, 1, 2, 3, 4, 5, 7, tzinfo=timezone.utc))
        assert earlier < later


# --------------------------------------------------------------------------- #
# INV-visibility-predicate — the single source of truth
# --------------------------------------------------------------------------- #
class TestVisibilityPredicate:
    author = make_principal(AUTHOR_ID, Role.USER)
    stranger = make_principal(OTHER_ID, Role.USER)
    admin = make_principal(ADMIN_ID, Role.ADMIN)

    def test_state_enums_are_frozen_and_canonical(self) -> None:
        assert {str(status) for status in ContentStatus} == {"draft", "published", "deleted"}
        assert {str(state) for state in ModerationState} == {"visible", "unlisted", "blocked"}
        assert {str(audience) for audience in Visibility} == {"public", "private"}
        assert {str(scope) for scope in VisibilityScope} == {"public", "self", "moderation"}

    def test_published_public_note_is_visible_to_everyone(self) -> None:
        live = content()
        assert is_visible_to(None, live) is True
        assert is_visible_to(self.stranger, live) is True
        assert is_visible_to(self.author, live) is True

    def test_draft_is_author_only_and_never_public(self) -> None:
        draft = content(status="draft")
        assert is_visible_to(None, draft) is False
        assert is_visible_to(self.stranger, draft) is False
        assert is_visible_to(self.author, draft, scope=VisibilityScope.SELF) is True
        assert is_visible_to(self.stranger, draft, scope=VisibilityScope.SELF) is False

    def test_deleted_is_invisible_to_everyone_including_admin(self) -> None:
        deleted = content(status="deleted")
        for actor in (None, self.author, self.stranger, self.admin):
            for scope in (VisibilityScope.PUBLIC, VisibilityScope.SELF):
                assert is_visible_to(actor, deleted, scope=scope) is False
        # tombstones are not even revealed in the moderation queue
        assert is_visible_to(self.admin, deleted, scope=VisibilityScope.MODERATION) is False

    def test_moderated_unlisted_note_is_removed_from_public_surfaces(self) -> None:
        unlisted = content(moderation="unlisted")
        assert is_visible_to(None, unlisted) is False
        assert is_visible_to(self.stranger, unlisted) is False
        assert is_visible_to(self.author, unlisted) is False
        assert is_visible_to(self.author, unlisted, scope=VisibilityScope.SELF) is True
        assert is_visible_to(self.admin, unlisted, scope=VisibilityScope.MODERATION) is True

    def test_blocked_comment_is_hidden_for_everyone_but_the_admin_queue(self) -> None:
        blocked = content(status="published", moderation="blocked")
        for actor in (None, self.author, self.stranger):
            assert is_visible_to(actor, blocked) is False
            assert is_visible_to(actor, blocked, scope=VisibilityScope.SELF) is False
        assert is_visible_to(self.admin, blocked, scope=VisibilityScope.MODERATION) is True

    def test_private_visibility_is_author_only(self) -> None:
        private = content(visibility="private")
        assert is_visible_to(None, private) is False
        assert is_visible_to(self.stranger, private) is False
        assert is_visible_to(self.author, private, scope=VisibilityScope.SELF) is True

    @pytest.mark.parametrize("bad_status", [None, "", "archived", 42, "PUBLISHED"])
    def test_unknown_or_missing_status_fails_closed(self, bad_status: object) -> None:
        assert is_visible_to(self.author, content(status=bad_status)) is False

    def test_moderation_scope_requires_admin(self) -> None:
        unlisted = content(moderation="unlisted")
        assert is_visible_to(self.stranger, unlisted, scope=VisibilityScope.MODERATION) is False
        assert is_visible_to(None, unlisted, scope=VisibilityScope.MODERATION) is False
        assert is_visible_to(self.admin, unlisted, scope=VisibilityScope.MODERATION) is True

    def test_none_content_is_never_visible(self) -> None:
        assert is_visible_to(self.admin, None) is False

    def test_missing_moderation_column_defaults_to_visible(self) -> None:
        row = {"author_id": AUTHOR_ID, "status": "published", "visibility": "public"}
        assert is_visible_to(None, row) is True
        facts = visibility_of(row)
        assert facts.author_id == AUTHOR_ID
        assert facts.status == "published"

    def test_filter_visible_is_the_batch_form_of_the_predicate(self) -> None:
        rows = [
            content(),
            content(status="draft"),
            content(status="deleted"),
            content(moderation="unlisted"),
            content(moderation="blocked"),
            content(author_id=OTHER_ID),
        ]
        public = filter_visible(self.stranger, rows)
        assert len(public) == 2
        assert all(is_visible_to(self.stranger, item) for item in public)

    def test_require_visible_never_discloses_existence(self) -> None:
        hidden = content(moderation="unlisted")
        with pytest.raises(ContractError) as excinfo:
            require_visible(self.stranger, hidden, not_found=ErrorCode.NOTE_NOT_FOUND)
        assert envelope_of(excinfo.value).code == ErrorCode.NOTE_NOT_FOUND
        assert envelope_of(excinfo.value).http_status == 404

    def test_scope_accepts_plain_strings(self) -> None:
        assert is_visible_to(self.author, content(status="draft"), scope="self") is True
        assert is_visible_to(self.stranger, content(status="draft"), scope="bogus") is False


# --------------------------------------------------------------------------- #
# INV-auth-session
# --------------------------------------------------------------------------- #
class TestAuthGuard:
    def setup_method(self) -> None:
        self.token = new_id()
        self.session = make_principal(token=self.token)
        self.registry = {self.token: self.session}

    def resolver(self, token: str) -> Principal | None:
        return self.registry.get(token)

    def test_missing_or_blank_token_raises_auth_required(self) -> None:
        for bad in (None, "", "   ", "not-a-uuid", new_id().upper(), 42):
            with pytest.raises(ContractError) as excinfo:
                require_auth(bad, self.resolver)
            assert envelope_of(excinfo.value).code == ErrorCode.AUTH_REQUIRED

    def test_unknown_token_raises_auth_required_not_not_found(self) -> None:
        with pytest.raises(ContractError) as excinfo:
            require_auth(new_id(), self.resolver)
        assert envelope_of(excinfo.value).code == ErrorCode.AUTH_REQUIRED
        assert envelope_of(excinfo.value).http_status == 401

    def test_valid_token_returns_principal(self) -> None:
        principal = require_auth(self.token, self.resolver)
        assert isinstance(principal, Principal)
        assert principal.user_id == AUTHOR_ID
        assert principal.role is Role.USER

    def test_resolver_object_with_resolve_method_is_supported(self) -> None:
        principal = require_auth(self.token, SimpleNamespace(resolve=self.resolver))
        assert principal.user_id == AUTHOR_ID

    def test_expired_session_raises_session_expired(self) -> None:
        expired = make_principal(
            token=self.token, expires_at=to_iso(now_utc() - timedelta(minutes=1))
        )
        with pytest.raises(ContractError) as excinfo:
            require_auth(self.token, lambda _t: expired)
        assert envelope_of(excinfo.value).code == ErrorCode.SESSION_EXPIRED

    def test_write_op_requires_auth_helper(self) -> None:
        assert is_authenticated(self.token, self.resolver) is True
        assert is_authenticated(None, self.resolver) is False
        assert is_authenticated("bogus", self.resolver) is False

    def test_principal_requires_valid_ids_and_hides_token(self) -> None:
        with pytest.raises(ContractError):
            Principal(user_id="nope", token=new_id())
        with pytest.raises(ContractError):
            Principal(user_id=AUTHOR_ID, token="nope")
        principal = make_principal()
        assert principal.token not in str(principal)
        assert "token" not in principal.to_dict()

    def test_require_principal_passthrough(self) -> None:
        from blacknote.contracts.auth_guard import require_principal

        principal = make_principal()
        assert require_principal(principal) is principal
        with pytest.raises(ContractError):
            require_principal(None)

    def test_anonymous_actor_is_none(self) -> None:
        from blacknote.contracts.auth_guard import ANONYMOUS

        assert ANONYMOUS is None


# --------------------------------------------------------------------------- #
# INV-role-check
# --------------------------------------------------------------------------- #
class TestRoleCheck:
    def test_role_enum_and_hierarchy(self) -> None:
        assert {str(role) for role in Role} == {"user", "admin"}
        assert ROLE_HIERARCHY[Role.ADMIN] > ROLE_HIERARCHY[Role.USER]
        assert Role.ADMIN.rank > Role.USER.rank

    def test_require_admin_accepts_admin(self) -> None:
        admin = make_principal(ADMIN_ID, Role.ADMIN)
        assert require_admin(admin) is admin
        assert is_admin(admin) is True

    def test_anonymous_gets_auth_required_before_permission_denied(self) -> None:
        with pytest.raises(ContractError) as excinfo:
            require_admin(None)
        assert envelope_of(excinfo.value).code == ErrorCode.AUTH_REQUIRED

    def test_plain_user_is_denied(self) -> None:
        user = make_principal(AUTHOR_ID, Role.USER)
        with pytest.raises(ContractError) as excinfo:
            require_admin(user)
        env = envelope_of(excinfo.value)
        assert env.code == ErrorCode.PERMISSION_DENIED
        assert env.http_status == 403
        assert env.details["actual_role"] == "user"

    def test_parse_role_rejects_unknown(self) -> None:
        assert parse_role("ADMIN") is Role.ADMIN
        with pytest.raises(ContractError):
            parse_role("superuser")
        with pytest.raises(ContractError):
            parse_role(None)

    def test_is_admin_fails_closed(self) -> None:
        assert is_admin(None) is False
        assert is_admin(SimpleNamespace(user_id=AUTHOR_ID, role="root")) is False
        assert is_admin(SimpleNamespace(user_id=AUTHOR_ID, role="admin")) is True

    def test_owner_or_admin(self) -> None:
        owner = make_principal(AUTHOR_ID, Role.USER)
        stranger = make_principal(OTHER_ID, Role.USER)
        admin = make_principal(ADMIN_ID, Role.ADMIN)
        assert require_owner_or_admin(owner, AUTHOR_ID) is owner
        assert require_owner_or_admin(admin, AUTHOR_ID) is admin
        with pytest.raises(ContractError) as excinfo:
            require_owner_or_admin(stranger, AUTHOR_ID)
        assert envelope_of(excinfo.value).code == ErrorCode.PERMISSION_DENIED
        with pytest.raises(ContractError) as excinfo:
            require_owner_or_admin(None, AUTHOR_ID)
        assert envelope_of(excinfo.value).code == ErrorCode.AUTH_REQUIRED


# --------------------------------------------------------------------------- #
# INV-field-validation
# --------------------------------------------------------------------------- #
class TestFieldValidation:
    def test_limits_cover_every_user_supplied_field(self) -> None:
        for name in (
            "email",
            "password",
            "nickname",
            "bio",
            "title",
            "body",
            "comment_body",
            "tag",
            "tags",
            "images",
            "keyword",
            "page_size",
            "cursor",
        ):
            assert name in LIMITS, name
            assert LIMITS[name]["max"] > 0

    def test_valid_payloads_pass(self) -> None:
        errors = check_fields({"email": "kai@example.com", "password": "s3cret-pass"}, "register",
                              raise_on_error=False)
        assert errors == ()
        assert check_fields({"email": "kai@example.com", "password": "s3cret-pass"}, "login",
                            raise_on_error=False) == ()

    def test_all_problems_are_collected_not_just_the_first(self) -> None:
        errors = check_fields(
            {"email": "not-an-email", "password": "short"}, "register", raise_on_error=False
        )
        assert {e.field for e in errors} == {"email", "password"}

    def test_required_missing_and_blank(self) -> None:
        errors = check_fields({"body": "   "}, "note_input", raise_on_error=False)
        codes = {e.field: e.code for e in errors}
        assert codes["title"] == "required"
        assert codes["body"] == "blank"

    def test_length_bounds_are_enforced(self) -> None:
        long_title = "t" * (LIMITS["title"]["max"] + 1)
        errors = check_fields({"title": long_title, "body": "ok"}, "note_input",
                              raise_on_error=False)
        assert errors[0].code == "too_long"

    def test_check_fields_raises_validation_envelope(self) -> None:
        with pytest.raises(ContractError) as excinfo:
            check_fields({"email": "bad", "password": "x"}, "register")
        env = envelope_of(excinfo.value)
        assert env.code == ErrorCode.VALIDATION_ERROR
        assert isinstance(env.details["fields"], list)
        assert all(set(item) == {"field", "code", "message"} for item in env.details["fields"])
        assert "bad" not in env.to_json()  # values are never echoed back

    def test_partial_patch_semantics(self) -> None:
        assert check_fields({}, "profile_patch", raise_on_error=False, partial=True) == ()
        assert check_fields({"nickname": "Kai"}, "profile_patch", raise_on_error=False,
                            partial=True) == ()
        errors = check_fields({"nickname": "x" * 60}, "profile_patch", raise_on_error=False,
                              partial=True)
        assert [e.field for e in errors] == ["nickname"]

    def test_validate_fields_returns_payload(self) -> None:
        payload = validate_fields({"limit": 10, "cursor": None}, "page_query")
        assert payload["limit"] == 10

    def test_page_size_bounds(self) -> None:
        errors = check_fields({"limit": 500}, "page_query", raise_on_error=False)
        assert errors[0].code == "too_large"
        assert check_fields({"limit": 20}, "page_query", raise_on_error=False) == ()

    def test_email_normalisation(self) -> None:
        assert normalize_email("  Kai@Example.COM ") == "kai@example.com"
        for bad in ("", "kai", "kai@example", "@example.com", "a b@example.com",
                    "x" * 300 + "@example.com"):
            with pytest.raises(ContractError):
                normalize_email(bad)

    def test_tags_support_the_three_fixed_topics_and_custom_ones(self) -> None:
        from blacknote.contracts.validation import FIXED_TOPIC_TAGS, is_known_topic

        assert FIXED_TOPIC_TAGS == ("AI新闻", "AI工具", "AI产品")
        assert normalize_tag("#AI工具") == "AI工具"
        assert normalize_tag("ai-tools") == "AI工具"
        assert normalize_tag("#LangGraph") == "LangGraph"
        assert is_known_topic("AI新闻") is True
        assert is_known_topic("LangGraph") is False

    def test_tag_count_and_shape_limits(self) -> None:
        too_many = [f"tag{i}" for i in range(LIMITS["tags"]["max"] + 1)]
        errors = check_fields({"title": "t", "body": "b", "tags": too_many}, "note_input",
                              raise_on_error=False)
        assert errors[0].code == "too_many_items"
        errors = check_fields({"title": "t", "body": "b", "tags": ["bad tag, with comma"]},
                              "note_input", raise_on_error=False)
        assert errors[0].field.startswith("tags")

    def test_ids_are_validated_by_kind(self) -> None:
        assert check_fields({"author_id": new_id()}, "feed_query", raise_on_error=False) == ()
        errors = check_fields({"author_id": "1"}, "feed_query", raise_on_error=False)
        assert errors[0].code == "invalid_id"


# --------------------------------------------------------------------------- #
# INV-config-db
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_timeout_constants_are_positive_seconds(self) -> None:
        assert DEFAULT_DB_TIMEOUT_SECONDS > 0
        assert DEFAULT_REQUEST_TIMEOUT_SECONDS > 0
        assert DEFAULT_SESSION_TTL_SECONDS > 0
        assert TIMEOUTS["db"] == DEFAULT_DB_TIMEOUT_SECONDS
        assert TIMEOUTS["session_ttl"] == DEFAULT_SESSION_TTL_SECONDS

    def test_default_settings_are_valid_and_persistent(self) -> None:
        settings = Settings()
        assert settings.database_url.startswith("sqlite:///")
        assert settings.is_sqlite
        assert settings.timeout("db") == DEFAULT_DB_TIMEOUT_SECONDS
        assert settings.max_page_size == LIMITS["page_size"]["max"]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"database_url": "data/blacknote.db"},          # missing scheme
            {"database_url": "mongodb://localhost/blacknote"},  # unsupported engine
            {"database_url": "sqlite:///:memory:"},         # non persistent
            {"db_timeout_seconds": 0},
            {"request_timeout_seconds": -1},
            {"session_ttl_seconds": 0},
            {"db_pool_size": 0},
            {"default_page_size": 500},
            {"password_hash_iterations": 10},
        ],
    )
    def test_invalid_configuration_fails_fast(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ContractError) as excinfo:
            Settings(**overrides)  # type: ignore[arg-type]
        assert envelope_of(excinfo.value).code == ErrorCode.CONFIG_ERROR

    def test_in_memory_database_is_allowed_only_for_tests(self) -> None:
        settings = Settings(environment="test", database_url="sqlite:///:memory:")
        assert not settings.is_sqlite or settings.environment == "test"

    def test_production_requires_secrets_and_a_server_database(self) -> None:
        with pytest.raises(ContractError):
            Settings(environment="prod")
        ok = Settings(
            environment="prod",
            database_url="postgresql://bn:pw@db.internal:5432/blacknote",
            secret_key="s" * 32,
            admin_password="Adm1n-strong-pw",
            secure_cookies=True,
        )
        assert ok.is_production

    def test_from_env_coerces_types(self) -> None:
        settings = Settings.from_env(
            {
                "BLACKNOTE_ENVIRONMENT": "test",
                "BLACKNOTE_DATABASE_URL": "sqlite:///:memory:",
                "BLACKNOTE_DB_TIMEOUT_SECONDS": "2.5",
                "BLACKNOTE_SESSION_TTL_SECONDS": "600",
                "BLACKNOTE_DB_ECHO": "true",
                "BLACKNOTE_ALLOWED_IMAGE_TYPES": "image/png, image/webp",
            }
        )
        assert settings.db_timeout_seconds == 2.5
        assert settings.session_ttl_seconds == 600
        assert settings.db_echo is True
        assert settings.allowed_image_types == ("image/png", "image/webp")

    def test_to_dict_redacts_secrets(self) -> None:
        settings = Settings(secret_key="top-secret", admin_password="pw12345678")
        payload = settings.to_dict()
        assert payload["secret_key"] == "***"
        assert payload["admin_password"] == "***"
        assert "top-secret" not in json.dumps(payload)

    def test_settings_are_frozen(self) -> None:
        settings = Settings()
        with pytest.raises(Exception):
            settings.database_url = "sqlite:///:memory:"  # type: ignore[misc]
        clone = settings.replace(db_pool_size=9)
        assert clone.db_pool_size == 9 and settings.db_pool_size != 9

    def test_all_settings_fields_have_defaults(self) -> None:
        names = {f.name for f in fields(Settings)}
        assert {"database_url", "session_ttl_seconds", "db_timeout_seconds"} <= names
        assert Settings()  # constructible without any argument


# --------------------------------------------------------------------------- #
# interfaces.py vs the DesignSpec IF-* list
# --------------------------------------------------------------------------- #
EXPECTED_INTERFACES: dict[str, tuple[list[str], list[str]]] = {
    "IF-account-auth": (
        ["register", "login", "logout", "current_user"],
        ["EMAIL_TAKEN", "VALIDATION_ERROR", "INVALID_CREDENTIALS", "AUTH_REQUIRED"],
    ),
    "IF-account-profile": (
        ["get_profile", "update_profile"],
        ["AUTH_REQUIRED", "PERMISSION_DENIED", "USER_NOT_FOUND", "VALIDATION_ERROR"],
    ),
    "IF-note-crud": (
        ["create_note", "update_note", "delete_note", "get_note"],
        ["AUTH_REQUIRED", "PERMISSION_DENIED", "NOTE_NOT_FOUND", "VALIDATION_ERROR"],
    ),
    "IF-note-feed": (
        ["list_feed", "search_notes"],
        ["VALIDATION_ERROR"],
    ),
    "IF-note-exists-for-interaction": (
        ["is_publicly_interactable", "get_note_author_id"],
        ["NOTE_NOT_FOUND"],
    ),
    "IF-interaction-actions": (
        ["like", "unlike", "favorite", "unfavorite", "comment", "list_comments",
         "list_favorites"],
        ["AUTH_REQUIRED", "NOTE_NOT_FOUND", "COMMENT_NOT_FOUND", "VALIDATION_ERROR"],
    ),
    "IF-moderation-actions": (
        ["unlist_note", "block_comment"],
        ["AUTH_REQUIRED", "PERMISSION_DENIED", "NOTE_NOT_FOUND", "COMMENT_NOT_FOUND"],
    ),
}

EXPECTED_PARAMETER_NAMES: dict[tuple[str, str], list[str]] = {
    ("IF-account-auth", "register"): ["self", "email", "password"],
    ("IF-account-auth", "login"): ["self", "email", "password"],
    ("IF-account-auth", "logout"): ["self", "token"],
    ("IF-account-auth", "current_user"): ["self", "token"],
    ("IF-account-profile", "get_profile"): ["self", "user_id"],
    ("IF-account-profile", "update_profile"): ["self", "token", "patch"],
    ("IF-note-crud", "create_note"): ["self", "token", "input"],
    ("IF-note-crud", "update_note"): ["self", "token", "note_id", "input"],
    ("IF-note-crud", "delete_note"): ["self", "token", "note_id"],
    ("IF-note-crud", "get_note"): ["self", "token_or_null", "note_id"],
    ("IF-note-feed", "list_feed"): ["self", "query"],
    ("IF-note-feed", "search_notes"): ["self", "keyword", "query"],
    ("IF-note-exists-for-interaction", "is_publicly_interactable"): ["self", "note_id"],
    ("IF-note-exists-for-interaction", "get_note_author_id"): ["self", "note_id"],
    ("IF-interaction-actions", "like"): ["self", "token", "note_id"],
    ("IF-interaction-actions", "unlike"): ["self", "token", "note_id"],
    ("IF-interaction-actions", "favorite"): ["self", "token", "note_id"],
    ("IF-interaction-actions", "unfavorite"): ["self", "token", "note_id"],
    ("IF-interaction-actions", "comment"): ["self", "token", "note_id", "parent_comment_id",
                                            "body"],
    ("IF-interaction-actions", "list_comments"): ["self", "note_id"],
    ("IF-interaction-actions", "list_favorites"): ["self", "token", "query"],
    ("IF-moderation-actions", "unlist_note"): ["self", "admin_token", "note_id"],
    ("IF-moderation-actions", "block_comment"): ["self", "admin_token", "comment_id"],
}

EXPECTED_DEPENDENCIES = {
    "IF-account-auth": ("mod-account", ("mod-web",)),
    "IF-account-profile": ("mod-account", ("mod-web",)),
    "IF-note-crud": ("mod-note", ("mod-web",)),
    "IF-note-feed": ("mod-note", ("mod-web",)),
    "IF-note-exists-for-interaction": ("mod-note", ("mod-interaction",)),
    "IF-interaction-actions": ("mod-interaction", ("mod-web",)),
    "IF-moderation-actions": ("mod-moderation", ("mod-web",)),
}


class TestInterfaces:
    def test_registry_covers_every_design_spec_interface(self) -> None:
        assert set(INTERFACES) == set(EXPECTED_INTERFACES)

    @pytest.mark.parametrize("interface_id", sorted(EXPECTED_INTERFACES))
    def test_functions_errors_and_ownership_match_the_spec(
        self, interface_id: str
    ) -> None:
        functions, errors = EXPECTED_INTERFACES[interface_id]
        spec = INTERFACES[interface_id]
        assert list(spec.functions) == functions
        assert list(spec.errors) == errors
        assert all(spec.has_error(code) for code in errors)
        assert spec.producer, spec.consumers
        assert (spec.producer, spec.consumers) == EXPECTED_DEPENDENCIES[interface_id]

    @pytest.mark.parametrize(("interface_id", "function"), sorted(EXPECTED_PARAMETER_NAMES))
    def test_frozen_signatures_match_the_spec(self, interface_id: str, function: str) -> None:
        spec = INTERFACES[interface_id]
        target = getattr(spec.protocol, function)
        params = list(inspect.signature(target).parameters)
        assert params == EXPECTED_PARAMETER_NAMES[(interface_id, function)]

    def test_error_codes_used_in_the_registry_exist(self) -> None:
        known = {str(code) for code in ErrorCode}
        for spec in INTERFACES.values():
            assert set(spec.errors) <= known

    def test_data_types_are_importable_and_named_after_the_spec(self) -> None:
        from blacknote.contracts import interfaces as mod

        for name in ("NoteInput", "FeedQuery", "PageQuery", "Counters", "ProfilePatch",
                     "ProfileView", "NoteView", "NoteSummary", "CommentView", "CommentThread",
                     "UserSummary", "SessionToken", "Page"):
            assert hasattr(mod, name), name

    def test_note_input_applies_the_shared_validation(self) -> None:
        note = NoteInput(
            title=" LangGraph checkpointing ",
            body="## notes\nsome **markdown**",
            tags=["#AI工具", "ai-news", "LangGraph"],
            image_urls=["https://cdn.example/a.png"],
        )
        assert note.title == "LangGraph checkpointing"
        assert note.tags == ("AI工具", "AI新闻", "LangGraph")
        assert note.visibility is Visibility.PUBLIC
        with pytest.raises(ContractError):
            NoteInput(title="", body="x")
        with pytest.raises(ContractError):
            NoteInput(title="t", body="x" * (LIMITS["body"]["max"] + 1))
        with pytest.raises(ContractError):
            NoteInput(title="t", body="x", tags=[f"tag{i}" for i in range(10)])

    def test_note_view_carries_author_identity_and_lifecycle(self) -> None:
        note = NoteView(
            id=NOTE_ID,
            author_id=AUTHOR_ID,
            author_nickname="Kai",
            title="Hello",
            body="World",
            tags=("AI产品",),
            image_urls=("https://cdn.example/a.png",),
            counters=Counters(likes=3, favorites=1, comments=2),
            created_at=now_iso(),
        )
        assert note.author_id == AUTHOR_ID
        assert note.is_publicly_available is True
        assert note.is_visible_to(None) is True
        assert note.summary().cover_image == "https://cdn.example/a.png"
        assert note.to_dict()["status"] == "published"
        moderated = NoteView(
            id=note.id, author_id=AUTHOR_ID, author_nickname="Kai", title="Hello", body="World",
            moderation=ModerationState.UNLISTED,
        )
        assert moderated.is_visible_to(None) is False
        assert moderated.is_visible_to(make_principal(ADMIN_ID, Role.ADMIN),
                                       scope=VisibilityScope.MODERATION) is True

    def test_note_id_must_be_uuid4(self) -> None:
        with pytest.raises(Exception):
            NoteView(id="1", author_id=AUTHOR_ID, author_nickname="Kai", title="t", body="b")

    def test_counters_never_go_negative(self) -> None:
        counters = Counters(likes=1, favorites=0, comments=2)
        assert counters.delta(likes=-1, favorites=-1, comments=-5) == Counters(0, 0, 0)
        with pytest.raises(ValueError):
            Counters(likes=-1)
        assert counters.to_dict() == {"likes": 1, "favorites": 0, "comments": 2}

    def test_page_contract(self) -> None:
        page: Page[NoteSummary] = Page((), next_cursor=None, total=0, limit=20)
        assert list(page) == [] and len(page) == 0 and page.has_more is False
        filled = Page(("a", "b"), next_cursor="cur", total=10)
        assert filled.has_more is True and filled[0] == "a"
        with pytest.raises(ValueError):
            Page(("a",), limit=10_000)

    def test_pagination_queries(self) -> None:
        assert PageQuery().limit == 20
        with pytest.raises(ContractError):
            PageQuery(limit=0)
        query = FeedQuery(viewer_token=None, topic="ai-tools", limit=10)
        assert query.is_anonymous is True
        assert query.topic == "AI工具"
        assert query.to_payload() == {"limit": 10, "topic": "AI工具"}
        with pytest.raises(ValueError):
            FeedQuery(viewer_token="not-a-uuid")

    def test_session_and_user_summary(self) -> None:
        summary = UserSummary(id=AUTHOR_ID, nickname="Kai")
        session = SessionToken(token=new_id(), user=summary, expires_at=now_iso())
        assert session.user_id == AUTHOR_ID
        assert session.to_dict()["token_type"] == "Bearer"
        with pytest.raises(ValueError):
            SessionToken(token="opaque", user=summary)
        with pytest.raises(ContractError):
            UserSummary(id="1", nickname="Kai")
        with pytest.raises(ValueError):
            UserSummary(id=AUTHOR_ID, nickname="Kai", role="root")

    def test_profile_patch_partial(self) -> None:
        assert ProfilePatch().is_empty() is True
        assert ProfilePatch(nickname="Kai").to_payload() == {"nickname": "Kai"}
        with pytest.raises(ContractError):
            ProfilePatch(nickname="x" * 100)
        assert ProfilePatch(bio="").bio == ""

    def test_comment_types_link_back_to_author_and_note(self) -> None:
        root = CommentView(id=COMMENT_ID, note_id=NOTE_ID, author_id=OTHER_ID,
                           author_nickname="Ana", body="nice")
        reply = CommentView(id=new_id(), note_id=NOTE_ID, author_id=AUTHOR_ID,
                            author_nickname="Kai", body="thanks", parent_comment_id=COMMENT_ID)
        thread = CommentThread(root, (reply,))
        assert len(thread) == 2 and reply.is_reply is True
        assert thread.to_dict()["replies"][0]["parent_comment_id"] == COMMENT_ID

    def test_protocols_are_runtime_checkable_for_implementations(self) -> None:
        class FakeBackend:
            def register(self, email: str, password: str): ...
            def login(self, email: str, password: str): ...
            def logout(self, token: str): ...
            def current_user(self, token: str): ...

        assert isinstance(FakeBackend(), AccountAuthApi)
        assert not isinstance(object(), AccountAuthApi)
        for protocol in (AccountProfileApi, NoteCrudApi, NoteFeedApi, NoteExistenceApi,
                         InteractionApi, ModerationApi):
            assert hasattr(protocol, "__protocol_attrs__") or inspect.isclass(protocol)


# --------------------------------------------------------------------------- #
# Boundary hygiene: the contract layer must stay dependency free
# --------------------------------------------------------------------------- #
class TestContractPurity:
    BUSINESS_PACKAGES = (
        "blacknote.account",
        "blacknote.note",
        "blacknote.interaction",
        "blacknote.moderation",
        "blacknote.web",
        "blacknote.storage",
        "blacknote.services",
    )

    def _imports(self, path: Path) -> list[tuple[int, str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.extend((0, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                found.append((node.level or 0, node.module or ""))
        return found

    def test_contract_modules_exist(self) -> None:
        names = {p.name for p in CONTRACTS_DIR.glob("*.py")}
        assert {
            "__init__.py",
            "errors.py",
            "ids.py",
            "timeutil.py",
            "visibility.py",
            "auth_guard.py",
            "roles.py",
            "validation.py",
            "config.py",
            "interfaces.py",
        } <= names

    def test_no_business_or_third_party_imports(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        for path in sorted(CONTRACTS_DIR.rglob("*.py")):
            for level, module in self._imports(path):
                if level or module == "__future__":
                    continue
                root = module.split(".")[0]
                assert root in stdlib, f"{path.name} imports non-stdlib {module!r}"
                assert not module.startswith(self.BUSINESS_PACKAGES), module
                assert module != "blacknote" or path.name == "__init__.py"

    def test_contracts_package_is_importable_standalone(self) -> None:
        import importlib

        module = importlib.import_module("blacknote.contracts")
        assert module.__version__
        assert "is_visible_to" in module.__all__

    def test_single_visibility_predicate_across_the_package(self) -> None:
        # ``is_visible_to`` must be defined exactly once, in visibility.py.
        definers: list[str] = []
        for path in sorted(CONTRACTS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                if isinstance(node, ast.FunctionDef) and node.name == "is_visible_to":
                    definers.append(path.name)
        assert definers == ["visibility.py"]

    def test_error_codes_are_always_referenced_through_the_enum(self) -> None:
        # codes are declared once in errors.py; every other module must raise
        # them via ErrorCode / helpers, never as bare string literals.
        offenders: list[str] = []
        for path in sorted(CONTRACTS_DIR.rglob("*.py")):
            if path.name == "errors.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name not in {"ContractError", "make_error", "status_for"}:
                    continue
                for arg in list(node.args)[:1] + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                            and arg.value.isupper() and "_" in arg.value:
                        offenders.append(f"{path.name}:{node.lineno}:{arg.value}")
        assert offenders == []


INVARIANT_MODULES = {
    "INV-error-envelope": "errors",
    "INV-id-format": "ids",
    "INV-time-encoding": "timeutil",
    "INV-visibility-predicate": "visibility",
    "INV-auth-session": "auth_guard",
    "INV-role-check": "roles",
    "INV-field-validation": "validation",
    "INV-config-db": "config",
}

REQUIRED_PUBLIC_SYMBOLS = {
    "is_visible_to": "visibility",
    "require_auth": "auth_guard",
    "require_admin": "roles",
    "check_fields": "validation",
    "ErrorEnvelope": "errors",
    "ErrorCode": "errors",
    "new_id": "ids",
    "now_iso": "timeutil",
    "Settings": "config",
    "NoteInput": "interfaces",
    "FeedQuery": "interfaces",
    "PageQuery": "interfaces",
    "Counters": "interfaces",
    "ProfilePatch": "interfaces",
}


class TestInvariantCoverage:
    """T0-C2 checklist: every design invariant has a documented owner."""

    @pytest.mark.parametrize(("invariant", "module"), sorted(INVARIANT_MODULES.items()))
    def test_invariant_is_declared_inside_its_own_module(
        self, invariant: str, module: str
    ) -> None:
        path = CONTRACTS_DIR / f"{module}.py"
        assert path.exists(), path
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        assert invariant in doc, f"{module}.py must document {invariant}"

    def test_package_docstring_indexes_all_invariants(self) -> None:
        doc = (CONTRACTS_DIR / "__init__.py").read_text(encoding="utf-8")
        for invariant in INVARIANT_MODULES:
            assert invariant in doc

    @pytest.mark.parametrize(("symbol", "module"), sorted(REQUIRED_PUBLIC_SYMBOLS.items()))
    def test_symbol_is_reexported_from_the_package_root(
        self, symbol: str, module: str
    ) -> None:
        from blacknote import contracts

        assert hasattr(contracts, symbol), symbol
        assert symbol in contracts.__all__
        assert getattr(contracts, symbol) is getattr(
            __import__(f"blacknote.contracts.{module}", fromlist=[module]), symbol
        )

    def test_error_envelope_is_the_only_cross_boundary_error_shape(self) -> None:
        # helpers must return envelopes, never ad-hoc dicts
        assert isinstance(make_error(ErrorCode.NOTE_NOT_FOUND), ErrorEnvelope)
        assert isinstance(as_envelope(ContractError(ErrorCode.NOTE_NOT_FOUND)), ErrorEnvelope)
        assert is_error(ErrorEnvelope(ErrorCode.AUTH_REQUIRED)) is True
        assert is_error(NoteInput(title="t", body="b")) is False


def test_readme_level_smoke_import() -> None:
    """The public API is reachable through the package root."""
    import blacknote
    from blacknote import contracts

    assert blacknote.__version__ == contracts.__version__
