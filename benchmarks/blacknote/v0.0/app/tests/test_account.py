"""Service level tests for ``mod-account`` (TaskNode T1-account, check T1-C1).

Coverage map
------------
``TestRegistration``     有效注册 → 会话可用 / EMAIL_TAKEN / VALIDATION_ERROR
``TestLogin``            登录 → 新会话；INVALID_CREDENTIALS 不泄露存在性
``TestLogout``           登出 → 会话失效 → current_user 返回 AUTH_REQUIRED
``TestSessionGuards``    INV-auth-session 守卫（require_auth / 过期 / 不透明令牌）
``TestProfile``          get_profile / update_profile 立即生效 + 越权守卫
``TestPasswordSecurity`` 加盐哈希存储，库内无明文
``TestRolesAndSeed``     INV-role-check：角色限定 {user, admin}、admin seed、越权日志
``TestPersistence``      INV-config-db：SQLite 连接串 + 重启不丢数据
``TestContractShape``    签名/信封/字段与 contracts/interfaces.py 冻结契约一致
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta, timezone
from pathlib import Path

import pytest

from blacknote.account.db import Database, parse_database_url
from blacknote.account.guards import (
    AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT,
    require_admin_session,
    require_session,
)
from blacknote.account.models import UserRecord
from blacknote.account.repository import AccountRepository
from blacknote.account.seed import DEV_ADMIN_PASSWORD, seed_admin_account
from blacknote.account.security import hash_password, verify_password
from blacknote.account.service import AccountService, create_account_service
from blacknote.contracts.auth_guard import Principal, require_auth
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ErrorEnvelope, ErrorCode
from blacknote.contracts.ids import UUID4_PATTERN, is_valid_id, new_id
from blacknote.contracts.interfaces import (
    INTERFACES,
    AccountAuthApi,
    AccountProfileApi,
    AccountService as AccountServiceProtocol,
    ProfilePatch,
    ProfileView,
    SessionToken,
    UserSummary,
    serialize_payload,
)
from blacknote.contracts.roles import Role, require_admin
from blacknote.contracts.timeutil import ISO8601_UTC_PATTERN, from_iso, now_utc, to_iso
from blacknote.contracts.validation import LIMITS, normalize_email

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
FAST_ITERATIONS = 1_000  # contracts require >= 1000; keeps the suite quick
EMAIL = "engineer@blacknote.dev"
PASSWORD = "correct horse battery"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "environment": "test",
        "database_url": f"sqlite:///{tmp_path / 'blacknote-test.db'}",
        "password_hash_iterations": FAST_ITERATIONS,
        "session_ttl_seconds": 3600,
        "admin_email": "admin@blacknote.local",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def service(settings: Settings) -> AccountService:
    instance = create_account_service(settings)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def registered(service: AccountService):
    """(payload, user_id, token) of one freshly registered account."""
    payload = service.register(EMAIL, PASSWORD)
    assert not isinstance(payload, ErrorEnvelope), payload
    return payload, payload.user.id, payload.token


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #
class TestRegistration:
    def test_valid_register_returns_session_token(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        assert is_valid_id(payload.token)
        assert payload.token_type == "Bearer"
        assert payload.user.role == "user"
        assert is_valid_id(payload.user.id)
        assert payload.expires_at and ISO8601_UTC_PATTERN.match(payload.expires_at)

    def test_register_immediately_opens_a_usable_session(self, service: AccountService) -> None:
        """有效注册 → 登录 → 进入首页所需会话可用。"""
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        summary = service.current_user(payload.token)
        assert isinstance(summary, UserSummary)
        assert summary.id == payload.user.id
        assert summary.role == "user"

    def test_duplicate_email_is_rejected(self, service: AccountService) -> None:
        first = service.register(EMAIL, PASSWORD)
        assert not isinstance(first, ErrorEnvelope)
        duplicate = service.register(EMAIL, "another-password-1")
        assert isinstance(duplicate, ErrorEnvelope)
        assert duplicate.code == ErrorCode.EMAIL_TAKEN
        assert duplicate.http_status == 409

    def test_email_uniqueness_is_normalised(self, service: AccountService) -> None:
        assert not isinstance(service.register("Engineer@BlackNote.dev", PASSWORD), ErrorEnvelope)
        again = service.register("  engineer@blacknote.dev  ", PASSWORD)
        assert isinstance(again, ErrorEnvelope) and again.code == ErrorCode.EMAIL_TAKEN

    @pytest.mark.parametrize(
        "password",
        ["", "short1", "1234567", "a" * (LIMITS["password"]["max"] + 1)],
    )
    def test_weak_password_is_validation_error(
        self, service: AccountService, password: str
    ) -> None:
        """密码 < 8 位 → VALIDATION_ERROR（并在 details 列出字段）。"""
        result = service.register(EMAIL, password)
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR
        fields = [item["field"] for item in result.details["fields"]]
        assert "password" in fields
        # nothing was written: only the seeded admin exists
        assert service.repository.get_user_by_email(normalize_email(EMAIL)) is None
        assert service.repository.count_users() == 1

    @pytest.mark.parametrize("email", ["", "not-an-email", "a@b", "a b@c.com", "x" * 300])
    def test_invalid_email_is_validation_error(self, service: AccountService, email: str) -> None:
        result = service.register(email, PASSWORD)
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR
        assert any(item["field"] == "email" for item in result.details["fields"]), result.to_dict()
        assert service.repository.count_users() == 1  # only the seeded admin
        assert service.repository.live_session_count() == 0

    def test_missing_password_field_is_reported(self, service: AccountService) -> None:
        result = service.register(EMAIL, None)  # type: ignore[arg-type]
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR

    def test_register_envelope_shape_is_the_contract_one(
        self, service: AccountService
    ) -> None:
        result = service.register("bad-email", "short")
        assert isinstance(result, ErrorEnvelope)
        payload = result.to_dict()
        assert set(payload) <= {"code", "message", "details", "request_id"}
        assert payload["message"]
        assert isinstance(payload["details"], dict)


# --------------------------------------------------------------------------- #
# login
# --------------------------------------------------------------------------- #
class TestLogin:
    def test_login_with_valid_credentials(self, service: AccountService) -> None:
        registration = service.register(EMAIL, PASSWORD)
        assert isinstance(registration, SessionToken)
        service.logout(registration.token)

        payload = service.login(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        assert payload.token != registration.token
        assert payload.user.id == registration.user.id
        assert service.current_user(payload.token).id == payload.user.id  # type: ignore[union-attr]

    def test_login_is_case_insensitive_on_email(self, service: AccountService) -> None:
        assert not isinstance(service.register(EMAIL, PASSWORD), ErrorEnvelope)
        payload = service.login("  ENGINEER@BLACKNOTE.DEV  ", PASSWORD)
        assert isinstance(payload, SessionToken)

    def test_wrong_password_is_invalid_credentials(self, service: AccountService) -> None:
        assert not isinstance(service.register(EMAIL, PASSWORD), ErrorEnvelope)
        result = service.login(EMAIL, "wrong password")
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.INVALID_CREDENTIALS
        assert result.http_status == 401

    def test_unknown_email_does_not_leak_existence(self, service: AccountService) -> None:
        missing = service.login("ghost@blacknote.dev", PASSWORD)
        assert isinstance(missing, ErrorEnvelope)
        assert missing.code == ErrorCode.INVALID_CREDENTIALS
        assert not isinstance(service.register(EMAIL, PASSWORD), ErrorEnvelope)
        wrong = service.login(EMAIL, "wrong password")
        assert isinstance(wrong, ErrorEnvelope)
        assert wrong.to_dict() == missing.to_dict()

    def test_login_rejects_malformed_input(self, service: AccountService) -> None:
        result = service.login("nope", "tiny")
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR

    def test_login_opens_a_new_session_per_call(self, service: AccountService) -> None:
        assert not isinstance(service.register(EMAIL, PASSWORD), ErrorEnvelope)
        tokens = {service.login(EMAIL, PASSWORD).token for _ in range(3)}  # type: ignore[union-attr]
        assert len(tokens) == 3
        # one session from register() plus three from login()
        assert service.repository.live_session_count() == 4


# --------------------------------------------------------------------------- #
# logout + current_user
# --------------------------------------------------------------------------- #
class TestLogout:
    def test_logout_invalidates_the_session(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert not isinstance(payload, ErrorEnvelope)
        assert isinstance(service.logout(payload.token), type(None))

        result = service.current_user(payload.token)
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.AUTH_REQUIRED
        assert result.http_status == 401

    def test_logout_is_authorized_only_for_a_live_session(
        self, service: AccountService
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert not isinstance(service.logout(payload.token), ErrorEnvelope)
        again = service.logout(payload.token)
        assert isinstance(again, ErrorEnvelope) and again.code == ErrorCode.AUTH_REQUIRED

    @pytest.mark.parametrize("token", ["", None, "not-a-uuid", "0" * 36, "  "])
    def test_garbage_tokens_are_auth_required(self, service: AccountService, token) -> None:
        result = service.current_user(token)
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.AUTH_REQUIRED
        assert service.logout(token).code == ErrorCode.AUTH_REQUIRED  # type: ignore[union-attr]

    def test_logout_only_touches_the_presented_session(
        self, service: AccountService
    ) -> None:
        first = service.register(EMAIL, PASSWORD)
        second = service.login(EMAIL, PASSWORD)
        assert isinstance(first, SessionToken) and isinstance(second, SessionToken)
        service.logout(first.token)
        assert isinstance(service.current_user(second.token), UserSummary)
        assert isinstance(service.current_user(first.token), ErrorEnvelope)

    def test_current_user_reflects_profile_updates(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        service.update_profile(payload.token, {"username": "kirito"})
        summary = service.current_user(payload.token)
        assert isinstance(summary, UserSummary)
        assert summary.nickname == "kirito"


# --------------------------------------------------------------------------- #
# INV-auth-session guards
# --------------------------------------------------------------------------- #
class TestSessionGuards:
    def test_contract_require_auth_resolves_the_account_service(
        self, service: AccountService, registered
    ) -> None:
        """The account layer is the SessionResolver every other layer calls."""
        payload, user_id, token = registered
        principal = require_auth(token, service)
        assert isinstance(principal, Principal)
        assert principal.user_id == user_id
        assert principal.role is Role.USER
        assert not principal.is_expired()

    def test_require_session_returns_principal(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        principal = require_session(payload.token, service)  # type: ignore[arg-type]
        assert isinstance(principal, Principal)
        assert principal.user_id == payload.user.id  # type: ignore[union-attr]
        assert principal.role is Role.USER
        assert principal.is_admin is False

    def test_require_session_rejects_anonymous(self, service: AccountService) -> None:
        for token in (None, "", "junk"):
            with pytest.raises(Exception) as excinfo:
                require_session(token, service)
            envelope = excinfo.value.to_envelope()  # type: ignore[attr-defined]
            assert envelope.code == ErrorCode.AUTH_REQUIRED

    def test_write_without_session_writes_nothing(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        before = service.get_profile(payload.user.id)
        result = service.update_profile("", ProfilePatch(nickname="hacker"))
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.AUTH_REQUIRED
        assert service.get_profile(payload.user.id) == before

    def test_session_token_is_opaque_uuid4(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        assert UUID4_PATTERN.match(payload.token)
        # An opaque credential carries no payload: no "."-separated JWT segments.
        assert payload.token.count("-") == 4
        assert "." not in payload.token

    def test_expired_session_is_rejected(self, settings: Settings, tmp_path: Path) -> None:
        short = settings.replace(session_ttl_seconds=3600)
        instance = create_account_service(short)
        try:
            payload = instance.register(EMAIL, PASSWORD)
            assert isinstance(payload, SessionToken)
            expired = to_iso(now_utc() - timedelta(seconds=5))
            conn = sqlite3.connect(instance.db.target)
            try:
                conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?",
                             (expired, payload.token))
                conn.commit()
            finally:
                conn.close()
            assert instance.resolve(payload.token) is None
            result = instance.current_user(payload.token)
            assert isinstance(result, ErrorEnvelope)
            assert result.code == ErrorCode.AUTH_REQUIRED
            assert instance.repository.get_session(payload.token) is None  # pruned
        finally:
            instance.close()

    def test_unknown_session_expiration_never_grants_access(
        self, service: AccountService
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        token = payload.token  # type: ignore[union-attr]
        with service.db.transaction():
            service.db.execute(
                "UPDATE sessions SET expires_at = ? WHERE token = ?", ("garbage", token)
            )
        assert service.resolve(token) is None

    def test_principal_and_repr_never_leak_secrets(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        principal = service.authenticate(payload.token)  # type: ignore[arg-type]
        assert EMAIL not in str(principal)
        assert PASSWORD not in str(principal)
        assert payload.token not in principal.to_dict().values()  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #
class TestProfile:
    def test_profile_counters_are_injected_not_computed_here(
        self, settings: Settings
    ) -> None:
        """mod-account forwards other modules' counters, it never reads them."""
        seen: list[str] = []

        def stats(user_id: str) -> dict[str, int]:
            seen.append(user_id)
            return {"notes_count": 7, "likes_received": 3, "favorites_received": 1}

        instance = create_account_service(settings, stats=stats)
        try:
            payload = instance.register(EMAIL, PASSWORD)
            profile = instance.get_profile(payload.user.id)  # type: ignore[union-attr]
            assert isinstance(profile, ProfileView)
            assert (profile.notes_count, profile.likes_received, profile.favorites_received) == (7, 3, 1)
            assert seen == [payload.user.id]
            updated = instance.update_profile(payload.token, {"bio": "hi"})  # type: ignore[arg-type]
            assert isinstance(updated, ProfileView) and updated.notes_count == 7
        finally:
            instance.close()

    def test_profile_counters_default_to_zero_and_reject_junk(
        self, settings: Settings
    ) -> None:
        hostile = create_account_service(
            settings, stats=lambda _: {"notes_count": -3, "likes_received": "many",
                                       "stolen_field": 1, "favorites_received": True}
        )
        try:
            payload = hostile.register(EMAIL, PASSWORD)
            profile = hostile.get_profile(payload.user.id)  # type: ignore[union-attr]
            assert isinstance(profile, ProfileView)
            assert profile.notes_count == 0 and profile.likes_received == 0
            assert profile.favorites_received == 0
            assert "stolen_field" not in profile.to_dict()
        finally:
            hostile.close()

        unwired = create_account_service(settings)
        try:
            payload = unwired.register("plain@blacknote.dev", PASSWORD)
            assert unwired.get_profile(payload.user.id).to_dict()["notes_count"] == 0  # type: ignore[union-attr]
        finally:
            unwired.close()

    def test_get_profile_returns_contract_view(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        profile = service.get_profile(payload.user.id)  # type: ignore[union-attr]
        assert isinstance(profile, ProfileView)
        assert profile.id == payload.user.id
        assert profile.role == "user"
        assert profile.nickname == normalize_email(EMAIL).split("@", 1)[0]

    def test_get_profile_unknown_user(self, service: AccountService) -> None:
        result = service.get_profile(new_id())
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.USER_NOT_FOUND
        assert result.http_status == 404

    def test_get_profile_rejects_malformed_id(self, service: AccountService) -> None:
        result = service.get_profile("42")
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR

    def test_update_profile_is_immediately_readable(
        self, service: AccountService
    ) -> None:
        """update_profile 后 get_profile 立即反映新内容。"""
        payload = service.register(EMAIL, PASSWORD)
        token = payload.token  # type: ignore[union-attr]
        user_id = payload.user.id  # type: ignore[union-attr]

        updated = service.update_profile(token, ProfilePatch(
            nickname="ai-engineer", bio="RAG + agents", avatar_url="https://cdn/bn/1.png"
        ))
        assert isinstance(updated, ProfileView)
        profile = service.get_profile(user_id)
        assert isinstance(profile, ProfileView)
        assert profile.nickname == "ai-engineer"
        assert profile.bio == "RAG + agents"
        assert profile.avatar_url == "https://cdn/bn/1.png"
        assert updated == profile

    def test_extended_identity_fields_round_trip(self, service: AccountService) -> None:
        """username / bio / specialty / affiliation / expertise_tags."""
        payload = service.register(EMAIL, PASSWORD)
        token = payload.token  # type: ignore[union-attr]
        user_id = payload.user.id  # type: ignore[union-attr]

        result = service.update_profile(token, {
            "username": "小铃",
            "bio": "AI 工程师，专注推荐系统",
            "specialty": "推荐系统",
            "affiliation": "BlackNote Labs",
            "expertise_tags": ["AI工具", "llm", "AI工具"],
        })
        assert isinstance(result, ProfileView), result
        details = service.get_profile_details(user_id)
        assert not isinstance(details, ErrorEnvelope)
        assert details.username == "小铃"
        assert details.bio == "AI 工程师，专注推荐系统"
        assert details.specialty == "推荐系统"
        assert details.affiliation == "BlackNote Labs"
        assert details.expertise_tags == ("AI工具", "llm")  # de-duplicated, canonical

        # the contract-facing view reflects the write immediately too
        profile = service.get_profile(user_id)
        assert isinstance(profile, ProfileView)
        assert profile.nickname == "小铃"
        assert profile.bio.startswith("AI 工程师")

    def test_nickname_alias_maps_to_username(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        service.update_profile(payload.token, ProfilePatch(nickname="via-contract"))  # type: ignore[arg-type]
        details = service.get_profile_details(payload.user.id)  # type: ignore[arg-type]
        assert details.username == "via-contract"  # type: ignore[union-attr]

    def test_partial_patch_only_touches_supplied_fields(
        self, service: AccountService
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        token, user_id = payload.token, payload.user.id  # type: ignore[var-annotated]
        service.update_profile(token, {"specialty": "NLP"})  # type: ignore[arg-type]
        first = service.get_profile_details(user_id)
        service.update_profile(token, {"affiliation": "TU"})  # type: ignore[arg-type]
        second = service.get_profile_details(user_id)
        assert first.specialty == "NLP" and second.specialty == "NLP"  # type: ignore[union-attr]
        assert second.affiliation == "TU"  # type: ignore[union-attr]

    def test_profile_update_bumps_updated_at(self, service: AccountService) -> None:
        import time

        payload = service.register(EMAIL, PASSWORD)
        before = service.get_profile_details(payload.user.id)  # type: ignore[arg-type]
        time.sleep(0.01)  # the UTC clock is microsecond precise; keep it deterministic
        service.update_profile(payload.token, {"bio": "hello"})  # type: ignore[arg-type]
        after = service.get_profile_details(payload.user.id)  # type: ignore[arg-type]
        assert after.updated_at > before.updated_at
        assert ISO8601_UTC_PATTERN.match(after.updated_at)

    @pytest.mark.parametrize(
        "patch, field",
        [
            ({"bio": "x" * (LIMITS["bio"]["max"] + 1)}, "bio"),
            ({"username": ""}, "username"),
            ({"username": "x" * (LIMITS["nickname"]["max"] + 1)}, "username"),
            ({"specialty": "x" * (LIMITS["nickname"]["max"] + 1)}, "specialty"),
            ({"affiliation": "x" * (LIMITS["nickname"]["max"] + 1)}, "affiliation"),
            ({"expertise_tags": ["x" * (LIMITS["tag"]["max"] + 1)]}, "expertise_tags[0]"),
            ({"expertise_tags": ["bad tag!"]}, "expertise_tags[0]"),
            ({"expertise_tags": ["t" for _ in range(LIMITS["tags"]["max"] + 1)]},
             "expertise_tags"),
            ({"expertise_tags": "not-a-list"}, "expertise_tags"),
            ({"unknown_field": "x"}, "unknown_field"),
            ({}, "profile"),
        ],
    )
    def test_invalid_profile_patches(
        self, service: AccountService, patch: dict, field: str
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        result = service.update_profile(payload.token, patch)  # type: ignore[arg-type]
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.VALIDATION_ERROR
        assert any(item["field"] == field for item in result.details["fields"]), result.to_dict()

    def test_profile_patch_cannot_escalate_role(self, service: AccountService) -> None:
        """INV-role-check: 个人资料写入不能提权。"""
        payload = service.register(EMAIL, PASSWORD)
        result = service.update_profile(payload.token, {"role": "admin"})  # type: ignore[arg-type]
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.PERMISSION_DENIED
        assert result.http_status == 403
        still = service.get_profile(payload.user.id)  # type: ignore[arg-type]
        assert isinstance(still, ProfileView) and still.role == "user"

    def test_profile_patch_cannot_change_email_or_password(
        self, service: AccountService
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        for field in ("email", "password"):
            result = service.update_profile(payload.token, {field: "whatever@x.io"})  # type: ignore[arg-type]
            assert isinstance(result, ErrorEnvelope)
            assert result.code == ErrorCode.VALIDATION_ERROR
            assert result.details["fields"][0]["field"] == field

    def test_one_user_cannot_edit_another_profile(
        self, service: AccountService
    ) -> None:
        alice = service.register("alice@blacknote.dev", PASSWORD)
        bob = service.register("bob@blacknote.dev", PASSWORD)
        assert isinstance(alice, SessionToken) and isinstance(bob, SessionToken)
        service.update_profile(bob.token, {"username": "bob-verified"})
        alice_profile = service.get_profile(alice.user.id)
        assert isinstance(alice_profile, ProfileView)
        assert alice_profile.nickname != "bob-verified"
        assert alice_profile.nickname == "alice"


# --------------------------------------------------------------------------- #
# password storage
# --------------------------------------------------------------------------- #
class TestPasswordSecurity:
    def test_password_is_stored_as_a_salted_hash(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        record = service.repository.get_user_by_id(payload.user.id)  # type: ignore[union-attr]
        assert isinstance(record, UserRecord)
        assert record.password_hash != PASSWORD
        assert PASSWORD not in record.password_hash
        assert record.password_hash.startswith(f"pbkdf2_sha256${FAST_ITERATIONS}$")
        assert verify_password(PASSWORD, record.password_hash)
        assert not verify_password(PASSWORD + "x", record.password_hash)

    def test_identical_passwords_produce_distinct_salts_and_hashes(
        self, service: AccountService
    ) -> None:
        for address in ("one@blacknote.dev", "two@blacknote.dev"):
            assert not isinstance(service.register(address, PASSWORD), ErrorEnvelope)
        rows = [
            row for row in service.repository.list_users(limit=10)
            if row.email in {"one@blacknote.dev", "two@blacknote.dev"}
        ]
        assert len(rows) == 2
        digests = {row.password_hash for row in rows}
        salts = {row.password_hash.split("$")[2] for row in rows}
        assert len(digests) == 2, "per-user random salt must differentiate digests"
        assert len(salts) == 2

    def test_database_file_contains_no_plaintext_password(
        self, settings: Settings
    ) -> None:
        instance = create_account_service(settings)
        try:
            instance.register(EMAIL, PASSWORD)
            instance.db.close()
            base = Path(parse_database_url(settings.database_url))
            blob = b"".join(
                candidate.read_bytes()
                for candidate in (base, base.with_suffix(base.suffix + "-wal"))
                if candidate.exists()
            )
        finally:
            instance.close()
        assert PASSWORD.encode() not in blob
        assert EMAIL.encode() in blob  # the e-mail itself is a plain column

    def test_tampered_digest_fails_closed(self) -> None:
        digest = hash_password(PASSWORD, iterations=FAST_ITERATIONS)
        assert verify_password(PASSWORD, digest)
        assert not verify_password(PASSWORD, digest[:-2] + "ff")
        assert not verify_password(PASSWORD, "plaintext-ish")
        assert not verify_password(PASSWORD, None)
        assert not verify_password("", digest)

    def test_login_rehashes_a_stale_work_factor(self, settings: Settings) -> None:
        instance = create_account_service(settings.replace(password_hash_iterations=FAST_ITERATIONS))
        try:
            legacy = hash_password(PASSWORD, iterations=FAST_ITERATIONS + 1_000)
            record = UserRecord(
                id=new_id(), email="legacy@blacknote.dev", password_hash=legacy,
                created_at=to_iso(now_utc()), updated_at=to_iso(now_utc()),
            )
            instance.repository.insert_user(record)
            payload = instance.login("legacy@blacknote.dev", PASSWORD)
            assert isinstance(payload, SessionToken)
            stored = instance.repository.get_user_by_email("legacy@blacknote.dev")
            assert stored.password_hash != legacy  # type: ignore[union-attr]
            assert verify_password(PASSWORD, stored.password_hash)  # type: ignore[arg-type]
        finally:
            instance.close()


# --------------------------------------------------------------------------- #
# roles + admin seed
# --------------------------------------------------------------------------- #
class TestRolesAndSeed:
    def test_role_vocabulary_is_frozen(self) -> None:
        assert {str(role) for role in Role} == {"user", "admin"}

    def test_admin_account_is_seeded_by_the_service_factory(
        self, service: AccountService, settings: Settings
    ) -> None:
        record = service.repository.get_user_by_email(settings.admin_email)
        assert isinstance(record, UserRecord)
        assert record.role is Role.ADMIN
        payload = service.login(settings.admin_email, DEV_ADMIN_PASSWORD)
        assert isinstance(payload, SessionToken)
        assert payload.user.role == "admin"
        principal = service.authenticate(payload.token)
        assert isinstance(principal, Principal) and principal.is_admin

    def test_admin_password_can_be_configured(self, tmp_path: Path) -> None:
        settings = make_settings(
            tmp_path, admin_password="configured-admin-password", environment="dev"
        )
        instance = create_account_service(settings)
        try:
            payload = instance.login(settings.admin_email, "configured-admin-password")
            assert isinstance(payload, SessionToken)
            assert instance.login(settings.admin_email, DEV_ADMIN_PASSWORD).code == (
                ErrorCode.INVALID_CREDENTIALS
            )
        finally:
            instance.close()

    def test_seeding_is_idempotent_and_opens_no_session(
        self, service: AccountService, settings: Settings
    ) -> None:
        before_users = service.repository.count_users()
        before_sessions = service.repository.live_session_count()
        again = seed_admin_account(service)
        third = service.seed_admin()
        assert service.repository.count_users() == before_users
        assert again.id == third.id
        assert service.repository.live_session_count() == before_sessions

    def test_seed_can_promote_an_existing_account(
        self, service: AccountService, settings: Settings
    ) -> None:
        assert not isinstance(
            service.register("ops@blacknote.dev", PASSWORD), ErrorEnvelope
        )
        promoted = seed_admin_account(service, email="ops@blacknote.dev", password=PASSWORD)
        assert promoted.role is Role.ADMIN
        assert isinstance(service.login("ops@blacknote.dev", PASSWORD), SessionToken)

    def test_admin_guard_orders_authz_errors(self, service: AccountService) -> None:
        member = service.register(EMAIL, PASSWORD)
        with pytest.raises(Exception) as anonymous:
            require_admin_session(None, service)
        assert anonymous.value.to_envelope().code == ErrorCode.AUTH_REQUIRED  # type: ignore[attr-defined]

        denied = service.list_users(member.token)  # type: ignore[arg-type]
        assert isinstance(denied, ErrorEnvelope)
        assert denied.code == ErrorCode.PERMISSION_DENIED

    def test_unauthorized_admin_attempt_is_logged_exactly_once(
        self, service: AccountService, caplog: pytest.LogCaptureFixture
    ) -> None:
        member = service.register(EMAIL, PASSWORD)
        with caplog.at_level("WARNING", logger="blacknote.security"):
            result = service.list_users(member.token)  # type: ignore[arg-type]
        assert isinstance(result, ErrorEnvelope)
        assert result.code == ErrorCode.PERMISSION_DENIED
        records = [
            item for item in caplog.records
            if AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT in item.getMessage()
        ]
        assert len(records) == 1
        assert member.token not in records[0].getMessage()  # type: ignore[union-attr]
        audit = service.repository.audit_rows(AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT)
        assert len(audit) == 1
        assert audit[0]["actor_id"] == member.user.id  # type: ignore[union-attr]

    def test_anonymous_admin_call_writes_no_attempt_log(
        self, service: AccountService, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="blacknote.security"):
            result = service.list_users("")
        assert isinstance(result, ErrorEnvelope) and result.code == ErrorCode.AUTH_REQUIRED
        assert AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT not in caplog.text
        assert service.repository.audit_rows(AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT) == []

    def test_admin_can_list_the_directory(self, service: AccountService,
                                          settings: Settings) -> None:
        member = service.register(EMAIL, PASSWORD)
        assert isinstance(member, SessionToken)
        admin = service.login(settings.admin_email, DEV_ADMIN_PASSWORD)
        assert isinstance(admin, SessionToken)
        listed = service.list_users(admin.token, limit=10)
        assert isinstance(listed, list) and listed
        identities = {item.id: item for item in listed}
        assert {admin.user.id, member.user.id} <= set(identities)
        assert all(is_valid_id(item.id) for item in listed)
        assert {item.role for item in listed} <= {"user", "admin"}
        assert identities[admin.user.id].role == "admin"

    def test_schema_constraints_reject_a_third_role(self, service: AccountService) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            with service.db.transaction():
                service.db.execute(
                    "INSERT INTO users (id, email, password_hash, role, username, created_at,"
                    " updated_at) VALUES (?, ?, ?, 'superuser', 'x', ?, ?)",
                    (new_id(), "su@blacknote.dev", "hash", to_iso(now_utc()), to_iso(now_utc())),
                )

    def test_repository_translates_email_conflict_into_email_taken(
        self, service: AccountService
    ) -> None:
        payload = service.register(EMAIL, PASSWORD)
        record = service.repository.get_user_by_id(payload.user.id)  # type: ignore[union-attr]
        with pytest.raises(Exception) as excinfo:
            service.repository.insert_user(
                UserRecord(
                    id=new_id(), email=record.email, password_hash=record.password_hash,
                    created_at=to_iso(now_utc()), updated_at=to_iso(now_utc()),
                )
            )
        assert excinfo.value.to_envelope().code == ErrorCode.EMAIL_TAKEN  # type: ignore[attr-defined]

    def test_require_admin_on_a_resolved_principal(self, service: AccountService,
                                                   settings: Settings) -> None:
        admin = service.login(settings.admin_email, DEV_ADMIN_PASSWORD)
        principal = service.authenticate(admin.token)  # type: ignore[arg-type]
        assert require_admin(principal) is principal
        member = service.register(EMAIL, PASSWORD)
        with pytest.raises(Exception) as excinfo:
            require_admin(service.authenticate(member.token))  # type: ignore[arg-type]
        assert excinfo.value.to_envelope().code == ErrorCode.PERMISSION_DENIED  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
class TestPersistence:
    def test_sqlite_wiring_comes_from_settings_only(self, tmp_path: Path) -> None:
        """INV-config-db: the layer reads Settings.database_url, nothing else."""
        settings = make_settings(tmp_path)
        db = Database(settings)
        try:
            db.initialize()
            db.initialize()  # idempotent DDL
            assert db.schema_version() == 1
            assert {"users", "sessions", "auth_audit", "schema_meta"} <= db.table_names()
            assert db.is_memory is False
            assert db.file_path == tmp_path / "blacknote-test.db"
            assert db.settings.database_url == settings.database_url
        finally:
            db.close()

    def test_in_memory_database_is_supported_for_probes(self) -> None:
        db = Database(Settings(environment="test", database_url="sqlite://",
                               password_hash_iterations=FAST_ITERATIONS))
        try:
            db.initialize()
            assert db.is_memory and db.file_path is None
            assert db.scalar("SELECT COUNT(*) FROM users") == 0
        finally:
            db.close()

    def test_accounts_and_sessions_survive_a_restart(self, settings: Settings) -> None:
        first = create_account_service(settings)
        try:
            payload = first.register(EMAIL, PASSWORD)
            assert isinstance(payload, SessionToken)
            first.update_profile(payload.token, {"specialty": "MLOps", "bio": "notes"})  # type: ignore[arg-type]
        finally:
            first.close()

        second = create_account_service(settings)
        try:
            assert isinstance(second.current_user(payload.token), UserSummary)  # type: ignore[arg-type]
            profile = second.get_profile(payload.user.id)  # type: ignore[arg-type]
            assert isinstance(profile, ProfileView)
            assert profile.bio == "notes"
            details = second.get_profile_details(payload.user.id)  # type: ignore[arg-type]
            assert details.specialty == "MLOps"
            # the profile write survived, and the rehash-free credential still works
            assert isinstance(second.login(EMAIL, PASSWORD), SessionToken)
            # seeding stayed idempotent across the restart
            assert second.repository.count_users() == 2
        finally:
            second.close()

    def test_logout_persisted_across_restart(self, settings: Settings) -> None:
        first = create_account_service(settings)
        try:
            payload = first.register(EMAIL, PASSWORD)
            first.logout(payload.token)  # type: ignore[arg-type]
        finally:
            first.close()
        second = create_account_service(settings)
        try:
            result = second.current_user(payload.token)  # type: ignore[arg-type]
            assert isinstance(result, ErrorEnvelope) and result.code == ErrorCode.AUTH_REQUIRED
        finally:
            second.close()

    def test_data_is_written_to_the_configured_sqlite_file(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        assert settings.database_url.startswith("sqlite:///")
        assert parse_database_url(settings.database_url) == str(
            tmp_path / "blacknote-test.db"
        )
        instance = create_account_service(settings)
        try:
            assert instance.db.file_path is not None and instance.db.file_path.exists()
            assert {"users", "sessions", "auth_audit"} <= instance.db.table_names()
            assert instance.db.schema_version() == 1
        finally:
            instance.close()

    def test_missing_parent_directory_is_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "blacknote.db"
        instance = create_account_service(
            Settings(environment="test", database_url=f"sqlite:///{nested}",
                     password_hash_iterations=FAST_ITERATIONS)
        )
        try:
            assert nested.parent.exists()
            assert not isinstance(instance.register(EMAIL, PASSWORD), ErrorEnvelope)
        finally:
            instance.close()

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("sqlite://", ":memory:"),
            ("sqlite:///:memory:", ":memory:"),
            ("sqlite:///data/blacknote.db", "data/blacknote.db"),
            ("sqlite:////tmp/bn/blacknote.db", "/tmp/bn/blacknote.db"),
            ("sqlite+aiosqlite:///data/b.db", "data/b.db"),
        ],
    )
    def test_database_url_parsing(self, url: str, expected: str) -> None:
        assert parse_database_url(url) == expected

    @pytest.mark.parametrize("url", ["mysql://localhost/x", "not-a-url", "postgres://u@h/db"])
    def test_unsupported_database_url_is_config_error(self, url: str) -> None:
        from blacknote.contracts.errors import ContractError

        with pytest.raises(ContractError) as excinfo:
            parse_database_url(url)
        assert excinfo.value.code == ErrorCode.CONFIG_ERROR

    def test_repository_requires_a_real_database(self, settings: Settings) -> None:
        from blacknote.contracts.errors import ContractError

        with pytest.raises(ContractError):
            AccountRepository(settings)  # type: ignore[arg-type]
        with pytest.raises(ContractError):
            AccountService("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# contract conformance (T1-C2 review support)
# --------------------------------------------------------------------------- #
EXPECTED_SIGNATURES = {
    "register": ["self", "email", "password"],
    "login": ["self", "email", "password"],
    "logout": ["self", "token"],
    "current_user": ["self", "token"],
    "get_profile": ["self", "user_id"],
    "update_profile": ["self", "token", "patch"],
}


class TestContractShape:
    def test_service_implements_the_frozen_protocols(self, service: AccountService) -> None:
        assert isinstance(service, AccountAuthApi)
        assert isinstance(service, AccountProfileApi)
        assert isinstance(service, AccountServiceProtocol)

    def test_signatures_match_the_registry(self) -> None:
        import inspect

        seen = set()
        for interface_id in ("IF-account-auth", "IF-account-profile"):
            for name in INTERFACES[interface_id].functions:
                params = list(inspect.signature(getattr(AccountService, name)).parameters)
                assert params == EXPECTED_SIGNATURES[name], name
                seen.add(name)
        assert seen == set(EXPECTED_SIGNATURES)

    def test_only_contract_error_codes_are_emitted(self, service: AccountService) -> None:
        known = {str(member) for member in ErrorCode}
        probes = [
            service.register("bad", "short"),
            service.login("bad", "short"),
            service.current_user("junk"),
            service.get_profile("junk"),
            service.update_profile("junk", {"bio": "x"}),
            service.logout("junk"),
            service.update_profile(service.register(EMAIL, PASSWORD).token, {"role": "admin"}),  # type: ignore[union-attr]
        ]
        for probe in probes:
            assert isinstance(probe, ErrorEnvelope), probe
            assert probe.code in known
            assert "error" not in probe.to_dict()  # envelope is the payload

    def test_success_payloads_carry_no_error_field(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        serialized = serialize_payload(payload)
        assert "error" not in serialized
        profile = serialize_payload(service.get_profile(payload.user.id))
        assert "error" not in profile
        details = serialize_payload(service.get_profile_details(payload.user.id))  # type: ignore[arg-type]
        assert "error" not in details
        assert set(details) == {
            "id", "username", "bio", "avatar_url", "specialty", "affiliation",
            "expertise_tags", "role", "created_at", "updated_at",
        }

    def test_timestamps_are_canonical_utc(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        assert ISO8601_UTC_PATTERN.match(payload.expires_at)
        record = service.repository.get_user_by_id(payload.user.id)  # type: ignore[union-attr]
        assert ISO8601_UTC_PATTERN.match(record.created_at)  # type: ignore[union-attr]
        assert from_iso(record.created_at).tzinfo is timezone.utc  # type: ignore[arg-type]

    def test_all_ids_are_uuid4(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        assert isinstance(payload, SessionToken)
        assert is_valid_id(payload.user.id) and is_valid_id(payload.token)
        assert len(payload.token) == 36

    def test_no_plaintext_secret_in_the_safe_projection(self, service: AccountService) -> None:
        payload = service.register("quiet@blacknote.dev", PASSWORD)
        stored = service.repository.get_user_by_email("quiet@blacknote.dev")
        assert isinstance(stored, UserRecord)
        safe = stored.without_credentials()
        assert PASSWORD not in repr(safe)
        assert "quiet@blacknote.dev" not in repr(safe)
        assert safe.password_hash == "" and safe.email == ""
        assert stored.summary().to_dict() == payload.user.to_dict()  # type: ignore[union-attr]

    def test_account_layer_never_imports_other_business_layers(self) -> None:
        import ast

        root = Path(__file__).resolve().parents[1] / "src" / "blacknote" / "account"
        forbidden = ("blacknote.note", "blacknote.interaction", "blacknote.moderation",
                     "blacknote.web")
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    assert not name.startswith(forbidden), f"{path.name} imports {name}"

    def test_email_is_not_exposed_by_the_public_profile(self, service: AccountService) -> None:
        payload = service.register(EMAIL, PASSWORD)
        profile = service.get_profile(payload.user.id)  # type: ignore[union-attr]
        assert EMAIL not in str(profile)
        assert not hasattr(profile, "email")
        assert EMAIL not in str(service.current_user(payload.token))
