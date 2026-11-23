"""Password hashing and session credential helpers (**INV-auth-session**).

Two rules from the invariant live here:

* **passwords are salted hashes, never plaintext** — PBKDF2-HMAC-SHA256 with a
  fresh 16 byte per-user salt from :mod:`secrets`; the iteration count comes
  from :class:`~blacknote.contracts.config.Settings.password_hash_iterations`
  and is recorded *inside* the stored digest so verification (and a future
  work-factor migration) stays exact;
* **session credentials never travel in the clear** — a session token is an
  opaque UUIDv4 from :func:`blacknote.contracts.ids.new_id`, carrying no data
  by itself (see :mod:`blacknote.contracts.auth_guard`).

Comparison uses :func:`hmac.compare_digest` (constant time) and every helper
fails *closed*: a malformed digest simply verifies as ``False``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Final

from blacknote.contracts.errors import ContractError, ErrorCode, FieldError
from blacknote.contracts.ids import new_id
from blacknote.contracts.validation import LIMITS

__all__ = [
    "ALGORITHM",
    "HASH_SEPARATOR",
    "MIN_ITERATIONS",
    "MAX_ITERATIONS",
    "SALT_BYTES",
    "derive_username",
    "generate_session_token",
    "hash_password",
    "password_needs_rehash",
    "verify_password",
]

ALGORITHM: Final[str] = "pbkdf2_sha256"
HASH_SEPARATOR: Final[str] = "$"
SALT_BYTES: Final[int] = 16
MIN_ITERATIONS: Final[int] = 1_000
MAX_ITERATIONS: Final[int] = 24_000_000

_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<algo>pbkdf2_sha256)\$(?P<iterations>\d+)\$(?P<salt>[0-9a-f]+)\$(?P<hash>[0-9a-f]+)$"
)

_HASH_NAME: Final[str] = "sha256"


def _clamp_iterations(iterations: int) -> int:
    try:
        value = int(iterations)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            ErrorCode.CONFIG_ERROR, "password_hash_iterations must be an integer"
        ) from exc
    if value < MIN_ITERATIONS or value > MAX_ITERATIONS:
        raise ContractError(
            ErrorCode.CONFIG_ERROR,
            f"password_hash_iterations must be between {MIN_ITERATIONS} and {MAX_ITERATIONS}",
        )
    return value


def _normalize_password(password: object) -> bytes:
    if not isinstance(password, str):
        raise ContractError.validation(
            [FieldError("password", "invalid_type", "password must be a string")]
        )
    if not password:
        raise ContractError.validation(
            [FieldError("password", "blank", "password must not be blank")]
        )
    # The length policy itself belongs to contracts.validation (LIMITS["password"]);
    # this layer only refuses structurally unusable input.
    if any(char in password for char in ("\r", "\n", "\t", "\x00")):
        raise ContractError.validation(
            [FieldError("password", "invalid_characters",
                        "password must not contain control characters")]
        )
    return password.encode("utf-8")


def hash_password(password: str, *, iterations: int, salt: bytes | None = None) -> str:
    """Return ``pbkdf2_sha256$<iterations>$<salt-hex>$<hash-hex>``.

    A brand new random salt is used unless one is supplied (only tests need
    that deterministic escape hatch).
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < 8:
        raise ContractError(
            ErrorCode.INTERNAL_ERROR, "password salt must be at least 8 random bytes"
        )
    work = _clamp_iterations(iterations)
    digest = hashlib.pbkdf2_hmac(_HASH_NAME, _normalize_password(password), bytes(salt), work)
    return HASH_SEPARATOR.join((ALGORITHM, str(work), bytes(salt).hex(), digest.hex()))


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time verification of *password* against a stored digest."""
    if not isinstance(stored, str) or not stored:
        return False
    match = _DIGEST_PATTERN.match(stored.strip())
    if match is None:
        return False  # foreign/corrupted digest → fail closed
    try:
        iterations = int(match.group("iterations"))
        salt = bytes.fromhex(match.group("salt"))
        expected = bytes.fromhex(match.group("hash"))
    except ValueError:  # pragma: no cover - guarded by the pattern
        return False
    if iterations < MIN_ITERATIONS or iterations > MAX_ITERATIONS:
        return False
    if not expected or len(salt) < 8:
        return False
    try:
        candidate = hashlib.pbkdf2_hmac(
            _HASH_NAME, _normalize_password(password), salt, iterations
        )
    except ContractError:
        # A structurally unusable password is simply "not the right one".
        return False
    return hmac.compare_digest(candidate, expected)


def password_needs_rehash(stored: str | None, *, iterations: int) -> bool:
    """True when *stored* uses a different (older) work factor."""
    try:
        work = _clamp_iterations(iterations)
    except ContractError:
        return True
    match = _DIGEST_PATTERN.match((stored or "").strip())
    return match is None or int(match.group("iterations")) != work


def derive_username(email: str) -> str:
    """Default display name for a fresh account: the e-mail local part.

    Kept inside ``LIMITS['nickname']`` bounds so it always satisfies the
    frozen validation contract.
    """
    from blacknote.contracts.validation import clean_text

    local = email.split("@", 1)[0] if isinstance(email, str) and "@" in email else ""
    upper = int(LIMITS["nickname"]["max"])
    lower = int(LIMITS["nickname"]["min"])
    text = clean_text(local)[:upper]
    if len(text) < lower:
        text = (text + "member")[:upper]
    return text


def generate_session_token() -> str:
    """Opaque, unpredictable session credential (UUIDv4, per the id contract)."""
    return new_id()
