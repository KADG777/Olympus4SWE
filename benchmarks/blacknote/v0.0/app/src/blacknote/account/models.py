"""Account row models and their mapping onto the frozen contract DTOs.

A persisted user is *not* the public profile: :class:`UserRecord` keeps the
credential digest and the private e-mail, while the contract value objects
(:class:`~blacknote.contracts.interfaces.UserSummary`,
:class:`~blacknote.contracts.interfaces.ProfileView`,
:class:`~blacknote.contracts.interfaces.SessionToken`) are the only shapes the
rest of the platform may see.  Conversions are declared here once so no other
layer has to guess which column means what.

Profile fields implemented for this domain
``IF-account-profile``
--------------------------------------------------
``username``         display name            → ``ProfileView.nickname``
``bio``              free text               → ``ProfileView.bio``
``avatar_url``       optional avatar         → ``ProfileView.avatar_url``
``specialty``        short "领域/专长" label  → :class:`ProfileDetails`
``affiliation``      short "机构/团队" label   → :class:`ProfileDetails`
``expertise_tags``   list of topic tags      → :class:`ProfileDetails`

The first three are carried by the frozen ``ProfileView``/``ProfilePatch``;
the remaining three are account-owned detail columns surfaced through
:class:`ProfileDetails` (an additive extension that never redefines a contract
shape).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.interfaces import ProfilePatch, ProfileView, UserSummary
from blacknote.contracts.roles import DEFAULT_ROLE, Role, parse_role
from blacknote.contracts.validation import clean_text, normalize_tag

__all__ = [
    "EXTRA_TAG_SEPARATOR",
    "NON_REQUESTABLE_FIELDS",
    "PRIVATE_USER_FIELDS",
    "PROFILE_PATCH_LIMITS",
    "PROFILE_TEXT_FIELDS",
    "PasswordCredential",
    "ProfileDetails",
    "SessionGrant",
    "UserRecord",
    "encode_tags",
    "decode_tags",
    "profile_stats",
    "row_values",
    "user_id_of",
    "PROFILE_STATS_FIELDS",
]

#: Free-text profile columns (single line, trimmed on write).
PROFILE_TEXT_FIELDS: Final[tuple[str, ...]] = ("username", "bio", "specialty", "affiliation")

#: ``patch key → LIMITS bucket``: every size bound stays owned by
#: :mod:`blacknote.contracts.validation` (INV-field-validation).
PROFILE_PATCH_LIMITS: Final[Mapping[str, str]] = {
    "username": "nickname",
    "bio": "bio",
    "avatar_url": "avatar_url",
    "specialty": "nickname",
    "affiliation": "nickname",
}

#: Never accepted from a caller: privilege or credential changes.
NON_REQUESTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"id", "user_id", "role", "email", "password", "password_hash", "created_at"}
)

#: Columns that must never leave the account layer.
PRIVATE_USER_FIELDS: Final[frozenset[str]] = frozenset({"password_hash", "email"})

EXTRA_TAG_SEPARATOR: Final[str] = "|"


def encode_tags(values: Sequence[str] | None) -> str:
    return EXTRA_TAG_SEPARATOR.join(values or ())


def decode_tags(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Sequence[Any] = tuple(raw)
    else:
        text = clean_text(raw)
        items = tuple(part for part in text.split(EXTRA_TAG_SEPARATOR) if part)
    out: list[str] = []
    for item in items:
        try:
            tag = normalize_tag(item)
        except ContractError:
            continue  # a corrupted legacy tag must not break the profile read
        if tag not in out:
            out.append(tag)
    return tuple(out)


def user_id_of(actor: Any) -> str | None:
    """Best-effort ``user_id`` extraction (Principal / record / mapping)."""
    if actor is None:
        return None
    if isinstance(actor, Mapping):
        value = actor.get("user_id") or actor.get("id")
    else:
        value = getattr(actor, "user_id", None) or getattr(actor, "id", None)
    return value if isinstance(value, str) else None


def row_values(row: Any) -> dict[str, Any]:
    """Normalise a DB row (``sqlite3.Row`` or mapping) into a plain dict."""
    if isinstance(row, Mapping):
        return dict(row)
    return dict(row)


#: ``ProfileView`` counters owned by other modules (never computed here).
PROFILE_STATS_FIELDS: Final[tuple[str, ...]] = (
    "notes_count",
    "likes_received",
    "favorites_received",
)


def profile_stats(stats: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Sanitise injected counters: known keys only, non-negative ints, else 0."""
    merged = dict.fromkeys(PROFILE_STATS_FIELDS, 0)
    for key, value in dict(stats or {}).items():
        if key in merged and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class UserRecord:
    """One row of ``users``."""

    id: str
    email: str
    password_hash: str
    role: Role = DEFAULT_ROLE
    username: str = "member"
    bio: str = ""
    avatar_url: str | None = None
    specialty: str = ""
    affiliation: str = ""
    expertise_tags: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", parse_role(self.role))
        object.__setattr__(self, "email", clean_text(self.email).lower())
        object.__setattr__(self, "username", clean_text(self.username) or "member")
        object.__setattr__(self, "expertise_tags", decode_tags(self.expertise_tags))

    # -- structural accessors used by the contract guards ---------------------
    @property
    def user_id(self) -> str:
        return self.id

    @property
    def nickname(self) -> str:
        """Contract alias of :attr:`username`."""
        return self.username

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    # -- projections ----------------------------------------------------------
    def summary(self) -> UserSummary:
        return UserSummary(
            id=self.id,
            nickname=self.username,
            role=str(self.role),
            avatar_url=self.avatar_url,
            created_at=self.created_at,
        )

    def profile_view(self, *, stats: Mapping[str, Any] | None = None) -> ProfileView:
        """Contract :class:`ProfileView`.

        ``notes_count`` / ``likes_received`` / ``favorites_received`` are owned
        by ``mod-note`` and ``mod-interaction``: this layer never reads their
        tables, it only forwards counters handed in by the wiring layer.
        """
        merged = profile_stats(stats)
        return ProfileView(
            id=self.id,
            nickname=self.username,
            bio=self.bio,
            avatar_url=self.avatar_url,
            role=str(self.role),
            created_at=self.created_at,
            **merged,
        )

    def details(self) -> "ProfileDetails":
        return ProfileDetails(
            id=self.id,
            username=self.username,
            bio=self.bio,
            avatar_url=self.avatar_url,
            specialty=self.specialty,
            affiliation=self.affiliation,
            expertise_tags=self.expertise_tags,
            role=str(self.role),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def principal(self, token: str = "") -> Principal:
        return Principal(
            user_id=self.id,
            token=token,
            role=self.role,
            email=self.email,
            nickname=self.username,
        )

    def without_credentials(self) -> "UserRecord":
        """Copy safe to log/serialise (digest and e-mail removed)."""
        return replace(self, password_hash="", email="")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "UserRecord":
        data = row_values(row)
        return cls(
            id=str(data["id"]),
            email=str(data["email"]),
            password_hash=str(data["password_hash"]),
            role=parse_role(data.get("role", DEFAULT_ROLE)),
            username=clean_text(data.get("username")) or "member",
            bio=clean_text(data.get("bio")),
            avatar_url=(None if data.get("avatar_url") in (None, "")
                        else clean_text(data.get("avatar_url"))),
            specialty=clean_text(data.get("specialty")),
            affiliation=clean_text(data.get("affiliation")),
            expertise_tags=decode_tags(data.get("expertise_tags")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": str(self.role),
            "username": self.username,
            "bio": self.bio,
            "avatar_url": self.avatar_url or "",
            "specialty": self.specialty,
            "affiliation": self.affiliation,
            "expertise_tags": encode_tags(self.expertise_tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.without_credentials().to_row()


@dataclass(frozen=True)
class PasswordCredential:
    """A password *before* it becomes a digest — never stored as text."""

    raw: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw, str):
            raise ContractError.validation(
                [FieldError("password", "invalid_type", "password must be a string")]
            )

    def reveal(self) -> str:
        """Explicit accessor: only hashing code may call this."""
        return self.raw


@dataclass(frozen=True)
class SessionGrant:
    """One row of ``sessions`` (token is opaque, carries no payload)."""

    token: str
    user_id: str
    created_at: str
    expires_at: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SessionGrant":
        data = row_values(row)
        return cls(
            token=str(data["token"]),
            user_id=str(data["user_id"]),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        # The token is a bearer credential: it is never serialised.
        return {"user_id": self.user_id, "created_at": self.created_at,
                "expires_at": self.expires_at}


@dataclass(frozen=True)
class ProfileDetails:
    """Account-owned profile payload including the fields ``ProfileView`` lacks."""

    id: str
    username: str = "member"
    bio: str = ""
    avatar_url: str | None = None
    specialty: str = ""
    affiliation: str = ""
    expertise_tags: tuple[str, ...] = ()
    role: str = "user"
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expertise_tags", decode_tags(self.expertise_tags))
        object.__setattr__(self, "username", clean_text(self.username) or "member")

    def to_profile_view(self, *, stats: Mapping[str, Any] | None = None) -> ProfileView:
        return ProfileView(
            id=self.id,
            nickname=self.username,
            bio=self.bio,
            avatar_url=self.avatar_url,
            role=self.role,
            created_at=self.created_at,
            **profile_stats(stats),
        )

    def to_patch(self) -> ProfilePatch:
        return ProfilePatch(
            nickname=self.username,
            bio=self.bio,
            avatar_url=self.avatar_url or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "bio": self.bio,
            "avatar_url": self.avatar_url,
            "specialty": self.specialty,
            "affiliation": self.affiliation,
            "expertise_tags": list(self.expertise_tags),
            "role": self.role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
