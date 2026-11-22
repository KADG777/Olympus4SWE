"""BlackNote shared contract layer (Wave 0, read-only for every other layer).

This package is the frozen boundary between the account, note, interaction and
moderation domains and the web assembly layer.  It is deliberately dependency
free apart from the Python standard library: **no module here may import a
business module** (``blacknote.account``, ``blacknote.note``,
``blacknote.interaction``, ``blacknote.moderation``, ``blacknote.web``, ...).
Import direction is strictly ``contracts <- business layers``.

Frozen invariants
-----------------
``INV-error-envelope``      :mod:`~blacknote.contracts.errors`
``INV-id-format``           :mod:`~blacknote.contracts.ids`
``INV-time-encoding``       :mod:`~blacknote.contracts.timeutil`
``INV-visibility-predicate`` :mod:`~blacknote.contracts.visibility`
``INV-auth-session``        :mod:`~blacknote.contracts.auth_guard`
``INV-role-check``          :mod:`~blacknote.contracts.roles`
``INV-field-validation``    :mod:`~blacknote.contracts.validation`
``INV-config-db``           :mod:`~blacknote.contracts.config`
Interface signatures        :mod:`~blacknote.contracts.interfaces`
"""

from __future__ import annotations

from .auth_guard import (
    ANONYMOUS,
    BEARER,
    Principal,
    SessionResolver,
    is_authenticated,
    require_auth,
    require_principal,
    resolve_session,
)
from .config import (
    DEFAULT_DB_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TTL_SECONDS,
    TIMEOUTS,
    Environment,
    Settings,
    load_settings,
)
from .errors import (
    ContractError,
    ErrorCode,
    ErrorEnvelope,
    FieldError,
    as_envelope,
    auth_error,
    is_error,
    make_error,
    not_found_error,
    permission_error,
    status_for,
    validation_error,
)
from .ids import (
    ID_VERSION,
    UUID4_PATTERN,
    generate_id,
    is_valid_id,
    new_id,
    normalize_id,
    parse_id,
    require_id,
)
from .interfaces import (
    INTERFACES,
    CommentThread,
    CommentView,
    Counters,
    FeedQuery,
    InterfaceSpec,
    NoteInput,
    NoteSummary,
    NoteView,
    Page,
    PageQuery,
    ProfilePatch,
    ProfileView,
    SessionToken,
    UserSummary,
    interface_errors,
    interface_functions,
    serialize_payload,
)
from .roles import (
    ROLE_HIERARCHY,
    HasRole,
    Role,
    can_moderate,
    is_admin,
    parse_role,
    require_admin,
    require_owner_or_admin,
    require_role,
)
from .timeutil import (
    ISO8601_UTC_PATTERN,
    from_iso,
    is_expired,
    is_iso8601,
    now_iso,
    now_utc,
    to_iso,
)
from .validation import (
    DEFAULT_PAGE_SIZE,
    FIXED_TOPIC_TAGS,
    LIMITS,
    MAX_PAGE_SIZE,
    SCHEMAS,
    TOPIC_TAG_ALIASES,
    FieldRule,
    check_fields,
    is_known_topic,
    is_valid_email,
    limit_for,
    normalize_email,
    normalize_tag,
    validate_fields,
)
from .visibility import (
    COMMENT_STATUS_VALUES,
    NOTE_STATUS_VALUES,
    ContentStatus,
    ModerationState,
    Visibility,
    VisibilityFacts,
    VisibilityScope,
    filter_visible,
    is_interactable,
    is_visible_to,
    require_visible,
    visibility_of,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # errors
    "ContractError", "ErrorCode", "ErrorEnvelope", "FieldError", "as_envelope", "auth_error",
    "is_error", "make_error", "not_found_error", "permission_error", "status_for",
    "validation_error",
    # ids
    "ID_VERSION", "UUID4_PATTERN", "generate_id", "is_valid_id", "new_id", "normalize_id",
    "parse_id", "require_id",
    # time
    "ISO8601_UTC_PATTERN", "from_iso", "is_expired", "is_iso8601", "now_iso", "now_utc", "to_iso",
    # visibility
    "COMMENT_STATUS_VALUES", "NOTE_STATUS_VALUES", "ContentStatus", "ModerationState",
    "Visibility", "VisibilityFacts", "VisibilityScope", "filter_visible", "is_interactable",
    "is_visible_to", "require_visible", "visibility_of",
    # auth
    "ANONYMOUS", "BEARER", "Principal", "SessionResolver", "is_authenticated", "require_auth",
    "require_principal", "resolve_session",
    # roles
    "ROLE_HIERARCHY", "HasRole", "Role", "can_moderate", "is_admin", "parse_role",
    "require_admin", "require_owner_or_admin", "require_role",
    # validation
    "DEFAULT_PAGE_SIZE", "FIXED_TOPIC_TAGS", "LIMITS", "MAX_PAGE_SIZE", "SCHEMAS",
    "TOPIC_TAG_ALIASES", "FieldRule", "check_fields", "is_known_topic", "is_valid_email",
    "limit_for", "normalize_email", "normalize_tag", "validate_fields",
    # config
    "DEFAULT_DB_TIMEOUT_SECONDS", "DEFAULT_REQUEST_TIMEOUT_SECONDS", "DEFAULT_SESSION_TTL_SECONDS",
    "TIMEOUTS", "Environment", "Settings", "load_settings",
    # interfaces
    "INTERFACES", "InterfaceSpec", "CommentThread", "CommentView", "Counters", "FeedQuery",
    "NoteInput", "NoteSummary", "NoteView", "Page", "PageQuery", "ProfilePatch", "ProfileView",
    "SessionToken", "UserSummary", "interface_errors", "interface_functions",
    "serialize_payload",
]
