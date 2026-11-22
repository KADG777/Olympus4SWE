"""Field validation contract for BlackNote.

Invariant implemented here: **INV-field-validation**.

* ``LIMITS`` is the single source of truth for every user supplied field size
  (email, password, nickname, bio, note title/body, comment body, tags,
  pagination, search keyword, ...).  Services and the web layer must not
  hard-code their own numbers.
* :func:`check_fields` is declarative and **collects all** problems instead of
  failing on the first one; the result is either a tuple of
  :class:`~blacknote.contracts.errors.FieldError` or a raised
  ``VALIDATION_ERROR`` envelope carrying ``details["fields"]``.
* Validation messages never echo the submitted value (no password / email
  leakage into logs or responses).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final, Iterable

from .errors import ContractError, ErrorCode, FieldError
from .ids import is_valid_id
from .visibility import ContentStatus, ModerationState, Visibility

__all__ = [
    "LIMITS",
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "EMAIL_PATTERN",
    "TAG_PATTERN",
    "FIXED_TOPIC_TAGS",
    "TOPIC_TAG_ALIASES",
    "VISIBILITY_VALUES",
    "STATUS_VALUES",
    "MODERATION_VALUES",
    "FieldRule",
    "SCHEMAS",
    "limit_for",
    "resolve_rules",
    "is_known_topic",
    "normalize_tag",
    "normalize_email",
    "is_valid_email",
    "clean_text",
    "check_fields",
    "validate_fields",
]

EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)
TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9_\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,23}$"
)

#: The three fixed discovery topics of the platform (design assumptions A2).
FIXED_TOPIC_TAGS: Final[tuple[str, ...]] = ("AI新闻", "AI工具", "AI产品")
TOPIC_TAG_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ai-news": "AI新闻",
        "ai_news": "AI新闻",
        "ai-tools": "AI工具",
        "ai_tools": "AI工具",
        "ai-products": "AI产品",
        "ai_products": "AI产品",
    }
)

#: ``{field: {"min": ..., "max": ...}}`` — min/max mean character count for
#: text fields and item count for list fields.
_limits: dict[str, dict[str, int]] = {
    "email": {"min": 6, "max": 254},
    "password": {"min": 8, "max": 128},
    "token": {"min": 36, "max": 36},
    "nickname": {"min": 1, "max": 48},
    "bio": {"min": 0, "max": 200},
    "avatar_url": {"min": 0, "max": 512},
    "title": {"min": 1, "max": 80},
    "body": {"min": 1, "max": 5000},
    "excerpt": {"min": 0, "max": 160},
    "comment_body": {"min": 1, "max": 500},
    "tag": {"min": 1, "max": 24},
    "tags": {"min": 0, "max": 5},
    "image_url": {"min": 1, "max": 512},
    "images": {"min": 0, "max": 9},
    "keyword": {"min": 1, "max": 50},
    "cursor": {"min": 0, "max": 256},
    "page_size": {"min": 1, "max": 50},
    "id": {"min": 36, "max": 36},
}
LIMITS: Final[Mapping[str, Mapping[str, int]]] = MappingProxyType(
    {name: MappingProxyType(bounds) for name, bounds in _limits.items()}
)

DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = int(_limits["page_size"]["max"])


def limit_for(name: str) -> tuple[int | None, int | None]:
    """Return ``(min, max)`` bounds registered for *name*."""
    bounds = LIMITS.get(name)
    if bounds is None:
        raise KeyError(f"unknown limit bucket: {name!r}")
    return bounds.get("min"), bounds.get("max")


@dataclass(frozen=True)
class FieldRule:
    """Declarative constraint for one field."""

    name: str
    label: str | None = None
    kind: str = "text"  # text | email | id | enum | int | list | bool
    required: bool = True
    allow_blank: bool = False
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    max_items: int | None = None
    pattern: re.Pattern[str] | None = None
    choices: tuple[str, ...] | None = None
    limit: str | None = None  # LIMITS bucket used when bounds are not inline
    item_limit: str | None = None  # LIMITS bucket applied to every list item
    item_rule: "FieldRule | None" = None

    @classmethod
    def from_limit(
        cls,
        name: str,
        limit: str,
        *,
        kind: str = "text",
        required: bool = True,
        allow_blank: bool = False,
        **kwargs: Any,
    ) -> "FieldRule":
        lo, hi = limit_for(limit)
        return cls(
            name=name,
            kind=kind,
            required=required,
            allow_blank=allow_blank,
            min_length=lo,
            max_length=hi,
            limit=limit,
            **kwargs,
        )

    # -- helpers --------------------------------------------------------------
    @property
    def display_name(self) -> str:
        return self.label or self.name

    def bounds(self) -> tuple[int | None, int | None]:
        if self.min_length is not None or self.max_length is not None:
            return self.min_length, self.max_length
        if self.limit:
            return limit_for(self.limit)
        return None, None

    def item_bounds(self) -> tuple[int | None, int | None]:
        if self.item_rule is not None:
            return self.item_rule.bounds()
        if self.item_limit:
            return limit_for(self.item_limit)
        return None, None

    def max_items_count(self) -> int | None:
        if self.max_items is not None:
            return self.max_items
        if self.limit:
            _lo, hi = limit_for(self.limit)
            return hi
        return None


def _text_rule(name: str, limit: str, **kw: Any) -> FieldRule:
    return FieldRule.from_limit(name, limit, **kw)


#: Enum vocabularies re-exported for the web layer's ``enum=`` form fields.
VISIBILITY_VALUES: Final[tuple[str, ...]] = tuple(str(v) for v in Visibility)
STATUS_VALUES: Final[tuple[str, ...]] = tuple(str(s) for s in ContentStatus)
MODERATION_VALUES: Final[tuple[str, ...]] = tuple(str(m) for m in ModerationState)

SCHEMAS: Final[Mapping[str, tuple[FieldRule, ...]]] = MappingProxyType(
    {
        "register": (
            _text_rule("email", "email", kind="email"),
            _text_rule("password", "password", kind="text"),
        ),
        "login": (
            _text_rule("email", "email", kind="email"),
            _text_rule("password", "password", kind="text"),
        ),
        "profile_patch": (
            _text_rule("nickname", "nickname", required=False),
            _text_rule("bio", "bio", required=False, allow_blank=True),
            _text_rule("avatar_url", "avatar_url", required=False, allow_blank=True),
        ),
        "note_input": (
            _text_rule("title", "title"),
            _text_rule("body", "body"),
            FieldRule(
                name="tags",
                kind="list",
                required=False,
                allow_blank=True,
                item_rule=FieldRule(name="item", min_length=LIMITS["tag"]["min"],
                                    max_length=LIMITS["tag"]["max"], pattern=TAG_PATTERN),
                max_items=int(_limits["tags"]["max"]),
            ),
            FieldRule(
                name="image_urls",
                kind="list",
                required=False,
                allow_blank=True,
                item_rule=FieldRule(name="item", min_length=LIMITS["image_url"]["min"],
                                    max_length=LIMITS["image_url"]["max"]),
                max_items=int(_limits["images"]["max"]),
            ),
            FieldRule(
                name="visibility",
                kind="enum",
                required=False,
                choices=VISIBILITY_VALUES,
            ),
        ),
        "comment_input": (
            _text_rule("body", "comment_body"),
            FieldRule(name="parent_comment_id", kind="id", required=False),
        ),
        "page_query": (
            FieldRule(
                name="limit",
                kind="int",
                required=False,
                limit="page_size",
                minimum=int(_limits["page_size"]["min"]),
                maximum=int(_limits["page_size"]["max"]),
            ),
            _text_rule("cursor", "cursor", required=False, allow_blank=True),
        ),
        "search": (
            _text_rule("keyword", "keyword"),
            FieldRule(
                name="limit",
                kind="int",
                required=False,
                minimum=int(_limits["page_size"]["min"]),
                maximum=int(_limits["page_size"]["max"]),
            ),
            _text_rule("cursor", "cursor", required=False, allow_blank=True),
        ),
        "feed_query": (
            FieldRule(name="topic", kind="text", required=False, allow_blank=True,
                      min_length=1, max_length=24),
            FieldRule(name="author_id", kind="id", required=False),
            FieldRule(
                name="limit",
                kind="int",
                required=False,
                minimum=int(_limits["page_size"]["min"]),
                maximum=int(_limits["page_size"]["max"]),
            ),
            _text_rule("cursor", "cursor", required=False, allow_blank=True),
        ),
    }
)


def is_valid_email(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    lo, hi = limit_for("email")
    if len(text) < (lo or 0) or len(text) > (hi or len(text)):
        return False
    return bool(EMAIL_PATTERN.match(text))


def normalize_email(value: object) -> str:
    """Lower-case + strip an address; raises ``VALIDATION_ERROR`` when invalid."""
    if not isinstance(value, str):
        raise ContractError.validation(
            [FieldError("email", "invalid_email", "must be a string")]
        )
    text = value.strip().lower()
    if not is_valid_email(text):
        raise ContractError.validation(
            [FieldError("email", "invalid_email", "must be a valid email address")]
        )
    return text


def clean_text(value: Any, *, collapse: bool = False) -> str:
    """Trim surrounding whitespace; optionally collapse inner runs."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if collapse:
        text = re.sub(r"\s+", " ", text)
    return text


def normalize_tag(value: object) -> str:
    """Strip a leading ``#``, resolve fixed-topic aliases and validate shape."""
    text = clean_text(value, collapse=True)
    if text.startswith("#"):
        text = text[1:].strip()
    canonical = TOPIC_TAG_ALIASES.get(text.lower(), text)
    lo, hi = limit_for("tag")
    if not canonical or len(canonical) < (lo or 1) or len(canonical) > (hi or 24):
        raise ContractError.validation(
            [FieldError("tags", "invalid_tag", "tag length is out of bounds")]
        )
    if not TAG_PATTERN.match(canonical):
        raise ContractError.validation(
            [FieldError("tags", "invalid_tag", "tag contains unsupported characters")]
        )
    return canonical


def is_known_topic(value: object) -> bool:
    """True for the three fixed topics (custom tags remain allowed)."""
    try:
        tag = normalize_tag(value)
    except ContractError:
        return False
    return tag in FIXED_TOPIC_TAGS


def _missing(data: Mapping[str, Any], rule: FieldRule) -> bool:
    if rule.name not in data:
        return True
    value = data[rule.name]
    return value is None


def _check_item(rule: FieldRule, item: Any, index: int, display: str) -> list[FieldError]:
    """Validate one element of a list field (``tags[i]``, ``image_urls[i]``)."""
    text = clean_text(item, collapse=True)
    if not text:
        return [FieldError(f"{rule.name}[{index}]", "blank",
                           f"{display} items must not be blank")]
    item_rule = rule.item_rule
    if item_rule is None:
        lo, hi = rule.item_bounds()
        item_rule = FieldRule(name=rule.name, min_length=lo, max_length=hi)
    found = _check_one(replace(item_rule, name=f"{rule.name}[{index}]"), text)
    return found


def _check_one(rule: FieldRule, value: Any) -> list[FieldError]:
    errors: list[FieldError] = []
    display = rule.display_name

    if rule.kind == "bool":
        if not isinstance(value, bool):
            errors.append(FieldError(rule.name, "invalid_type", f"{display} must be a boolean"))
        return errors

    if rule.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(FieldError(rule.name, "invalid_type", f"{display} must be an integer"))
            return errors
        lo = rule.minimum if rule.minimum is not None else rule.min_length
        hi = rule.maximum if rule.maximum is not None else rule.max_length
        if lo is not None and value < lo:
            errors.append(FieldError(rule.name, "too_small", f"{display} must be >= {lo}"))
        if hi is not None and value > hi:
            errors.append(FieldError(rule.name, "too_large", f"{display} must be <= {hi}"))
        return errors

    if rule.kind == "id":
        if not is_valid_id(value):
            errors.append(FieldError(rule.name, "invalid_id",
                                     f"{display} must be a canonical UUIDv4 string"))
        return errors

    if rule.kind == "enum":
        text = clean_text(value)
        choices = rule.choices or ()
        if text not in choices:
            errors.append(FieldError(rule.name, "invalid_choice",
                                     f"{display} must be one of: {', '.join(choices)}"))
        return errors

    if rule.kind == "list":
        if isinstance(value, (str, bytes)) or not isinstance(value, (Sequence, set, frozenset)):
            errors.append(FieldError(rule.name, "invalid_type", f"{display} must be a list"))
            return errors
        items = list(value)
        max_items = rule.max_items_count()
        if max_items is not None and len(items) > max_items:
            errors.append(FieldError(rule.name, "too_many_items",
                                     f"{display} accepts at most {max_items} items"))
        for index, item in enumerate(items):
            errors.extend(_check_item(rule, item, index, display))
        return errors

    # text / email
    if not isinstance(value, str):
        errors.append(FieldError(rule.name, "invalid_type", f"{display} must be a string"))
        return errors
    text = value if rule.kind == "password" else clean_text(value, collapse=False)
    if rule.kind == "password" and any(char in value for char in ("\r", "\n", "\t", "\x00")):
        errors.append(FieldError(rule.name, "invalid_characters",
                                 f"{display} must not contain control characters"))
    lo, hi = rule.bounds()
    if not text.strip():
        if not rule.allow_blank:
            errors.append(FieldError(rule.name, "blank", f"{display} must not be blank"))
        return errors
    if lo is not None and len(text) < lo:
        errors.append(FieldError(rule.name, "too_short", f"{display} needs at least {lo} characters"))
    if hi is not None and len(text) > hi:
        errors.append(FieldError(rule.name, "too_long", f"{display} accepts at most {hi} characters"))
    if rule.kind == "email" and not is_valid_email(text):
        errors.append(FieldError(rule.name, "invalid_email", f"{display} must be a valid email address"))
    if rule.pattern is not None and not rule.pattern.match(text):
        errors.append(FieldError(rule.name, "invalid_format", f"{display} has an invalid format"))
    if rule.choices and text not in rule.choices:
        errors.append(FieldError(rule.name, "invalid_choice",
                                 f"{display} must be one of: {', '.join(rule.choices)}"))
    return errors


def resolve_rules(spec: str | FieldRule | Iterable[FieldRule] | Mapping[str, Any]) -> tuple[FieldRule, ...]:
    """Normalise a schema name / rule iterable into a tuple of rules."""
    if isinstance(spec, FieldRule):
        return (spec,)
    if isinstance(spec, str):
        try:
            return SCHEMAS[spec]
        except KeyError:
            raise ContractError(
                ErrorCode.CONFIG_ERROR, f"unknown validation schema: {spec}"
            ) from None
    if isinstance(spec, Mapping):
        rules: list[FieldRule] = []
        for name, constraint in spec.items():
            if isinstance(constraint, FieldRule):
                rules.append(constraint if constraint.name == name else
                             _rename(constraint, name))
            elif isinstance(constraint, Mapping):
                rules.append(FieldRule(name=name, **dict(constraint)))
            elif constraint is None:
                rules.append(_text_rule(name, name))
            else:
                raise TypeError(f"unsupported constraint for {name!r}")
        return tuple(rules)
    return tuple(spec)


def _rename(rule: FieldRule, name: str) -> FieldRule:
    return replace(rule, name=name)


def check_fields(
    data: Mapping[str, Any],
    spec: str | FieldRule | Iterable[FieldRule] | Mapping[str, Any] = "note_input",
    *,
    raise_on_error: bool = True,
    partial: bool = False,
) -> tuple[FieldError, ...]:
    """Validate *data* against *spec*.

    ``partial=True`` downgrades ``required`` violations to "present fields only"
    semantics (used by PATCH style updates such as :class:`ProfilePatch`).
    Returns the collected field errors; raises :class:`ContractError` with a
    ``VALIDATION_ERROR`` envelope when ``raise_on_error`` is set.
    """
    if not isinstance(data, Mapping):
        raise ContractError.validation(
            [FieldError("_body", "invalid_type", "payload must be a mapping")]
        )
    rules = resolve_rules(spec)
    partial = bool(partial)

    errors: list[FieldError] = []
    for rule in rules:
        if _missing(data, rule):
            effective_required = rule.required and not partial
            if effective_required:
                errors.append(FieldError(rule.name, "required",
                                         f"{rule.display_name} is required"))
            continue
        errors.extend(_check_one(rule, data[rule.name]))

    if errors and raise_on_error:
        raise ContractError.validation(errors)
    return tuple(errors)


def validate_fields(
    data: Mapping[str, Any],
    spec: str | FieldRule | Iterable[FieldRule] | Mapping[str, Any] = "note_input",
    *,
    partial: bool = False,
) -> dict[str, Any]:
    """Like :func:`check_fields` but returns the (shallow copied) payload."""
    check_fields(data, spec, raise_on_error=True, partial=partial)
    return dict(data)
