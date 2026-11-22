"""Note level input rules layered on top of the frozen field validation.

``INV-field-validation`` keeps every *size* limit in
:mod:`blacknote.contracts.validation.LIMITS`; this module never re-declares a
number, it only reads the contract:

===============================  =========================================
design rule                       enforced through
===============================  =========================================
标题 ≤100 / 正文 ≤10000           ``LIMITS["title"]`` / ``LIMITS["body"]``
图片 ≤9 张                        ``LIMITS["images"]`` (via ``check_fields``)
单图 ≤5MB                         ``Settings.max_image_bytes`` (``INV-config-db``)
话题标签 ≥1 且 topic ∈ 固定话题     :func:`require_topic`
status ∈ {draft, published}      :func:`require_write_status`
===============================  =========================================

Why the numbers are the contract's and not the ticket's: ``NoteInput``
(validated inside the frozen dataclass, i.e. *before* any note service runs)
already rejects a title longer than ``LIMITS["title"]["max"]``.  Duplicating a
different ceiling here would either be dead code or a silent contradiction of
``INV-field-validation``, so the single source of truth stays in ``contracts``.
"""

from __future__ import annotations

import binascii
import re
from collections.abc import Sequence
from typing import Any, Final

from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ContractError, FieldError
from blacknote.contracts.validation import FIXED_TOPIC_TAGS, LIMITS, normalize_tag
from blacknote.contracts.visibility import ContentStatus

__all__ = [
    "MAX_IMAGES",
    "MIN_TOPIC_TAGS",
    "WRITABLE_STATUSES",
    "resolve_topic",
    "require_topic",
    "require_write_status",
    "check_image_urls",
    "check_tags",
]

#: Image count ceiling, mirrored from the contract (never hard-coded twice).
MAX_IMAGES: Final[int] = int(LIMITS["images"]["max"])
#: Every note must carry at least one discovery topic (design assumption A2).
MIN_TOPIC_TAGS: Final[int] = 1
#: Statuses an author may set through the write API; ``deleted`` is reserved
#: for :func:`~blacknote.note.service.NoteService.delete_note` (soft delete).
WRITABLE_STATUSES: Final[tuple[str, ...]] = (
    str(ContentStatus.DRAFT),
    str(ContentStatus.PUBLISHED),
)

_ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "data"})
_SCHEME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*):")
_DATA_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^data:(?P<mime>[^;,]*)?(?P<params>[^,]*);base64,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def check_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise + de-duplicate tags, rejecting malformed entries.

    Count and length bounds come from ``LIMITS`` through the shared
    ``check_fields`` call made by ``NoteInput``; this helper only guarantees the
    canonical spelling of every tag.
    """
    cleaned: list[str] = []
    for tag in tags or ():
        try:
            canonical = normalize_tag(tag)
        except ContractError as exc:
            raise ContractError.from_envelope(exc.to_envelope()) from None
        if canonical not in cleaned:
            cleaned.append(canonical)
    return tuple(cleaned)


def resolve_topic(tags: Sequence[str] | None) -> str | None:
    """Return the note's discovery topic (one of ``FIXED_TOPIC_TAGS``).

    ``None`` when the note only carries custom tags — which the platform
    forbids on write (see :func:`require_topic`) but tolerates on read for
    historical rows.
    """
    for tag in tags or ():
        if tag in FIXED_TOPIC_TAGS:
            return str(tag)
    return None


def require_topic(tags: Sequence[str] | None) -> str:
    """Guarantee ``话题标签≥1`` **and** ``topic ∈ {AI新闻, AI工具, AI产品}``.

    Raises ``VALIDATION_ERROR`` listing the offending field so the web layer can
    render it straight away.
    """
    if not tags:
        raise ContractError.validation(
            [FieldError("tags", "required",
                        "at least one topic tag is required "
                        f"({', '.join(FIXED_TOPIC_TAGS)})")]
        )
    topic = resolve_topic(tags)
    if topic is None:
        raise ContractError.validation(
            [FieldError("tags", "invalid_topic",
                        "one tag must be a platform topic: "
                        + ", ".join(FIXED_TOPIC_TAGS))]
        )
    return topic


def require_write_status(status: Any) -> ContentStatus:
    """Validate the lifecycle a caller may request through create/update."""
    if status is None:
        return ContentStatus.PUBLISHED
    value = getattr(status, "value", status)
    text = str(value).strip().lower()
    if text not in WRITABLE_STATUSES:
        raise ContractError.validation(
            [FieldError("status", "invalid_choice",
                        "status must be one of: " + ", ".join(WRITABLE_STATUSES))]
        )
    return ContentStatus(text)


def _decoded_size(b64_payload: str) -> int:
    """Exact byte length of a canonical base64 payload (no full decode)."""
    compact = re.sub(r"\s+", "", b64_payload)
    padding = len(compact) - len(compact.rstrip("="))
    if padding > 2:
        raise ValueError("invalid base64 padding")
    return (len(compact) // 4) * 3 - padding


def check_image_urls(image_urls: Sequence[str] | None, settings: Settings) -> tuple[str, ...]:
    """Validate the image list of a note.

    * count is capped by ``LIMITS["images"]`` (already asserted by
      ``NoteInput``; re-checked here so the rule holds for every entry point);
    * only ``http``, ``https`` and ``data`` URLs are accepted;
    * inline ``data:`` images must use a MIME type from
      ``Settings.allowed_image_types`` and stay below
      ``Settings.max_image_bytes`` (the 5MB single-image rule).
    """
    urls = tuple(str(url).strip() for url in (image_urls or ()) if str(url).strip())
    if len(urls) > MAX_IMAGES:
        raise ContractError.validation(
            [FieldError("image_urls", "too_many_items",
                        f"image_urls accepts at most {MAX_IMAGES} items")]
        )
    allowed_types = {str(kind).lower() for kind in settings.allowed_image_types}
    max_bytes = int(settings.max_image_bytes)
    for index, url in enumerate(urls):
        match = _SCHEME_PATTERN.match(url)
        scheme = (match.group("scheme").lower() if match else "")
        if scheme and scheme not in _ALLOWED_URL_SCHEMES:
            raise ContractError.validation(
                [FieldError(f"image_urls[{index}]", "invalid_url",
                            "image url must be an http(s) or data URL")]
            )
        if scheme != "data":
            continue
        data_match = _DATA_URL_PATTERN.match(url)
        if data_match is None:
            raise ContractError.validation(
                [FieldError(f"image_urls[{index}]", "invalid_data_url",
                            "inline images must be base64 data URLs")]
            )
        mime = (data_match.group("mime") or "").strip().lower()
        if allowed_types and mime not in allowed_types:
            raise ContractError.validation(
                [FieldError(f"image_urls[{index}]", "unsupported_image_type",
                            "image type must be one of: " + ", ".join(sorted(allowed_types)))]
            )
        try:
            size = _decoded_size(data_match.group("data"))
        except (ValueError, binascii.Error):
            raise ContractError.validation(
                [FieldError(f"image_urls[{index}]", "invalid_data_url",
                            "inline image payload is not valid base64")]
            ) from None
        if size > max_bytes:
            raise ContractError.validation(
                [FieldError(f"image_urls[{index}]", "image_too_large",
                            f"each image must stay below {max_bytes} bytes")]
            )
    return urls
