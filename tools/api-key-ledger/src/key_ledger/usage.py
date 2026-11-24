"""Protocol usage extraction, incremental SSE parsing and fixed-point pricing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .config import Dialect, Price

PICOS_PER_UNIT = 10**12
MAX_SSE_EVENT = 1024 * 1024


def money(units: int) -> str:
    whole, fraction = divmod(units, PICOS_PER_UNIT)
    return f"{whole}.{fraction:012d}".rstrip("0").rstrip(".")


def cap_units(amount: Decimal) -> int:
    return int(amount * PICOS_PER_UNIT)


def _tokens(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class Usage:
    model: str | None
    uncached: int | None = None
    output: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    @property
    def input(self) -> int | None:
        if self.uncached is None:
            return None
        return self.uncached + (self.cache_read or 0) + (self.cache_write or 0)

    def update(self, dialect: Dialect, payload: dict[str, Any]) -> bool:
        model = payload.get("model")
        if isinstance(model, str):
            self.model = model
        data = _object(payload.get("usage"))
        if dialect == "openai":
            prompt = _tokens(data.get("prompt_tokens"))
            output = _tokens(data.get("completion_tokens"))
            if prompt is not None:
                cached = _tokens(_object(data.get("prompt_tokens_details")).get("cached_tokens"))
                if cached is None:
                    cached = _tokens(data.get("prompt_cache_hit_tokens")) or 0
                # Invalid subsets must not produce a negative input charge.
                if cached <= prompt:
                    self.uncached = prompt - cached
                    self.cache_read = cached
                    self.cache_write = 0
            if output is not None:
                self.output = output
            return self.input is not None and self.output is not None

        input_count = _tokens(data.get("input_tokens"))
        output_count = _tokens(data.get("output_tokens"))
        cached = _tokens(data.get("cache_read_input_tokens"))
        written = _tokens(data.get("cache_creation_input_tokens"))
        if input_count is not None:
            self.uncached = input_count
            if self.cache_read is None:
                self.cache_read = 0
            if self.cache_write is None:
                self.cache_write = 0
        if output_count is not None:
            self.output = output_count
        if cached is not None:
            self.cache_read = cached
        if written is not None:
            self.cache_write = written
        return self.input is not None and self.output is not None

    def charge(self, price: Price | None) -> tuple[str, int | None]:
        if price is None:
            return "unpriced", None
        counts = (self.uncached, self.output, self.cache_read, self.cache_write)
        if all(count is None for count in counts):
            return "usage_unavailable", None
        rates = (
            price.input_per_1m,
            price.output_per_1m,
            price.cached_input_per_1m,
            price.cache_write_per_1m,
        )
        units = sum(
            (count or 0) * int(rate * 10**6) for count, rate in zip(counts, rates, strict=True)
        )
        return (
            "priced" if self.input is not None and self.output is not None else "partial"
        ), units


@dataclass
class SSEUsage:
    dialect: Dialect
    usage: Usage
    terminal: bool = False
    errored: bool = False
    _buffer: bytearray = field(default_factory=bytearray)
    _event: bytearray = field(default_factory=bytearray)
    _discard_line: bool = False
    _discard_event: bool = False
    _skip_lf: bool = False

    def feed(self, chunk: bytes) -> None:
        # SSE permits LF, CRLF and CR. A CRLF can span network chunks.
        if not chunk:
            return
        chunk = chunk.removeprefix(b"\n") if self._skip_lf else chunk
        self._skip_lf = chunk.endswith(b"\r")
        if not chunk:
            return
        data = chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        lines = data.split(b"\n")
        for line in lines[:-1]:
            self._line_part(line)
            if self._discard_line:
                self._discard_line = False
            else:
                self._line(bytes(self._buffer))
            self._buffer.clear()
        self._line_part(lines[-1])

    def finish(self) -> None:
        if self._buffer and not self._discard_line:
            self._line(bytes(self._buffer))
        self._dispatch()

    def _line_part(self, part: bytes) -> None:
        if self._discard_line:
            return
        if len(self._buffer) + len(part) > MAX_SSE_EVENT:
            self._discard_line = self._discard_event = True
            self._buffer.clear()
            self._event.clear()
            return
        self._buffer.extend(part)

    def _line(self, line: bytes) -> None:
        if not line:
            self._dispatch()
        elif not self._discard_event and line.startswith(b"data:"):
            data = line[5:].removeprefix(b" ")
            if len(data) + len(self._event) >= MAX_SSE_EVENT:
                self._discard_event = True
                self._event.clear()
            else:
                if self._event:
                    self._event.extend(b"\n")
                self._event.extend(data)
        elif line.strip() == b"event: error":
            self.errored = True

    def _dispatch(self) -> None:
        try:
            if self._discard_event or not self._event:
                return
            if self._event.strip() == b"[DONE]":
                self.terminal = True
                return
            try:
                event = _object(json.loads(self._event))
            except (ValueError, RecursionError):
                return
            kind = event.get("type")
            if kind == "error" or event.get("error"):
                self.errored = True
            elif self.dialect == "openai":
                if self.usage.update(self.dialect, event) and event.get("usage"):
                    self.terminal = True
            elif kind == "message_start":
                self.usage.update(self.dialect, _object(event.get("message")))
            elif kind == "message_delta":
                self.usage.update(self.dialect, event)
            elif kind == "message_stop":
                self.terminal = True
        finally:
            self._event.clear()
            self._discard_event = False
