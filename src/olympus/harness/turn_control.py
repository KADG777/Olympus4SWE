"""Synchronized control for one native turn."""

from threading import Lock


class NativeTurnControl:
    def __init__(self) -> None:
        self._lock = Lock()
        self._guidance: list[tuple[str, str]] = []
        self._cancel_reason: str | None = None
        self._closed = False

    @property
    def supports_steer(self) -> bool:
        return True

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancel_reason is not None

    def steer(self, guidance: str, *, source: str) -> bool:
        with self._lock:
            if self._closed or self._cancel_reason is not None:
                return False
            self._guidance.append((guidance, source))
            return True

    def cancel(self, *, reason: str) -> None:
        with self._lock:
            if not self._closed and self._cancel_reason is None:
                self._cancel_reason = reason

    def take_guidance(self) -> list[tuple[str, str]]:
        with self._lock:
            if self._cancel_reason is not None:
                return []
            pending, self._guidance = self._guidance, []
            return pending

    def close(self) -> bool:
        with self._lock:
            self._closed = True
            self._guidance.clear()
            return self._cancel_reason is not None
