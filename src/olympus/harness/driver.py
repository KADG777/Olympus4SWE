"""Execution contract for built-in and external agent harnesses."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

from pydantic import Field, StringConstraints

from olympus.core.contracts import StrictModel

DriverEvent = Callable[[str, dict[str, Any]], None]
NonEmptySetting = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DriverTurnRequest(StrictModel):
    workspace_path: Path
    env_root: Path
    system_prompt: str
    user_prompt: str
    actor: str
    task_id: str | None = None
    lease_generation: int | None = None
    read_only: bool = False
    write_scope: list[str] = Field(default_factory=list)
    forbidden_globs: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20, ge=1)
    trace_path: Path | None = None
    model: str | None = None


@dataclass(frozen=True)
class TurnOutcome:
    completed: bool
    stop_reason: str
    final_content: str
    steps_taken: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)
    driver_name: str = ""


class ActiveTurnHandle(Protocol):
    @property
    def supports_steer(self) -> bool: ...

    def steer(self, guidance: str, *, source: str) -> bool: ...

    def cancel(self, *, reason: str) -> None: ...


class AgentDriver(Protocol):
    """Block until the turn ends, publishing its handle synchronously before execution."""

    name: str

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome: ...


class ExecutionDriversConfig(StrictModel):
    worker_driver: NonEmptySetting = "native"
    reviewer_driver: NonEmptySetting = "native"
    monitor_model: NonEmptySetting | None = None

    @classmethod
    def resolve(
        cls,
        *,
        worker_driver: str | None = None,
        reviewer_driver: str | None = None,
        monitor_model: str | None = None,
    ) -> ExecutionDriversConfig:
        explicit = {
            "worker_driver": worker_driver,
            "reviewer_driver": reviewer_driver,
            "monitor_model": monitor_model,
        }
        values = {}
        for field_name, value in explicit.items():
            if value is not None:
                values[field_name] = value
            elif setting := os.getenv(f"OLYMPUS_{field_name.upper()}", "").strip():
                values[field_name] = setting
        return cls.model_validate(values)


class DriverRegistry:
    """Resolve driver specs against the native factory and injected drivers."""

    def __init__(self, drivers: Mapping[str, AgentDriver] | None = None) -> None:
        self._drivers = dict(drivers or {})

    def resolve(
        self, spec: str, *, native_factory: Callable[[str | None], AgentDriver]
    ) -> tuple[AgentDriver, str | None]:
        name, separator, raw_model = spec.partition("@")
        model = raw_model.strip() or None
        if separator and model is None:
            raise ValueError(f"Missing model override in driver spec {spec!r}")
        if name == "native":
            driver = native_factory(model)
        else:
            try:
                driver = self._drivers[name]
            except KeyError:
                raise ValueError(
                    f"Unknown agent driver {name!r}; registered drivers: "
                    f"{', '.join(['native', *sorted(self._drivers)])}"
                ) from None
        return driver, model
