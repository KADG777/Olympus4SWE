"""Execution contract for built-in and external agent harnesses."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import Field, StringConstraints

from olympus.core.contracts import StrictModel

if TYPE_CHECKING:
    from olympus.core.store import RunStore

DriverEvent = Callable[[str, dict[str, Any]], None]
DriverFactory = Callable[[], "AgentDriver"]
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
    max_steps: int | None = Field(default=None, ge=1)
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
    """Publish a handle before execution and emit callbacks during the turn.

    Tool milestones use agent.tool_started / agent.tool_finished, with step,
    tool and (for completion) status, duration_ms and error_signature.
    agent.steered carries the source of guidance. text_delta carries text,
    with replace=True when a driver supplies a complete message.
    Drivers own local JSONL traces. Finished tools use event="tool_end" with
    the milestone's tool fields plus arguments and output. Text records use
    event="text_delta" with the corresponding text payload.
    """

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
    """Resolve driver specs against explicit drivers, lazy factories, and the native factory."""

    def __init__(
        self,
        drivers: Mapping[str, AgentDriver] | None = None,
        *,
        factories: Mapping[str, DriverFactory] | None = None,
    ) -> None:
        self._drivers = dict(drivers or {})
        self._factories = dict(factories or {})
        self._built: dict[str, AgentDriver] = {}

    def with_factories(self, factories: Mapping[str, DriverFactory]) -> DriverRegistry:
        """Return a registry that keeps explicit drivers and adds lazy default drivers."""
        merged = dict(self._factories)
        merged.update(factories)
        return DriverRegistry(self._drivers, factories=merged)

    def resolve(
        self, spec: str, *, native_factory: Callable[[str | None], AgentDriver]
    ) -> tuple[AgentDriver, str | None]:
        name, separator, raw_model = spec.partition("@")
        model = raw_model.strip() or None
        if separator and model is None:
            raise ValueError(f"Missing model override in driver spec {spec!r}")
        if name == "native":
            driver = native_factory(model)
        elif name in self._drivers:
            driver = self._drivers[name]
        elif name in self._factories:
            built = self._built.get(name)
            if built is None:
                built = self._factories[name]()
                self._built[name] = built
            driver = built
        else:
            registered = sorted({"native", *self._drivers, *self._factories})
            raise ValueError(
                f"Unknown agent driver {name!r}; registered drivers: "
                f"{', '.join(registered)}"
            )
        return driver, model


def default_driver_factories(
    *,
    store: RunStore | None,
    run_id: str | None,
) -> dict[str, DriverFactory]:
    """Lazily build the external drivers shipped with Olympus (never at import time)."""
    from olympus.harness.pi import PiDriver

    return {"pi": lambda: PiDriver(store=store, run_id=run_id)}
