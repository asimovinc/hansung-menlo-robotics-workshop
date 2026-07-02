from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


LEVEL_DELIVERY_POINTS = {
    0: 5,
    1: 20,
    2: 40,
}

ROUND_TIME_LIMITS_S = {
    "round1": 5 * 60,
    "round2": 10 * 60,
    "round3": 15 * 60,
}

DEFAULT_ROUND = "round2"
DEFAULT_MAX_DELIVERED_CUBES = 12
LEVEL_FIRST_DELIVERY_BONUS = {
    1: 60,
    2: 60,
}


class CompletionTimeout(TimeoutError):
    """Raised when a scored completion run exhausts its strict time limit."""


@dataclass(frozen=True)
class CompletionConfig:
    """Stop conditions and scoring settings for a project completion run."""

    level: int | None = None
    points_per_delivery: int | None = None
    max_delivered_cubes: int | None = None
    max_elapsed_s: float | None = None

    def validate(self) -> None:
        if self.level is not None and self.level not in LEVEL_DELIVERY_POINTS:
            raise ValueError("level must be 0, 1, 2, or None.")
        if self.points_per_delivery is not None and self.points_per_delivery < 0:
            raise ValueError("points_per_delivery must be zero or greater.")
        if self.max_delivered_cubes is not None and self.max_delivered_cubes < 0:
            raise ValueError("max_delivered_cubes must be zero or greater.")
        if self.max_elapsed_s is not None and self.max_elapsed_s <= 0:
            raise ValueError("max_elapsed_s must be greater than zero.")
        if self.max_delivered_cubes is None and self.max_elapsed_s is None:
            raise ValueError("Set max_delivered_cubes, max_elapsed_s, or both.")

    def delivery_points(self) -> int:
        if self.points_per_delivery is not None:
            return self.points_per_delivery
        if self.level is None:
            return 0
        return LEVEL_DELIVERY_POINTS[self.level]


def round_time_limit_s(round_name: str, *, manual_seconds: float | None = None) -> float:
    """Resolve a named project round or manual time limit into seconds."""
    normalized = round_name.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"1", "round1"}:
        return float(ROUND_TIME_LIMITS_S["round1"])
    if normalized in {"2", "round2"}:
        return float(ROUND_TIME_LIMITS_S["round2"])
    if normalized in {"3", "round3"}:
        return float(ROUND_TIME_LIMITS_S["round3"])
    if normalized == "manual":
        if manual_seconds is None or manual_seconds <= 0:
            raise ValueError("manual_seconds must be greater than zero for manual round timing.")
        return float(manual_seconds)
    raise ValueError("round_name must be round1, round2, round3, or manual.")


def completion_config_for_round(
    level: int,
    *,
    round_name: str = DEFAULT_ROUND,
    manual_seconds: float | None = None,
    max_delivered_cubes: int = DEFAULT_MAX_DELIVERED_CUBES,
) -> CompletionConfig:
    """Build the standard project completion config for a scored round."""
    return CompletionConfig(
        level=level,
        max_delivered_cubes=max_delivered_cubes,
        max_elapsed_s=round_time_limit_s(round_name, manual_seconds=manual_seconds),
    )


class CompletionTracker:
    """Measure a run from the first agent cycle and report completion reasons."""

    def __init__(
        self,
        config: CompletionConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate()
        self.config = config
        self._clock = clock
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.end_reason: str | None = None

    def start_first_cycle(self) -> None:
        if self.started_at is None:
            self.started_at = self._clock()

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        stop_at = self.ended_at if self.ended_at is not None else self._clock()
        return max(0.0, stop_at - self.started_at)

    def remaining_s(self) -> float | None:
        if self.config.max_elapsed_s is None:
            return None
        return max(0.0, self.config.max_elapsed_s - self.elapsed_s())

    def stop_reason(self, delivered_count: int) -> str | None:
        if (
            self.config.max_delivered_cubes is not None
            and delivered_count >= self.config.max_delivered_cubes
        ):
            return f"delivered {delivered_count}/{self.config.max_delivered_cubes} cubes"
        if self.config.max_elapsed_s is not None and self.elapsed_s() >= self.config.max_elapsed_s:
            return f"elapsed {self.elapsed_s():.1f}/{self.config.max_elapsed_s:.1f} seconds"
        return None

    async def robot_fall_reason(self, ctx: Any) -> str | None:
        """Return a stop reason if the public robot status reports a fall."""
        try:
            status = await ctx.state("robot_status")
        except Exception:
            return None

        robot = getattr(status, "robot", None)
        robot_status = getattr(robot, "status", None)
        value = getattr(robot_status, "value", robot_status)
        if str(value).lower() != "fallen":
            return None

        remaining = self.remaining_s()
        if remaining is None:
            return "robot fallen"
        return f"robot fallen; remaining {remaining:.1f} seconds"

    async def scene_delivered_count(self, ctx: Any) -> int:
        """Count delivered cubes using the shared destination-pad helper."""
        from menlo_runner.scene import delivered_cube_ids

        return len(await delivered_cube_ids(ctx))

    async def stop_reason_from_scene(self, ctx: Any) -> str | None:
        """Check stop conditions using authoritative scene progress."""
        delivered_reason = self.stop_reason(await self.scene_delivered_count(ctx))
        if delivered_reason is not None:
            return delivered_reason
        return await self.robot_fall_reason(ctx)

    async def wait_for_remaining(self, awaitable: Any, label: str) -> Any:
        """Await a step, cutting it off exactly when the run time expires."""
        remaining = self.remaining_s()
        if remaining is None:
            return await awaitable
        if remaining <= 0:
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise CompletionTimeout(
                f"elapsed {self.elapsed_s():.1f}/{self.config.max_elapsed_s:.1f} seconds "
                f"before {label}"
            )

        import asyncio

        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise CompletionTimeout(
                f"elapsed {self.config.max_elapsed_s:.1f}/{self.config.max_elapsed_s:.1f} "
                f"seconds while waiting for {label}"
            ) from exc

    def delivery_score(self, delivered_count: int) -> int:
        if delivered_count <= 0:
            return 0
        if self.config.level in LEVEL_FIRST_DELIVERY_BONUS:
            return LEVEL_FIRST_DELIVERY_BONUS[self.config.level] + (
                (delivered_count - 1) * self.config.delivery_points()
            )
        return delivered_count * self.config.delivery_points()

    def score_description(self) -> str:
        if self.config.level in LEVEL_FIRST_DELIVERY_BONUS:
            return (
                f"{LEVEL_FIRST_DELIVERY_BONUS[self.config.level]} points for the first delivery, "
                f"then {self.config.delivery_points()} points per additional delivery"
            )
        delivery_points = self.config.delivery_points()
        return (
            f"{delivery_points} points per delivery"
            if delivery_points
            else "delivery scoring not configured"
        )

    def mark_ended(self, reason: str) -> None:
        if self.ended_at is None:
            self.ended_at = self._clock()
            self.end_reason = reason

    def print_start(self) -> None:
        target_cubes = (
            self.config.max_delivered_cubes
            if self.config.max_delivered_cubes is not None
            else "any"
        )
        time_limit = self.config.max_elapsed_s if self.config.max_elapsed_s is not None else "none"
        print(
            "Completion timer started at first cycle "
            f"(target cubes={target_cubes}, "
            f"time limit={time_limit}s, "
            f"{self.score_description()})."
        )

    def print_summary(self, delivered_count: int) -> None:
        reason = self.end_reason or self.stop_reason(delivered_count) or "agent stopped"
        remaining = self.remaining_s()
        remaining_text = "" if remaining is None else f"; remaining={remaining:.1f}s"
        print(
            "Completion run ended: "
            f"{reason}; elapsed={self.elapsed_s():.1f}s; delivered={delivered_count}; "
            f"delivery_score={self.delivery_score(delivered_count)}{remaining_text}."
        )

    async def print_summary_from_scene(self, ctx: Any) -> None:
        """Print the completion summary using authoritative scene progress."""
        self.print_summary(await self.scene_delivered_count(ctx))


def level_from_program_name(program_name: str) -> int | None:
    if "level-0" in program_name or "level_0" in program_name:
        return 0
    if "level-1" in program_name or "level_1" in program_name:
        return 1
    if "level-2" in program_name or "level_2" in program_name:
        return 2
    return None
