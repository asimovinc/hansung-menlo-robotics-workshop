from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

from menlo_runner.completion import (
    DEFAULT_MAX_DELIVERED_CUBES,
    DEFAULT_ROUND,
    ROUND_TIME_LIMITS_S,
    CompletionConfig,
    completion_config_for_round,
)


CUBE_COLOR_ORDER_KEYS = (
    1,
    7,
    10,
    16,
    19,
    24,
    31,
    32,
    34,
    40,
    44,
    46,
    48,
    49,
    52,
    62,
    65,
    68,
    77,
    81,
    84,
    92,
    99,
)

START_X_RANGE = (-4.0, 2.1)
START_Y_RANGE = (-6.5, 6.4)
DEFAULT_OBSTACLE_CLEARANCE_M = 0.45
MAX_START_SAMPLES = 10_000
ROUND_OPTION_COUNT = 50


@dataclass(frozen=True)
class EvaluationSetup:
    level: int
    setup_seed: str
    cube_color_order_key: int
    start_x: float
    start_y: float


def _rng_for_round_option(round_name: str, setup_option: int) -> random.Random:
    normalized = normalize_round_name(round_name)
    return random.Random(f"hansung-menlo-eval:{normalized}:option-{setup_option}")


def normalize_round_name(round_name: str) -> str:
    normalized = round_name.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"1", "round1"}:
        return "round1"
    if normalized in {"2", "round2"}:
        return "round2"
    if normalized in {"3", "round3"}:
        return "round3"
    if normalized == "manual":
        return "manual"
    raise ValueError("round must be round1, round2, round3, or manual.")


def validate_setup_option(setup_option: int) -> int:
    if not 1 <= setup_option <= ROUND_OPTION_COUNT:
        raise ValueError(f"setup option must be between 1 and {ROUND_OPTION_COUNT}.")
    return setup_option


def choose_round_evaluation_setup(
    level: int,
    round_name: str,
    setup_option: int,
) -> EvaluationSetup:
    """Choose one of 50 shared setup options for a scored round."""
    if level not in {0, 1, 2}:
        raise ValueError("level must be one of 0, 1, or 2")
    setup_option = validate_setup_option(setup_option)
    normalized_round = normalize_round_name(round_name)
    rng = _rng_for_round_option(normalized_round, setup_option)
    return EvaluationSetup(
        level=level,
        setup_seed=f"{normalized_round}-{setup_option:02d}",
        cube_color_order_key=rng.choice(CUBE_COLOR_ORDER_KEYS),
        start_x=rng.uniform(*START_X_RANGE),
        start_y=rng.uniform(*START_Y_RANGE),
    )


def _box_bounds_xy(obstacle: dict[str, Any], clearance_m: float) -> tuple[float, float, float, float] | None:
    if obstacle.get("kind") != "box":
        return None
    try:
        x, y = obstacle["pose"]["position"][:2]
        sx, sy = obstacle["size"][:2]
    except (KeyError, TypeError, ValueError):
        return None
    half_x = float(sx) / 2.0 + clearance_m
    half_y = float(sy) / 2.0 + clearance_m
    return float(x) - half_x, float(x) + half_x, float(y) - half_y, float(y) + half_y


def point_is_clear_of_obstacles(
    x: float,
    y: float,
    obstacles: list[dict[str, Any]],
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> bool:
    for obstacle in obstacles:
        bounds = _box_bounds_xy(obstacle, clearance_m)
        if bounds is None:
            continue
        min_x, max_x, min_y, max_y = bounds
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return False
    return True


def choose_clear_round_start_xy(
    round_name: str,
    setup_option: int,
    obstacles: list[dict[str, Any]],
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> tuple[float, float]:
    rng = _rng_for_round_option(round_name, validate_setup_option(setup_option))
    rng.choice(CUBE_COLOR_ORDER_KEYS)

    for _attempt in range(MAX_START_SAMPLES):
        x = rng.uniform(*START_X_RANGE)
        y = rng.uniform(*START_Y_RANGE)
        if point_is_clear_of_obstacles(x, y, obstacles, clearance_m=clearance_m):
            return x, y

    raise RuntimeError(
        "Could not sample a round start position clear of scene_layout obstacles. "
        f"Try lowering clearance_m below {clearance_m}."
    )


async def current_scene_id(ctx: Any) -> str | None:
    """Return the runtime scene id when the viewer exposes scene_layout."""
    try:
        layout = await ctx.state("scene_layout")
    except Exception:
        return None
    if isinstance(layout, dict):
        scene_id = layout.get("scene_id")
        return str(scene_id) if scene_id is not None else None
    scene_id = getattr(layout, "scene_id", None)
    return str(scene_id) if scene_id is not None else None


async def get_scene_layout(ctx: Any) -> dict[str, Any] | None:
    try:
        layout = await ctx.state("scene_layout")
    except Exception:
        return None
    return layout if isinstance(layout, dict) else None


async def apply_clear_round_start_from_layout(
    ctx: Any,
    setup: EvaluationSetup,
    round_name: str,
    setup_option: int,
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> EvaluationSetup:
    layout = await get_scene_layout(ctx)
    if layout is None:
        print("scene_layout unavailable; using unfiltered sampled start position.")
        return setup

    obstacles = layout.get("obstacles", [])
    if not isinstance(obstacles, list):
        print("scene_layout obstacles unavailable; using unfiltered sampled start position.")
        return setup

    x, y = choose_clear_round_start_xy(
        round_name,
        setup_option,
        obstacles,
        clearance_m=clearance_m,
    )
    return EvaluationSetup(
        level=setup.level,
        setup_seed=setup.setup_seed,
        cube_color_order_key=setup.cube_color_order_key,
        start_x=x,
        start_y=y,
    )


async def reload_current_scene(ctx: Any) -> None:
    """Reload the current scene if the runtime select_scene skill is available."""
    scene_id = await current_scene_id(ctx)
    if not scene_id:
        print("Scene id unavailable; skip select_scene reload.")
        return
    result = await ctx.invoke("select_scene", {"scene_id": scene_id}, timeout_s=30)
    status = getattr(result, "status", result)
    print(f"select_scene {scene_id!r} -> {status}")


async def go_to_start_position(ctx: Any, setup: EvaluationSetup) -> Any:
    target = {
        "kind": "pose",
        "pose": {
            "frame_id": "world",
            "position": [setup.start_x, setup.start_y, 0.0],
        },
    }
    result = await ctx.invoke("go_to", {"target": target}, timeout_s=300)
    status = getattr(result, "status", result)
    print(f"go_to start ({setup.start_x:+.2f}, {setup.start_y:+.2f}) -> {status}")
    return result


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{text}{suffix}: ").strip()
    except EOFError:
        value = ""
    return value or default


async def prepare_evaluation_round(
    ctx: Any,
    level: int,
    *,
    clearance_m: float = DEFAULT_OBSTACLE_CLEARANCE_M,
) -> CompletionConfig:
    """Prompt for round timing/setup and optionally place the robot at the shared start.

    Teams can skip shared setup by leaving the option blank. For live evaluation,
    enter the announced round and setup option number from 1 to 50.
    """
    env_round = os.environ.get("MENLO_EVAL_ROUND")
    round_name = normalize_round_name(env_round or _prompt("Round (round1/round2/round3/manual)", DEFAULT_ROUND))
    manual_seconds = None
    if round_name == "manual":
        env_seconds = os.environ.get("MENLO_EVAL_SECONDS")
        seconds_text = env_seconds or _prompt("Manual round time in seconds", str(ROUND_TIME_LIMITS_S[DEFAULT_ROUND]))
        manual_seconds = float(seconds_text)

    env_option = os.environ.get("MENLO_EVAL_OPTION")
    option_text = env_option if env_option is not None else _prompt(
        f"Evaluation setup option 1-{ROUND_OPTION_COUNT} (blank to skip shared setup)",
        "",
    )
    if option_text:
        setup_option = validate_setup_option(int(option_text))
        setup = choose_round_evaluation_setup(level, round_name, setup_option)
        setup = await apply_clear_round_start_from_layout(
            ctx,
            setup,
            round_name,
            setup_option,
            clearance_m=clearance_m,
        )

        print("=" * 60)
        print("Evaluation setup")
        print(f"round: {round_name}")
        print(f"setup_option: {setup_option}")
        print(f"level: {setup.level}")
        print(f"cube_color_order_key: {setup.cube_color_order_key}")
        print(f"start_xy: ({setup.start_x:+.3f}, {setup.start_y:+.3f})")
        print(f"obstacle_clearance_m: {clearance_m:.2f}")
        print("=" * 60)

        await reload_current_scene(ctx)
        _prompt(
            "In the viewer seed box, enter the cube_color_order_key above, "
            "apply/reset the scene, then press Enter here..."
        )
        await go_to_start_position(ctx, setup)
    else:
        print("Shared evaluation setup skipped; using the current scene and robot pose.")

    return completion_config_for_round(
        level,
        round_name=round_name,
        manual_seconds=manual_seconds,
        max_delivered_cubes=DEFAULT_MAX_DELIVERED_CUBES,
    )
