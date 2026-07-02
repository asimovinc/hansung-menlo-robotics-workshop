from __future__ import annotations

"""Level 0 project starter for the Menlo AI robot sorting challenge.

This file is intentionally a starter, not a solution.

Sections marked SUPPORT CODE provide small wrappers and data structures so you do
not need to rewrite workshop setup code. Sections marked STUDENT TODO are where
your project design belongs.

Run setup:
- By default, run(ctx) asks for round1, round2, round3, or manual timing.
  Round limits are 5, 10, and 15 minutes, and every round stops after at most
  12 delivered cubes.
- For normal practice, press Enter to use round2 and leave the evaluation setup
  option blank. This keeps the current scene and robot pose.
- For shared evaluation practice, enter the announced round and an option number
  from 1 to 50. The starter prints the cube_color_order_key and moves the robot
  to the matching deterministic start position after you apply/reset the viewer.
- Manual timing is available by typing manual, then entering the time in seconds.

Level 0 rule: scene_state, exact entity IDs, and entity-target go_to are allowed.
The main challenge is still to build a meaningful LLM-assisted high-level
decision loop rather than a fixed script.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTimeout, CompletionTracker
from menlo_runner.perception import compress_jpeg
from menlo_runner.programs.evaluation_setup import prepare_evaluation_round
from menlo_runner.scene import COLOR_TO_PAD, delivered_cube_ids, held_cube_info, visible_cubes


# ---------------------------------------------------------------------------
# SUPPORT CODE: shared task definition and required LLM decision schema
# ---------------------------------------------------------------------------
TASK = "Find and sort cubes from the source area into their matching destination pads."

# Model selection for notebook and Python starter users.
# Edit this value or set MENLO_LLM_MODEL in your environment/.env before running.
APPROVED_LLM_MODELS = (
    "minimaxai/minimax-m3",
    "qwen/qwen3.6-35b-a3b",
)
LLM_MODEL = os.environ.setdefault("MENLO_LLM_MODEL", "minimaxai/minimax-m3")
VLM_MODEL = os.environ.setdefault("MENLO_VLM_MODEL", "qwen/qwen3.6-35b-a3b")

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}

ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}


@dataclass
class AgentDecision:
    """Validated high-level decision returned by the LLM."""

    next_action: str
    target_color: str | None = None
    target_entity_id: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """State your agent carries across observe-decide-act cycles."""

    delivered_count: int = 0
    held_color: str | None = None
    held_entity_id: str | None = None
    active_cube_id: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_cube_ids: list[str] = field(default_factory=list)
    skipped_cube_ids: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """Compact full-state observation passed to the LLM and action code."""

    robot_status: Any
    visible_cubes: list[dict[str, Any]]
    held_cube: dict[str, str] | None
    delivered_cube_ids: list[str]
    color_to_pad: dict[str, str]
    note: str = ""


def parse_agent_decision(text: str) -> AgentDecision | None:
    """Parse and validate the required structured LLM JSON output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    target_color = data.get("target_color")
    if target_color is not None and not isinstance(target_color, str):
        return None

    target_entity_id = data.get("target_entity_id")
    if target_entity_id is not None and not isinstance(target_entity_id, str):
        return None

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        target_entity_id=target_entity_id,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert full-state information into a compact text context for the LLM."""
    return {
        "task": task,
        "visible_cubes": observation.visible_cubes,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
        "color_to_pad": observation.color_to_pad,
        "memory": {
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
            "held_entity_id": memory.held_entity_id,
            "active_cube_id": memory.active_cube_id,
            "active_color": memory.active_color,
            "stage": memory.stage,
            "failed_attempts": memory.failed_attempts,
            "completed_cube_ids": memory.completed_cube_ids,
            "skipped_cube_ids": memory.skipped_cube_ids,
        },
        "last_result": last_result,
        "note": observation.note,
    }


# ---------------------------------------------------------------------------
# SUPPORT CODE: Level 0 SDK wrappers
# ---------------------------------------------------------------------------

async def get_robot_status(ctx: Any) -> Any:
    """Read robot pose, motion status, and neck state."""
    return await ctx.state("robot_status")


async def get_camera_frame(
    ctx: Any,
    *,
    compressed: bool = False,
    max_width: int = 800,
    quality: int = 70,
) -> bytes:
    """Capture the POV camera frame, optionally resized/re-encoded for VLM use."""
    jpeg = await ctx.get_vision("pov")
    if compressed:
        return compress_jpeg(jpeg, max_width=max_width, quality=quality)
    return jpeg


async def observe_full_state(ctx: Any) -> Observation:
    """Collect a project Level 0 observation from scene_state helpers."""
    robot_status = await get_robot_status(ctx)
    cubes = [
        {
            "entity_id": cube.entity_id,
            "color": cube.color,
            "position": cube.position,
            "distance_from_robot": round(cube.distance_from_robot, 2),
        }
        for cube in await visible_cubes(ctx)
    ]
    held = await held_cube_info(ctx)
    held_dict = {"entity_id": held[0], "color": held[1]} if held else None
    delivered = await delivered_cube_ids(ctx)
    return Observation(
        robot_status=robot_status,
        visible_cubes=cubes,
        held_cube=held_dict,
        delivered_cube_ids=delivered,
        color_to_pad=dict(COLOR_TO_PAD),
    )


async def go_to_entity(ctx: Any, entity_id: str) -> Any:
    """Level 0 entity-target navigation."""
    return await ctx.invoke(
        "go_to",
        {"target": {"kind": "entity", "entity_id": entity_id}},
        timeout_s=300,
    )


async def pick_cube_by_id(ctx: Any, cube_id: str) -> Any:
    """Pick a specific cube entity after navigating close enough."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": cube_id}},
        timeout_s=300,
    )


async def place_on_pad_by_id(ctx: Any, pad_id: str) -> Any:
    """Place the held cube on a specific pad entity."""
    return await ctx.invoke(
        "place_entity",
        {"target": {"kind": "entity", "entity_id": pad_id}},
        timeout_s=300,
    )


def result_summary(result: Any) -> dict[str, Any]:
    """Convert SDK results into a small loggable dictionary."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


# ---------------------------------------------------------------------------
# STUDENT TODO: LLM decision function
# ---------------------------------------------------------------------------

async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Use a text LLM to choose the next high-level action.

    TODO:
    - Build a prompt from build_decision_context(...).
    - Call menlo_runner.llm.call_llm or your approved LLM helper.
      If the helper is synchronous/blocking, use await asyncio.to_thread(...)
      so the strict round timer can stop waiting when time expires.
    - Require JSON with next_action, target_color, target_entity_id, and reason.
    - Validate with parse_agent_decision before execution.

    The placeholder below stops safely. Replace it with a real LLM call for
    submission; do not hard-code a fixed action sequence.
    """
    _decision_context = build_decision_context(task, observation, memory, last_result)
    return AgentDecision(
        next_action="stop",
        reason="TODO: call the text LLM, validate its JSON decision, then execute it.",
    )


# ---------------------------------------------------------------------------
# STUDENT TODO: observation, execution, verification, and memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """Collect the current Level 0 observation for the LLM and action code.

    TODO:
    - Add natural-language task parsing results if useful.
    - Add compact scene summaries if useful.
    - Keep the observation small enough for repeated LLM calls.
    """
    return await observe_full_state(ctx)


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """Translate one validated LLM decision into a Level 0 robot action.

    TODO:
    - Validate that the selected target is consistent with the observation.
    - Use go_to_entity, pick_cube_by_id, and place_on_pad_by_id where appropriate.
    - Decide how to handle search/recover/skip/stop for your policy.
    - Avoid fixed scripts; let the LLM and memory choose the high-level sequence.
    """
    if decision.next_action == "stop":
        return {"action": "stop", "status": "stopped"}

    return {
        "action": decision.next_action,
        "status": "todo",
        "reason": "Implement Level 0 action execution for this validated decision.",
    }


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """Check whether the last action appears to have worked.

    TODO:
    - Re-observe scene_state after important actions.
    - Check whether the robot is holding a cube after pick.
    - Check delivered_cube_ids after place.
    - Return facts the next LLM call can use for recovery.
    """
    observation = await observe_full_state(ctx)
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "held_cube": observation.held_cube,
        "delivered_cube_ids": observation.delivered_cube_ids,
    }


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """Update persistent state after each cycle.

    TODO:
    - Track active cube, held cube, delivered count, failures, and skip history.
    - Add concise logs that you can show during presentations.
    """
    memory.logs.append(
        {
            "observation": {
                "visible_cube_count": len(observation.visible_cubes),
                "held_cube": observation.held_cube,
                "delivered_count": memory.delivered_count,
            },
            "llm_decision": decision.__dict__,
            "verified": verified,
        }
    )


async def run_agent(
    ctx: Any,
    *,
    task: str = TASK,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """Thin observe-LLM-act loop. Edit the TODO functions, not just this loop."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    async def run_step(awaitable: Any, label: str) -> Any:
        if tracker is None:
            return await awaitable
        return await tracker.wait_for_remaining(awaitable, label)

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 0] Cycle {cycle}")
        try:
            if tracker is not None:
                first_cycle = tracker.started_at is None
                tracker.start_first_cycle()
                if first_cycle:
                    tracker.print_start()
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached before cycle action: {reason}.")
                    break

            observation = await run_step(observe_world(ctx, memory), "observe_world")
            decision = await run_step(
                decide_next_action(task, observation, memory, last_result),
                "LLM decision",
            )
            print("LLM decision:", decision)

            if decision.next_action == "stop":
                break

            action_result = await run_step(
                execute_decision(ctx, decision, observation, memory),
                "execute action",
            )
            verified = await run_step(
                verify_outcome(ctx, decision, action_result),
                "verify outcome",
            )
            update_memory(memory, observation, decision, verified)
            last_result = verified
            if tracker is not None:
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached after cycle action: {reason}.")
                    break
        except CompletionTimeout as exc:
            if tracker is not None:
                tracker.mark_ended(str(exc))
            print(f"Completion timer expired: {exc}.")
            break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Running Level 0 full-state project starter")
    completion = await prepare_evaluation_round(ctx, level=0)
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=completion,
    )
    print("\nRun complete.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)
