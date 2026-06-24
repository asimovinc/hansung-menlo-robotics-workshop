from __future__ import annotations

"""Project starter for the Menlo AI robot sorting challenge.

This file is intentionally a starter, not a solution.

Sections marked SUPPORT CODE provide small wrappers and data structures so you do
not need to rewrite workshop setup code. You may read them and adapt them if your
architecture needs it, but most teams should leave them mostly unchanged.

Sections marked STUDENT TODO are where your project design belongs. These are the
parts your team should edit, improve, test, and explain in the presentation.
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.llm import ask_vlm
from menlo_runner.perception import detect_color_blobs


# ---------------------------------------------------------------------------
# SUPPORT CODE: shared task definition and required LLM decision schema
# ---------------------------------------------------------------------------
# Keep the task fixed. The challenge is to make one agent that handles different
# cube-color orders and starting positions without source-code changes.
TASK = "Find and sort the six cubes in the warehouse into their matching destination pads."

# Fixed signage is allowed information. Do not turn this into exact coordinates
# or entity IDs; use it only to interpret observations.
DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
SIGNAGE_NOTE = (
    "A is the conveyor/cube source area, not a destination. "
    "Destination signs are B red, C green, D blue, E yellow."
)

# The LLM must choose high-level actions from this set. It should not output raw
# velocity commands; deterministic code should translate decisions into robot actions.
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
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """State your agent carries across observe-decide-act cycles.

    Start simple, then add fields your strategy needs: target history, failed
    locations, scan results, confidence scores, held-object estimates, etc.
    """

    delivered_count: int = 0
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """Compact observation passed to the LLM and action code."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    """Color detection annotated with the head pose used for that camera frame.

    This is intentionally strategy-neutral. Level 1 teams can use the full
    bearing for coordinate estimates; Level 2 teams can use it for closed-loop
    visual centering. Teams may add confidence, target type, or depth fields.
    """

    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        """Approximate body-relative bearing: image angle plus head yaw."""
        return self.angle_deg + math.degrees(self.head_yaw)


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

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert robot state into a compact text context for the LLM.

    Keep raw images out of this text context unless you are explicitly using a
    VLM. The LLM should receive enough information to choose the next high-level
    step, while your code handles low-level control and safety.
    """
    visible = [
        {
            "color": detection.color,
            "angle_deg": detection.angle_deg,
            "full_bearing_deg": round(getattr(detection, "full_bearing_deg", detection.angle_deg), 1),
            "blob_area": detection.blob_area,
            "bbox": detection.bbox,
        }
        for detection in observation.detections
    ]
    return {
        "task": task,
        "visible_targets": visible,
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "delivered_count": memory.delivered_count,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
        "vlm_summary": observation.vlm_summary,
    }


# ---------------------------------------------------------------------------
# SUPPORT CODE: project-safe SDK wrappers
# ---------------------------------------------------------------------------
# These wrappers expose only project-safe inputs. Do not add scene_state,
# ground-truth coordinates, exact cube IDs, or global asset maps here.

async def get_robot_status(ctx: Any) -> Any:
    """Read robot pose, motion status, and neck state."""
    return await ctx.state("robot_status")


async def get_camera_frame(ctx: Any) -> bytes:
    """Capture the POV camera frame."""
    return await ctx.get_vision("pov")


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """Build a strategy-neutral prompt for reading fixed warehouse signage."""
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" The robot is holding a {held_color} cube, so the target destination sign is {DESTINATION_SIGN_RULES[held_color]}."
    return (
        "Read the floating warehouse signs visible in this robot camera frame. "
        f"{SIGNAGE_NOTE} "
        "Return JSON with visible sign letters, colors, rough left/center/right positions, and confidence."
        + target
    )


async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    """Ask the project-allowed VLM helper about the current POV frame."""
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)


async def perceive(ctx: Any) -> list[Any]:
    """Run the Workshop 2 color-blob detector on the current camera frame."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """Aim the camera without changing the walking direction."""
    args: dict[str, float] = {}
    if yaw is not None:
        args["yaw"] = yaw
    if pitch is not None:
        args["pitch"] = pitch
    return await ctx.invoke("set_head", args, timeout_s=10)


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """Send one short body-frame velocity command, then stop."""
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=30,
    )


async def cancel_action(ctx: Any) -> Any:
    """Cancel the currently active runtime action."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Pick the nearest cube after your code has visually positioned the robot."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """Place on the nearest zone after your code has reached the matching pad."""
    return await ctx.invoke("place_entity", {}, timeout_s=300)


def result_summary(result: Any) -> dict[str, Any]:
    """Convert SDK results into a small loggable dictionary."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


async def scan_head(
    ctx: Any,
    *,
    yaws: tuple[float, ...] = (-0.8, 0.0, 0.8),
    pitch: float = 0.15,
) -> list[Any]:
    """Simple scan helper. Students can replace this with richer search."""
    all_detections: list[Any] = []
    for yaw in yaws:
        await set_head(ctx, yaw=yaw, pitch=pitch)
        await asyncio.sleep(0.4)
        for detection in await perceive(ctx):
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    head_yaw=yaw,
                    head_pitch=pitch,
                )
            )
    return all_detections


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
    - Build a clear prompt from decision_context.
    - Call menlo_runner.llm.call_llm or your approved LLM helper.
    - Require JSON with next_action, target_color, and reason.
    - Validate with parse_agent_decision.
    - If validation fails, return a safe recovery decision.

    The fallback below is intentionally weak. Replace it for submission.
    """
    decision_context = build_decision_context(task, observation, memory, last_result)

    # Example prompt shape:
    # system: Return ONLY JSON using this schema:
    # {"next_action": "search_cube", "target_color": "red", "reason": "..."}
    # user: json.dumps(decision_context)

    visible = decision_context["visible_targets"]
    if not visible:
        return AgentDecision(next_action="search_cube", reason="Fallback: no visible target.")

    largest = max(visible, key=lambda item: item["blob_area"])
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=largest["color"],
        reason="Fallback: choose the largest visible color blob.",
    )


# ---------------------------------------------------------------------------
# STUDENT TODO: observation, execution, verification, and memory
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """Collect the current observation for both the LLM and your action code.

    TODO:
    - Decide when to scan with set_head and when to use a single frame.
    - Add VLM output, confidence, target type, or search notes if useful.
      For signage, use build_signage_vlm_prompt() with ask_vlm_about_frame().
    - Keep scene_state and exact entity IDs out of submitted code.
    """
    robot_status = await get_robot_status(ctx)
    detections = await scan_head(ctx)
    return Observation(robot_status=robot_status, detections=detections)


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """Check whether the last action appears to have worked.

    TODO:
    - Re-observe after important actions.
    - Check robot_status, camera evidence, and SDK result status.
    - Return information the next LLM call can use for recovery.
    """
    return {"decision": decision.__dict__, "action_result": action_result}


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """Update persistent state after each cycle.

    TODO:
    - Track completed cubes, held color, failed attempts, and recovery history.
    - Add concise logs that you can show during interim/final presentations.
    """
    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# LEVEL 2 STUDENT TODO: vision-only action implementation
# ---------------------------------------------------------------------------
# Level 2 must not invoke go_to. Navigate with camera observations, set_head,
# set_velocity, memory, and recovery behavior.


async def visual_search(ctx: Any, target_color: str | None = None) -> bool:
    """Search for a cube or pad using camera movement and robot motion.

    TODO:
    - Design scan patterns with set_head and/or body rotation.
    - Decide how to distinguish cubes from pads if needed.
    - Use detection.full_bearing_deg when useful for visual centering.
    - Return whether a useful target was found.
    """
    await scan_head(ctx)
    return False


async def visual_navigate_to_target(ctx: Any, target_color: str | None) -> bool:
    """Closed-loop vision-only navigation to a selected target.

    TODO:
    - Observe, move briefly, observe again, then correct or stop.
    - Handle target loss, overshoot, and obstacles.
    - Use detection.full_bearing_deg when useful for steering corrections.
    - Use set_head and set_velocity only; do not call go_to.
    """
    return False


async def recover_motion(ctx: Any, memory: AgentMemory, reason: str | None = None) -> dict[str, Any]:
    """Recover from target loss, blocked motion, or failed manipulation.

    TODO:
    - Step back, rotate, rescan, choose a detour, or ask the LLM to skip.
    - Use memory to avoid repeating the same failed action forever.
    """
    await move_velocity(ctx, vx=-0.15, duration_s=0.8)
    await move_velocity(ctx, wz=0.35, duration_s=0.8)
    return {"action": "recover", "reason": reason, "status": "stepped_back_and_rotated"}


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """Translate one validated LLM decision into a Level 2 robot action.

    TODO:
    - Implement search/navigation without go_to.
    - Pick only after visual positioning near the intended cube.
    - Place only after visual positioning near the matching pad.
    - Use recovery when the target disappears or movement fails.
    """
    if decision.next_action in {"search_cube", "search_pad"}:
        found = await visual_search(ctx, decision.target_color)
        return {"action": decision.next_action, "found": found}

    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        reached = await visual_navigate_to_target(ctx, decision.target_color)
        return {"action": decision.next_action, "reached": reached}

    if decision.next_action == "pick_cube":
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        result = await place_nearest_zone(ctx)
        return {"action": "place_cube", "result": result_summary(result)}

    if decision.next_action == "recover":
        return await recover_motion(ctx, memory, decision.recovery_strategy)

    return {"action": decision.next_action, "status": "no_op"}


async def run_agent(ctx: Any, *, max_cycles: int = 20) -> AgentMemory:
    """Thin observe-LLM-act loop. Edit the TODO functions, not just this loop."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 2] Cycle {cycle}")
        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(TASK, observation, memory, last_result)
        print("LLM decision:", decision)

        if decision.next_action == "stop":
            break

        action_result = await execute_decision(ctx, decision, observation, memory)
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        last_result = verified

    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Running Level 2 vision-guided project starter")
    memory = await run_agent(ctx)
    print("\nRun complete.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)
