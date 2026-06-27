from __future__ import annotations

"""Level 1 프로젝트 스타터입니다.

이 파일은 완성된 해답이 아니라 최소 scaffold입니다.

SUPPORT CODE 영역은 반복해서 작성할 필요가 없는 wrapper, 자료 구조,
schema validation을 제공합니다. STUDENT TODO 영역은 팀이 직접 설계하고,
개선하고, 테스트하고, 발표에서 설명해야 하는 부분입니다.

Level 1 규칙: `scene_state`와 정확한 entity ID는 사용할 수 없습니다.
camera observation과 `set_velocity`로 먼저 접근하고, coordinate `go_to`는
pick/place 성공을 검증한 뒤 학생 시스템이 기록한 좌표에만 사용하세요.
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.llm import ask_vlm
from menlo_runner.perception import detect_color_blobs


# ---------------------------------------------------------------------------
# SUPPORT CODE: 공통 과제 정의와 필수 LLM decision schema
# ---------------------------------------------------------------------------
TASK = "Find and sort the six cubes in the warehouse into their matching destination pads."

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
    """LLM이 반환하고 코드가 검증한 고수준 decision입니다."""

    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 유지하는 agent 상태입니다."""

    delivered_count: int = 0
    delivery_limit: int | None = None
    priority_colors: list[str] = field(default_factory=list)
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    search_turns: int = 0
    learned_cube_source_xy: tuple[float, float] | None = None
    learned_pad_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 action code에 전달할 compact observation입니다."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    """head pose를 함께 저장한 색상 detection입니다."""

    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        """image angle에 head yaw를 더한 대략적인 body-relative bearing입니다."""
        return self.angle_deg + math.degrees(self.head_yaw)


def parse_agent_decision(text: str) -> AgentDecision | None:
    """LLM의 JSON 응답을 parse하고 필수 schema를 검증합니다."""
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
    """robot state를 LLM에 전달하기 좋은 compact text context로 변환합니다."""
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
        "delivery_limit": memory.delivery_limit,
        "priority_colors": memory.priority_colors,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
        "vlm_summary": observation.vlm_summary,
    }


# ---------------------------------------------------------------------------
# SUPPORT CODE: Level 1에서 허용되는 SDK wrapper
# ---------------------------------------------------------------------------
# 이 wrapper에는 scene_state, 정답 좌표, 정확한 cube/pad entity ID를 추가하지 마세요.

async def get_robot_status(ctx: Any) -> Any:
    """robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def get_camera_frame(ctx: Any) -> bytes:
    """POV camera frame을 가져옵니다."""
    return await ctx.get_vision("pov")


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """고정 창고 표지판을 읽기 위한 VLM prompt입니다."""
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
    """현재 POV frame에 대해 project-allowed VLM helper로 질문합니다."""
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)


async def perceive(ctx: Any) -> list[Any]:
    """현재 camera frame에서 Workshop 2 color-blob detector를 실행합니다."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """walking direction은 바꾸지 않고 camera 방향만 조정합니다."""
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
    """짧은 body-frame velocity command를 보냅니다."""
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=30,
    )


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """의도한 cube 가까이 시각적으로 이동한 뒤 nearest cube를 집습니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """matching pad 가까이 이동한 뒤 nearest zone에 내려놓습니다."""
    return await ctx.invoke("place_entity", {}, timeout_s=300)


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log에 넣기 쉬운 작은 dictionary로 변환합니다."""
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
    """간단한 scan helper입니다. 더 나은 search 전략으로 교체할 수 있습니다."""
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
# STUDENT TODO: LLM decision 함수
# ---------------------------------------------------------------------------

async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """text LLM으로 다음 고수준 action을 선택합니다.

    TODO:
    - decision_context로 명확한 prompt를 만드세요.
    - menlo_runner.llm.call_llm 또는 승인된 LLM helper를 호출하세요.
    - next_action, target_color, reason을 포함한 JSON을 요구하세요.
    - parse_agent_decision으로 검증하세요.
    - 검증 실패 시 안전한 recovery decision을 반환하세요.

    아래 fallback은 의도적으로 약합니다. 제출 전 반드시 교체하세요.
    """
    decision_context = build_decision_context(task, observation, memory, last_result)

    # Prompt 예시:
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
# STUDENT TODO: observation, verification, memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 action code에 전달할 현재 observation을 수집합니다.

    TODO:
    - 언제 set_head scan을 할지, 언제 single frame만 사용할지 결정하세요.
    - 필요하다면 VLM output, confidence, target type, search note를 추가하세요.
    - 제출 코드에서는 scene_state와 정확한 entity ID를 사용하지 마세요.
    """
    robot_status = await get_robot_status(ctx)
    detections = await scan_head(ctx)
    return Observation(robot_status=robot_status, detections=detections)


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공했는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 관찰하세요.
    - robot_status, camera evidence, SDK result status를 함께 확인하세요.
    - 다음 LLM call의 recovery에 사용할 정보를 반환하세요.
    """
    return {"decision": decision.__dict__, "action_result": action_result}


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 persistent state를 갱신합니다.

    TODO:
    - completed cube, held color, failed attempts, recovery history를 추적하세요.
    - 발표에서 보여줄 수 있는 간결한 log를 남기세요.
    """
    memory.logs.append(
        {
            "observation": {
                "visible_count": len(observation.detections),
                "note": observation.note,
            },
            "llm_decision": decision.__dict__,
            "verified": verified,
        }
    )


# ---------------------------------------------------------------------------
# LEVEL 1 STUDENT TODO: visual approach와 coordinate-memory 구현
# ---------------------------------------------------------------------------
# Level 1은 camera observation과 set_velocity로 먼저 접근합니다.
# coordinate go_to는 pick/place 성공 후 기록한 좌표에만 사용하세요.
# entity ID, scene_state, 정답 좌표는 사용하지 마세요.

def recorded_xy_for_decision(memory: AgentMemory, decision: AgentDecision) -> tuple[float, float] | None:
    """Return a coordinate recorded after earlier success, if one exists."""
    if decision.next_action == "navigate_to_cube":
        return memory.learned_cube_source_xy
    if decision.next_action == "navigate_to_pad" and decision.target_color:
        return memory.learned_pad_xy.get(decision.target_color)
    return None


async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    """Coordinate-based go_to. Use only with coordinates recorded after success."""
    return await ctx.invoke(
        "go_to",
        {
            "target": {
                "kind": "pose",
                "pose": {"frame_id": "world", "position": [x, y, 0]},
            }
        },
        timeout_s=300,
    )


def best_detection(detections: list[Any], target_color: str | None = None) -> Any | None:
    """Choose the largest matching color blob from a scan."""
    candidates = [
        detection
        for detection in detections
        if target_color is None or detection.color == target_color
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda detection: detection.blob_area)


async def visual_approach_until_close(ctx: Any, target_color: str, *, arrival_area: int = 7000) -> bool:
    """Starter closed-loop approach: observe, move briefly, observe again."""
    for _ in range(10):
        detections = await scan_head(ctx, yaws=(0.0,), pitch=0.1)
        detection = best_detection(detections, target_color)
        if detection is None:
            await move_velocity(ctx, vx=0.0, wz=0.25, duration_s=0.6)
            continue
        if detection.blob_area >= arrival_area:
            return True
        wz = -0.2 if detection.angle_deg > 8 else 0.2 if detection.angle_deg < -8 else 0.0
        await move_velocity(ctx, vx=0.3, wz=wz, duration_s=0.7)
    return False


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM decision 하나를 Level 1 robot action으로 변환합니다.

    TODO:
    - search action에서는 안전하게 scan하거나 reposition하세요.
    - navigation action에서는 set_velocity로 시각 기반 접근을 구현하세요.
    - go_to_xy는 이전 pick/place 성공 후 기록한 coordinate에만 사용하세요.
    - pick/place 전에는 의도한 target 가까이에 있는지 검증하세요.
    - recover/skip/stop에 대한 팀의 policy를 구현하세요.
    """
    if decision.next_action in {"search_cube", "search_pad"}:
        await scan_head(ctx)
        return {"action": decision.next_action, "status": "scanned"}

    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        recorded_xy = recorded_xy_for_decision(memory, decision)
        go_to_result = None
        if recorded_xy is not None:
            result = await go_to_xy(ctx, *recorded_xy)
            go_to_result = result_summary(result)
        if decision.target_color is None:
            return {"action": decision.next_action, "status": "failed", "reason": "missing target color"}
        approached = await visual_approach_until_close(ctx, decision.target_color)
        return {
            "action": decision.next_action,
            "status": "visually_approached" if approached else "failed",
            "recorded_go_to": go_to_result,
            "approached": approached,
        }

    if decision.next_action == "pick_cube":
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        result = await place_nearest_zone(ctx)
        return {"action": "place_cube", "result": result_summary(result)}

    if decision.next_action == "recover":
        await move_velocity(ctx, vx=-0.15, duration_s=0.8)
        return {"action": "recover", "status": "stepped_back"}

    return {"action": decision.next_action, "status": "no_op"}


async def run_agent(ctx: Any, *, max_cycles: int = 20) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 1] Cycle {cycle}")
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
    print("Running Level 1 adaptive-navigation project starter")
    memory = await run_agent(ctx)
    print("\nRun complete.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)
