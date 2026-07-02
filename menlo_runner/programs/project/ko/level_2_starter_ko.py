from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 2 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
필요하면 읽고 수정할 수 있지만, 대부분의 팀은 지원 코드를 크게 바꾸지 않는 편이 좋습니다.
학생 TODO 섹션은 팀이 수정하고, 개선하고, test하고, presentation에서 설명해야 하는 부분입니다.

실행 설정:
- 기본 run(ctx)는 round1, round2, round3 또는 manual 시간을 묻습니다.
  라운드 제한 시간은 각각 5분, 10분, 15분이며, 모든 라운드는 최대 12개
  cube delivery에서 자동으로 멈춥니다.
- 일반 연습에서는 Enter를 눌러 round2를 사용하고 evaluation setup option은
  비워 두세요. 그러면 현재 scene과 robot pose를 그대로 사용합니다.
- 공통 평가 조건으로 연습할 때는 지정된 round와 1~50 사이 option 번호를
  입력하세요. Starter가 cube_color_order_key를 출력하고, viewer에서 해당
  key를 적용/reset한 뒤 결정된 시작 위치로 robot을 이동합니다.
- manual을 입력하면 원하는 제한 시간을 초 단위로 직접 입력할 수 있습니다.

Level 2 규칙: scene_state, 정확한 entity ID, coordinate go_to는 사용할 수 없습니다.
Camera observation, set_head, set_velocity, memory로 navigation을 구현하세요.
"""

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTimeout, CompletionTracker
from menlo_runner.llm import ask_vlm
from menlo_runner.perception import compress_jpeg, detect_color_blobs
from menlo_runner.programs.evaluation_setup import prepare_evaluation_round
from menlo_runner.scene import delivered_cube_ids, held_cube_info


# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
# 과제 문장은 고정합니다. 목표는 cube 색상 순서와 시작 위치가 달라져도
# 소스 코드 변경 없이 처리하는 하나의 agent를 만드는 것입니다.
TASK = "Find and sort cubes from the source area into their matching destination pads."

# Notebook/Python starter에서 사용할 LLM 모델 선택입니다.
# 이 값을 바꾸거나 실행 전에 환경 변수/.env의 MENLO_LLM_MODEL을 설정하세요.
APPROVED_LLM_MODELS = (
    "minimaxai/minimax-m3",
    "qwen/qwen3.6-35b-a3b",
)
LLM_MODEL = os.environ.setdefault("MENLO_LLM_MODEL", "minimaxai/minimax-m3")
VLM_MODEL = os.environ.setdefault("MENLO_VLM_MODEL", "qwen/qwen3.6-35b-a3b")

# 고정 표지판 정보는 사용할 수 있습니다. 단, 이를 정확한 coordinate나 entity ID로
# 바꾸지 말고 관찰을 해석하는 데만 사용하세요.
DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)

# LLM은 아래 set에서 상위 단계 행동을 선택해야 합니다. 원시 속도 명령을
# 직접 출력하지 말고, 결정적 코드가 결정을 robot 행동으로 변환해야 합니다.
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
    """LLM이 반환하고 코드가 검증한 상위 단계 결정입니다."""

    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다.

    간단하게 시작한 뒤, 팀 전략에 필요한 field를 추가하세요. 예: target history,
    failed location, scan result, confidence score, held-object estimate 등.
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
    """LLM과 실행 코드에 전달할 간결한 관찰입니다."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    """해당 camera frame을 얻을 때 사용한 head pose가 함께 기록된 color detection입니다.

    이 구조는 특정 strategy에 묶이지 않도록 의도적으로 중립적입니다. Level 1 팀은 coordinate estimate에 full bearing을 사용할 수 있고, Level 2 팀은 closed-loop visual centering에 사용할 수 있습니다. 필요하면 confidence, target type, depth field를 추가하세요.
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
        """대략적인 body-relative bearing입니다. Image angle에 head yaw를 더합니다."""
        return self.angle_deg + math.degrees(self.head_yaw)


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 structured LLM JSON output을 parse하고 validate합니다."""
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
    """Robot state를 LLM에 전달하기 좋은 간결한 text context로 변환합니다.

    VLM을 명시적으로 사용하는 경우가 아니라면 raw image는 이 text context에 넣지 마세요. LLM은 다음 high-level step을 고를 만큼의 정보만 받고, low-level control과 safety는 code가 처리해야 합니다.
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
# 지원 코드: project 규칙에 맞는 SDK wrapper
# ---------------------------------------------------------------------------
# 이 래퍼들은 프로젝트 규칙에 맞는 input을 노출합니다. 아래 progress helper는
# completion과 robot이 cube를 들고 있는지 추적할 수 있도록 허용됩니다.
# Ground-truth coordinate, 정확한 target ID, global asset map은 추가하지 마세요.

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def get_camera_frame(
    ctx: Any,
    *,
    compressed: bool = False,
    max_width: int = 800,
    quality: int = 70,
) -> bytes:
    """POV camera frame을 가져오며, VLM용으로 resize/re-encode할 수 있습니다."""
    jpeg = await ctx.get_vision("pov")
    if compressed:
        return compress_jpeg(jpeg, max_width=max_width, quality=quality)
    return jpeg


async def get_delivered_count(ctx: Any) -> int:
    """공통 workshop progress helper로 delivered cube 수를 셉니다."""
    return len(await delivered_cube_ids(ctx))


async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    """Robot이 cube를 들고 있으면 현재 held cube id/color를 반환합니다."""
    held = await held_cube_info(ctx)
    return {"entity_id": held[0], "color": held[1]} if held else None


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """고정 warehouse signage를 읽기 위한 strategy-neutral prompt를 만듭니다."""
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" Robot이 {held_color} cube를 들고 있으므로 target destination sign은 {DESTINATION_SIGN_RULES[held_color]}입니다."
    return (
        "이 robot camera frame에 보이는 warehouse sign을 읽으세요. "
        f"{SIGNAGE_NOTE} "
        "보이는 sign letter, color, 대략적인 left/center/right 위치, confidence를 JSON으로 반환하세요."
        + target
    )


async def ask_vlm_about_frame(
    ctx: Any,
    prompt: str,
    *,
    api_key: str,
    compressed: bool = True,
    max_width: int = 800,
    quality: int = 70,
) -> str:
    """Project에서 허용되는 VLM helper로 현재 POV frame에 대해 질문합니다."""
    jpeg = await get_camera_frame(
        ctx,
        compressed=compressed,
        max_width=max_width,
        quality=quality,
    )
    return ask_vlm(jpeg, prompt, api_key=api_key)


async def perceive(ctx: Any) -> list[Any]:
    """현재 camera frame에서 Workshop 2 color-blob detector를 실행합니다."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """Walking direction을 바꾸지 않고 camera 방향을 조정합니다."""
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
    """짧은 body-frame velocity command를 보낸 뒤 멈춥니다."""
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=30,
    )


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Code가 robot을 시각적으로 충분히 위치시킨 뒤 nearest cube를 집습니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """Matching pad에 도달한 뒤 nearest zone에 place합니다."""
    return await ctx.invoke("place_entity", {}, timeout_s=300)


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
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
# 학생 TODO: LLM decision 함수
# ---------------------------------------------------------------------------
async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Text LLM을 사용해 다음 상위 단계 행동을 선택합니다.

    TODO:
    - decision_context로 명확한 prompt를 만드세요.
    - menlo_runner.llm.call_llm 또는 승인된 LLM helper를 호출하세요.
      helper가 synchronous/blocking이면 await asyncio.to_thread(...)로 감싸세요.
      그래야 strict round timer가 시간 초과 시 model 대기를 중단할 수 있습니다.
    - next_action, target_color, reason이 포함된 JSON을 요구하세요.
    - parse_agent_decision으로 validate하세요.
    - Validation이 실패하면 안전한 recovery decision을 반환하세요.

    아래 fallback은 의도적으로 약하게 만들어져 있습니다. 제출 전에는 교체하세요.
    """
    decision_context = build_decision_context(task, observation, memory, last_result)

    # Prompt 예시 형태:
    # system: 이 schema에 맞는 JSON만 반환하도록 요구합니다.
    # {"next_action": "search_cube", "target_color": "red", "reason": "..."}
    # user: json.dumps(decision_context)

    visible = decision_context["visible_targets"]
    if not visible:
        return AgentDecision(next_action="search_cube", reason="대체 동작: 보이는 target이 없습니다.")

    largest = max(visible, key=lambda item: item["blob_area"])
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=largest["color"],
        reason="대체 동작: 가장 큰 visible color blob을 선택합니다.",
    )


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 관찰을 수집합니다.

    TODO:
    - 언제 set_head scan을 사용할지, 언제 single frame을 사용할지 결정하세요.
    - 필요하면 VLM output, confidence, target type, search note를 추가하세요.
      Signage에는 build_signage_vlm_prompt()와 ask_vlm_about_frame()을 사용하세요.
    - 제출 code에서는 scene_state와 정확한 entity ID를 사용하지 마세요.
    """
    robot_status = await get_robot_status(ctx)
    detections = await scan_head(ctx)
    return Observation(robot_status=robot_status, detections=detections)


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 observe하세요.
    - robot_status, camera evidence, SDK result status를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 정보를 반환하세요.
    """
    held = await get_held_cube_info(ctx)
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": await get_delivered_count(ctx),
        "held_cube": held,
        "held_color": held["color"] if held else None,
    }


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 지속 상태를 update합니다.

    TODO:
    - completed cube, held color, failed attempt, recovery history를 추적하세요.
    - interim/final presentation에서 보여줄 수 있는 간결한 log를 남기세요.
    """
    if "delivered_count" in verified:
        memory.delivered_count = int(verified["delivered_count"])
    memory.held_color = verified.get("held_color")

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# LEVEL 2 학생 TODO: vision-only action 구현
# ---------------------------------------------------------------------------
# Level 2는 go_to를 호출하면 안 됩니다. Camera observation, set_head,
# set_velocity, memory, recovery behavior로 navigate하세요.


async def visual_search(ctx: Any, target_color: str | None = None) -> bool:
    """Camera movement와 robot motion으로 cube 또는 pad를 search합니다.

    TODO:
    - set_head 또는 body rotation을 사용하는 scan pattern을 설계하세요.
    - 필요하면 cube와 pad를 어떻게 구분할지 결정하세요.
    - Visual centering에 도움이 되면 detection.full_bearing_deg를 사용하세요.
    - 유용한 target을 찾았는지 반환하세요.
    """
    await scan_head(ctx)
    return False


async def visual_navigate_to_target(ctx: Any, target_color: str | None) -> bool:
    """선택한 target까지 closed-loop vision-only navigation을 수행합니다.

    TODO:
    - Observe, 짧게 move, 다시 observe한 뒤 correct하거나 stop하세요.
    - Target loss, overshoot, obstacle을 처리하세요.
    - Steering correction에 도움이 되면 detection.full_bearing_deg를 사용하세요.
    - set_head와 set_velocity만 사용하고 go_to는 호출하지 마세요.
    """
    return False


async def recover_motion(ctx: Any, memory: AgentMemory, reason: str | None = None) -> dict[str, Any]:
    """Target loss, blocked motion, failed manipulation에서 recover합니다.

    TODO:
    - Step back, rotate, rescan, detour 선택, LLM skip 요청 등을 구현하세요.
    - 같은 failed action을 무한 반복하지 않도록 memory를 사용하세요.
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
    """검증된 LLM 결정 하나를 Level 2 robot 행동으로 변환합니다.

    TODO:
    - go_to 없이 search/navigation을 구현하세요.
    - Intended cube 가까이에 visual positioning한 뒤에만 pick하세요.
    - Matching pad 가까이에 visual positioning한 뒤에만 place하세요.
    - Target이 사라지거나 movement가 실패하면 recovery를 사용하세요.
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


async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. 이 loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    async def run_step(awaitable: Any, label: str) -> Any:
        if tracker is None:
            return await awaitable
        return await tracker.wait_for_remaining(awaitable, label)

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 2] Cycle {cycle}")
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
                decide_next_action(TASK, observation, memory, last_result),
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
    print("Level 2 autonomous-vision project starter 실행")
    completion = await prepare_evaluation_round(ctx, level=2)
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=completion,
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)



