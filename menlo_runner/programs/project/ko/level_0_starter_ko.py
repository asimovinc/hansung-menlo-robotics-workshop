from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 0 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
학생 TODO 섹션은 팀의 프로젝트 설계를 직접 구현하는 부분입니다.

Level 0 규칙: scene_state, 정확한 entity ID, entity-target go_to를 사용할 수 있습니다.
핵심 과제는 고정 script가 아니라 의미 있는 LLM 보조 상위 단계 결정 loop를 구현하는 것입니다.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.scene import COLOR_TO_PAD, delivered_cube_ids, held_cube_info, visible_cubes


# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
TASK = "Find and sort cubes from the source area into their matching destination pads."

# Notebook/Python starter에서 사용할 LLM 모델 선택입니다.
# 이 값을 바꾸거나 실행 전에 환경 변수/.env의 MENLO_LLM_MODEL을 설정하세요.
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
    """LLM이 반환하고 코드가 검증한 상위 단계 결정입니다."""

    next_action: str
    target_color: str | None = None
    target_entity_id: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다."""

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
    """LLM과 실행 코드에 전달할 간결한 full-state 관찰입니다."""

    robot_status: Any
    visible_cubes: list[dict[str, Any]]
    held_cube: dict[str, str] | None
    delivered_cube_ids: list[str]
    color_to_pad: dict[str, str]
    note: str = ""


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 구조화 LLM JSON 출력을 parse하고 validate합니다."""
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
    """Full-state 정보를 LLM에 전달하기 좋은 간결한 text context로 변환합니다."""
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
# 지원 코드: Level 0 SDK wrapper
# ---------------------------------------------------------------------------

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def observe_full_state(ctx: Any) -> Observation:
    """scene_state helper로 프로젝트 Level 0 관찰을 수집합니다."""
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
    """Level 0 entity-target navigation입니다."""
    return await ctx.invoke(
        "go_to",
        {"target": {"kind": "entity", "entity_id": entity_id}},
        timeout_s=300,
    )


async def pick_cube_by_id(ctx: Any, cube_id: str) -> Any:
    """충분히 가까이 navigation한 뒤 특정 cube entity를 pick합니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": cube_id}},
        timeout_s=300,
    )


async def place_on_pad_by_id(ctx: Any, pad_id: str) -> Any:
    """들고 있는 cube를 특정 pad entity에 place합니다."""
    return await ctx.invoke(
        "place_entity",
        {"target": {"kind": "entity", "entity_id": pad_id}},
        timeout_s=300,
    )


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


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
    - build_decision_context(...)로 prompt를 만드세요.
    - menlo_runner.llm.call_llm 또는 승인된 LLM helper를 호출하세요.
    - next_action, target_color, target_entity_id, reason이 포함된 JSON을 요구하세요.
    - 실행 전에 parse_agent_decision으로 validate하세요.

    아래 placeholder는 안전하게 멈추도록 되어 있습니다. 제출 전에는 실제 LLM
    call로 교체하고, 고정 action sequence를 hard-code하지 마세요.
    """
    _decision_context = build_decision_context(task, observation, memory, last_result)
    return AgentDecision(
        next_action="stop",
        reason="TODO: text LLM을 호출하고 JSON decision을 validate한 뒤 실행하세요.",
    )


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------

async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 Level 0 관찰을 수집합니다.

    TODO:
    - 필요하면 natural-language task parsing result를 추가하세요.
    - 필요하면 compact scene summary를 추가하세요.
    - 반복 LLM call에 부담이 없도록 observation을 작게 유지하세요.
    """
    return await observe_full_state(ctx)


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 0 robot 행동으로 변환합니다.

    TODO:
    - 선택한 target이 observation과 일치하는지 validate하세요.
    - 필요한 곳에서 go_to_entity, pick_cube_by_id, place_on_pad_by_id를 사용하세요.
    - 팀 policy에서 search/recover/skip/stop을 어떻게 처리할지 결정하세요.
    - 고정 script를 피하고 LLM과 memory가 high-level sequence를 선택하게 하세요.
    """
    if decision.next_action == "stop":
        return {"action": "stop", "status": "stopped"}

    return {
        "action": decision.next_action,
        "status": "todo",
        "reason": "검증된 decision에 대한 Level 0 action execution을 구현하세요.",
    }


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 scene_state를 다시 observe하세요.
    - Pick 뒤 robot이 cube를 들고 있는지 확인하세요.
    - Place 뒤 delivered_cube_ids를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 fact를 반환하세요.
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
    """각 cycle 뒤 지속 상태를 update합니다.

    TODO:
    - active cube, held cube, delivered count, failure, skip history를 추적하세요.
    - presentation에서 보여줄 수 있는 간결한 log를 남기세요.
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
    """얇은 observe-LLM-act loop입니다. 이 loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 0] Cycle {cycle}")
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

        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(task, observation, memory, last_result)
        print("LLM decision:", decision)

        if decision.next_action == "stop":
            break

        action_result = await execute_decision(ctx, decision, observation, memory)
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        last_result = verified
        if tracker is not None:
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached after cycle action: {reason}.")
                break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 0 full-state project starter 실행")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=0, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)



