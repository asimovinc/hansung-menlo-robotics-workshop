from __future__ import annotations

"""Level 1 coordinate-guided sorting agent.

Level 1 규칙을 지킵니다.
- scene_state를 navigation 입력으로 사용하지 않습니다.
- 정확한 cube/pad entity id나 ground-truth object coordinate를 사용하지 않습니다.
- camera observation과 robot_status로 추정한 world coordinate에만 pose go_to를 사용합니다.
"""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.llm import ask_vlm, call_llm
from menlo_runner.perception import decode_jpeg, detect_color_blobs
from menlo_runner.programs.evaluation_setup import (
    DEFAULT_OBSTACLE_CLEARANCE_M,
    apply_clear_start_from_layout,
    choose_evaluation_setup,
    go_to_start_position,
    reload_current_scene,
)
from menlo_runner.scene import held_cube_info


TASK = "Find and sort cubes from the source area into their matching destination pads."

DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}

SIGN_BACKGROUND_COLORS = {
    "A": "green",
    "B": "red",
    "C": "green",
    "D": "blue",
    "E": "yellow",
}

SCAN_YAWS = (0.0, -0.8, 0.8)
SCAN_PITCH = 0.15
HEAD_SETTLE_S = 0.2
SAVE_DEBUG_IMAGES = True
DEBUG_DIR = Path("outputs/my_real_level_1")

LETTER_TEMPLATE_SIZE = 84
LETTER_MIN_SCORE = 0.42

CUBE_PICK_MIN_AREA = 16000
CUBE_APPROACH_AREA = 9000
CUBE_DISTANCE_K = 260.0
SIGN_DISTANCE_K = 125.0
PAD_SIGN_DISTANCE_K = 220.0
IMAGE_BEARING_GAIN = 0.90
PAD_IMAGE_BEARING_GAIN = 2.15
SOURCE_APPROACH_STANDOFF_M = 0.20
SOURCE_TRAVEL_SCALE = 1.00
SOURCE_FORWARD_RETRY_S = 0.8
PAD_APPROACH_STANDOFF_M = 0.70
PAD_REFINE_STANDOFF_M = 0.55
PAD_TRAVEL_SCALE = 0.75
PAD_ALIGN_SCALE = 0.95
PAD_CLOSE_DISTANCE_M = 5.00
PAD_CLOSE_TRAVEL_SCALE = 0.55
PAD_CLOSE_ALIGN_SCALE = 0.75
SEARCH_TURN_DEGREES = 130
GOTO_STUCK_SECONDS = 1.5
GOTO_STUCK_MOVE_EPS_M = 0.05
VLM_IMAGE_MAX_DIMENSION = 768
VLM_IMAGE_JPEG_QUALITY = 68
STUCK_VLM_MODELS = ("qwen/qwen3.6-35b-a3b", "minimaxai/minimax-m3")
LLM_DECISION_MODELS = ("minimaxai/minimax-m3", "qwen/qwen3.6-35b-a3b")
SIDE_HEAD_YAW_EPS = 0.1
LETTER_SCAN_CACHE_MAX_XY_DELTA_M = 0.15
LETTER_SCAN_CACHE_MAX_YAW_DELTA_DEG = 8.0

MAX_DELIVERIES = 4


@dataclass
class AgentMemory:
    delivered_count: int = 0
    held_color: str | None = None
    stage: str = "need_cube"
    active_color: str | None = None
    source_estimate: tuple[float, float] | None = None
    last_pick_xy: tuple[float, float] | None = None
    pad_estimates: dict[str, tuple[float, float]] = field(default_factory=dict)
    letter_scan_cache: dict[str, LetterDetection] = field(default_factory=dict)
    letter_frame_cache: dict[str, list[ScanFrame]] = field(default_factory=dict)
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    last_result: dict[str, Any] | None = None
    cycle: int = 0


@dataclass(frozen=True)
class ScanFrame:
    index: int
    yaw: float
    pitch: float
    jpeg: bytes
    path: Path | None = None
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_yaw_deg: float = 0.0


@dataclass(frozen=True)
class ScannedColor:
    color: str
    angle_deg: float
    full_bearing_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    frame: ScanFrame


@dataclass(frozen=True)
class LetterDetection:
    letter: str
    score: float
    angle_deg: float
    full_bearing_deg: float
    bbox: tuple[int, int, int, int]
    frame: ScanFrame

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class Observation:
    robot_status: Any
    held_color: str | None
    delivered_count: int
    colors: list[ScannedColor]


@dataclass
class PlaceAttempt:
    result: Any
    held_after: str | None
    delivered_after: int
    released: bool
    scored: bool
    placed: bool


@dataclass(frozen=True)
class StuckMoveAdvice:
    action: str
    duration_s: float
    reason: str


@dataclass(frozen=True)
class AgentDecision:
    action: str
    target_color: str | None = None
    target_letter: str | None = None
    reason: str = ""


async def get_robot_status(ctx: Any) -> Any:
    return await ctx.state("robot_status")


async def get_camera_frame(ctx: Any) -> bytes:
    return await ctx.get_vision("pov")


async def get_delivered_count(ctx: Any, tracker: CompletionTracker | None = None) -> int:
    if tracker is None:
        return 0
    return await tracker.scene_delivered_count(ctx)


async def get_held_cube_color(ctx: Any) -> str | None:
    held = await held_cube_info(ctx)
    return held[1] if held else None


async def set_head(ctx: Any, *, yaw: float = 0.0, pitch: float = SCAN_PITCH) -> Any:
    return await ctx.invoke("set_head", {"yaw": yaw, "pitch": pitch}, timeout_s=15)


async def cancel_action(ctx: Any) -> Any:
    print("▶️ 이동명령: 현재 동작 cancel 호출")
    return await ctx.invoke("cancel", {}, timeout_s=15)


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    print(f"▶️ 이동명령: vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} duration={duration_s:.2f}s")
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=max(30, int(duration_s + 20)),
    )


async def turn_scan(ctx: Any, *, direction: str = "left", degrees: int = 90) -> None:
    duration = 4.0 if degrees == SEARCH_TURN_DEGREES else (5.8 if degrees >= 180 else 3.14)
    if direction == "right":
        print(f"▶️ 회전명령: 탐색을 위해 뒤로 가며 오른쪽 약 {degrees}도")
        await move_velocity(ctx, vx=-0.2, wz=-0.5, duration_s=duration)
    else:
        print(f"▶️ 회전명령: 탐색을 위해 뒤로 가며 왼쪽 약 {degrees}도")
        await move_velocity(ctx, vx=-0.2, wz=0.5, duration_s=duration)


def is_provider_fallback(text: str) -> bool:
    lowered = text.lower()
    return (
        "fallback response" in lowered
        or "trouble reaching the model" in lowered
        or "cannot access the image" in lowered
        or "unable to view" in lowered
    )


def jpeg_debug_info(jpeg: bytes) -> str:
    try:
        from PIL import Image

        with Image.open(BytesIO(jpeg)) as image:
            return f"{image.width}x{image.height}, {len(jpeg) / 1024:.1f}KB"
    except Exception:
        return f"unknown-dim, {len(jpeg) / 1024:.1f}KB"


def compress_jpeg_for_vlm(jpeg: bytes, *, label: str) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        print(f"AI 이미지 압축 경고: PIL이 없어 원본을 전송합니다. {label}={jpeg_debug_info(jpeg)}")
        return jpeg
    try:
        with Image.open(BytesIO(jpeg)) as image:
            image = image.convert("RGB")
            image.thumbnail((VLM_IMAGE_MAX_DIMENSION, VLM_IMAGE_MAX_DIMENSION))
            output = BytesIO()
            image.save(output, format="JPEG", quality=VLM_IMAGE_JPEG_QUALITY, optimize=True)
            compressed = output.getvalue()
    except Exception as exc:
        print(f"AI 이미지 압축 경고: 실패하여 원본을 전송합니다. 원인={exc}")
        return jpeg
    print(
        f"AI 이미지 압축: {label} | "
        f"original={jpeg_debug_info(jpeg)} -> compressed={jpeg_debug_info(compressed)}"
    )
    return compressed or jpeg


def parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = match.group(1) if match else None
    if blob is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        blob = match.group(0) if match else None
    if blob is None:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_stuck_move_advice(text: str) -> StuckMoveAdvice | None:
    data = parse_json_object(text)
    if data is None:
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in {"back_up", "turn_left", "turn_right", "forward", "retry_go_to", "stop_retry_next_cycle"}:
        return None
    try:
        duration_s = float(data.get("duration_s", 1.0))
    except (TypeError, ValueError):
        duration_s = 1.0
    duration_s = max(0.0, min(duration_s, 3.0))
    reason = str(data.get("reason", "")).strip()
    return StuckMoveAdvice(action=action, duration_s=duration_s, reason=reason)


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def build_llm_state(observation: Observation, memory: AgentMemory) -> dict[str, Any]:
    pad_estimates = {
        letter: {"x": round(x, 2), "y": round(y, 2)}
        for letter, (x, y) in sorted(memory.pad_estimates.items())
    }
    last_pick_xy = None
    if memory.last_pick_xy is not None:
        last_pick_xy = {"x": round(memory.last_pick_xy[0], 2), "y": round(memory.last_pick_xy[1], 2)}
    return {
        "stage": memory.stage,
        "held_color": observation.held_color,
        "delivered_count": observation.delivered_count,
        "last_pick_xy": last_pick_xy,
        "known_destination_coordinates": pad_estimates,
        "completed_colors": list(memory.completed_colors),
        "last_result": memory.last_result,
        "rules": {
            "source_sign": "A",
            "destinations": DESTINATION_SIGN_RULES,
        },
    }


def build_llm_decision_prompt(state: dict[str, Any]) -> str:
    return (
        "You are the high-level task controller for a warehouse sorting robot. "
        "You receive text state only; no images are provided. "
        "Do not output raw movement, coordinates, velocity, or camera instructions. "
        "Choose the next semantic action for the executor.\n\n"
        "Rules:\n"
        "- If held_color is null, choose pick_source with target_letter A.\n"
        "- If held_color is red, choose deliver_held_cube with target_letter B.\n"
        "- If held_color is green, choose deliver_held_cube with target_letter C.\n"
        "- If held_color is blue, choose deliver_held_cube with target_letter D.\n"
        "- If held_color is yellow, choose deliver_held_cube with target_letter E.\n"
        "- If a previous place succeeded and held_color is null, choose pick_source; the executor may use a saved pick coordinate before searching A.\n"
        "- Use recover only if last_result says the action failed and repeating the normal action is unsafe.\n\n"
        "Return JSON only with this schema:\n"
        '{"action":"pick_source|deliver_held_cube|recover|stop","target_color":null,'
        '"target_letter":"A|B|C|D|E|null","reason":"short reason"}\n\n'
        f"Current state:\n{safe_json(state)}"
    )


def parse_agent_decision(text: str, observation: Observation) -> AgentDecision | None:
    data = parse_json_object(text)
    if data is None:
        return None
    action = str(data.get("action", "")).strip().lower()
    aliases = {
        "search_source": "pick_source",
        "navigate_to_source": "pick_source",
        "pick_cube": "pick_source",
        "go_destination": "deliver_held_cube",
        "navigate_to_pad": "deliver_held_cube",
        "place_cube": "deliver_held_cube",
        "deliver": "deliver_held_cube",
    }
    action = aliases.get(action, action)
    if action not in {"pick_source", "deliver_held_cube", "recover", "stop"}:
        return None

    target_color = data.get("target_color")
    target_color = target_color.lower() if isinstance(target_color, str) else None
    if target_color in {"", "none", "null"}:
        target_color = None
    if target_color is not None and target_color not in DESTINATION_SIGN_RULES:
        target_color = None

    target_letter = data.get("target_letter")
    target_letter = target_letter.upper() if isinstance(target_letter, str) else None
    if target_letter in {"", "NONE", "NULL"}:
        target_letter = None
    if target_letter is not None and target_letter not in {"A", "B", "C", "D", "E"}:
        target_letter = None

    held_color = observation.held_color
    if held_color is None and action == "deliver_held_cube":
        print("LLM 판단 보정: 큐브를 들고 있지 않아 deliver 대신 pick_source를 실행합니다.")
        action = "pick_source"
        target_color = None
        target_letter = "A"
    elif held_color in DESTINATION_SIGN_RULES and action == "pick_source":
        print("LLM 판단 보정: 큐브를 들고 있어 pick_source 대신 deliver_held_cube를 실행합니다.")
        action = "deliver_held_cube"
        target_color = held_color
        target_letter = DESTINATION_SIGN_RULES[held_color]
    elif held_color in DESTINATION_SIGN_RULES and action == "deliver_held_cube":
        target_color = held_color
        target_letter = DESTINATION_SIGN_RULES[held_color]
    elif held_color is None and action == "pick_source":
        target_letter = "A"

    return AgentDecision(
        action=action,
        target_color=target_color,
        target_letter=target_letter,
        reason=str(data.get("reason", "")).strip(),
    )


def deterministic_decision(observation: Observation) -> AgentDecision:
    if observation.held_color in DESTINATION_SIGN_RULES:
        color = observation.held_color
        return AgentDecision(
            action="deliver_held_cube",
            target_color=color,
            target_letter=DESTINATION_SIGN_RULES[color],
            reason="fallback: holding cube, deliver to matching destination",
        )
    return AgentDecision(
        action="pick_source",
        target_letter="A",
        reason="fallback: not holding cube, pick from source",
    )


async def ask_llm_agent_decision(ctx: Any, observation: Observation, memory: AgentMemory) -> AgentDecision:
    state = build_llm_state(observation, memory)
    api_key = getattr(ctx.config, "tokamak_api_key", "")
    if not api_key:
        print("LLM 판단 생략: TOKAMAK_API_KEY가 없어 규칙 기반 판단으로 대체합니다.")
        return deterministic_decision(observation)

    prompt = build_llm_decision_prompt(state)
    messages = [{"role": "user", "content": prompt}]
    attempt = 0
    while True:
        attempt += 1
        model = LLM_DECISION_MODELS[(attempt - 1) % len(LLM_DECISION_MODELS)]
        print(f"LLM 판단 요청: 현재 상태 텍스트만 전송합니다. attempt={attempt}, model={model}")
        try:
            reply = call_llm(messages, api_key=api_key, model=model, timeout_s=45)
        except Exception as exc:
            print(f"LLM 판단 오류: 요청 실패. 모델을 바꿔 재시도합니다. 원인={exc}")
            await asyncio.sleep(0.4)
            continue
        print(f"LLM 판단 응답 앞부분(model={model}): {reply[:180]}")
        if is_provider_fallback(reply):
            print("LLM 판단 재시도: provider fallback 응답입니다. 모델을 바꿔 다시 요청합니다.")
            await asyncio.sleep(0.4)
            continue
        decision = parse_agent_decision(reply, observation)
        if decision is None:
            print("LLM 판단 재시도: JSON 형식 또는 action이 유효하지 않습니다.")
            await asyncio.sleep(0.4)
            continue
        print(
            f"LLM 판단 승인: action={decision.action}, target_color={decision.target_color}, "
            f"target_letter={decision.target_letter}, reason={decision.reason}"
        )
        return decision


def build_stuck_vlm_prompt(
    *,
    target_xy: tuple[float, float],
    current_xy_value: tuple[float, float],
    stuck_seconds: float,
) -> str:
    return (
        "The robot was executing go_to toward a world coordinate but its world position barely changed, "
        f"so it was cancelled after {stuck_seconds:.1f}s. "
        f"Current xy is ({current_xy_value[0]:+.2f}, {current_xy_value[1]:+.2f}); "
        f"cancelled target xy was ({target_xy[0]:+.2f}, {target_xy[1]:+.2f}). "
        "Use the current front camera image to recommend exactly one short recovery action. "
        "If an obstacle is close in front, prefer backing up or backward turning. "
        "Return JSON only with this schema: "
        '{"action":"back_up|turn_left|turn_right|forward|retry_go_to|stop_retry_next_cycle",'
        '"duration_s":1.0,"reason":"short reason"}'
    )


async def ask_stuck_vlm_advice(
    ctx: Any,
    *,
    target_xy: tuple[float, float],
    current_xy_value: tuple[float, float],
) -> StuckMoveAdvice:
    api_key = getattr(ctx.config, "tokamak_api_key", "")
    if not api_key:
        print("정체 복구 AI 생략: TOKAMAK_API_KEY가 없어 다음 cycle 재탐색으로 넘깁니다.")
        return StuckMoveAdvice("stop_retry_next_cycle", 0.0, "TOKAMAK_API_KEY missing")

    jpeg = await get_camera_frame(ctx)
    compressed = compress_jpeg_for_vlm(jpeg, label="go_to stuck recovery")
    prompt = build_stuck_vlm_prompt(
        target_xy=target_xy,
        current_xy_value=current_xy_value,
        stuck_seconds=GOTO_STUCK_SECONDS,
    )

    attempt = 0
    while True:
        attempt += 1
        model = STUCK_VLM_MODELS[(attempt - 1) % len(STUCK_VLM_MODELS)]
        print(f"정체 복구 AI 요청: attempt={attempt}, model={model}")
        try:
            reply = ask_vlm(compressed, prompt, api_key=api_key, model=model)
        except Exception as exc:
            print(f"정체 복구 AI 오류: 요청 실패. 모델을 바꿔 재시도합니다. 원인={exc}")
            await asyncio.sleep(0.4)
            continue
        print(f"정체 복구 AI 응답 앞부분(model={model}): {reply[:180]}")
        if is_provider_fallback(reply):
            print("정체 복구 AI 재시도: provider fallback 응답입니다. 모델을 바꿔 다시 요청합니다.")
            await asyncio.sleep(0.4)
            continue
        advice = parse_stuck_move_advice(reply)
        if advice is None:
            print("정체 복구 AI 재시도: JSON 형식 또는 action이 유효하지 않습니다.")
            await asyncio.sleep(0.4)
            continue
        print(
            f"정체 복구 AI 승인: action={advice.action}, duration={advice.duration_s:.2f}s, "
            f"reason={advice.reason}"
        )
        return advice


async def execute_stuck_vlm_advice(ctx: Any, advice: StuckMoveAdvice) -> Any:
    if advice.action == "back_up":
        print(f"정체 복구 실행: 후진 | reason={advice.reason}")
        return await move_velocity(ctx, vx=-0.35, duration_s=max(advice.duration_s, 0.8))
    if advice.action == "turn_left":
        print(f"정체 복구 실행: 뒤로 가며 왼쪽 회전 | reason={advice.reason}")
        return await move_velocity(ctx, vx=-0.20, wz=0.50, duration_s=max(advice.duration_s, 1.0))
    if advice.action == "turn_right":
        print(f"정체 복구 실행: 뒤로 가며 오른쪽 회전 | reason={advice.reason}")
        return await move_velocity(ctx, vx=-0.20, wz=-0.50, duration_s=max(advice.duration_s, 1.0))
    if advice.action == "forward":
        print(f"정체 복구 실행: 짧게 전진 | reason={advice.reason}")
        return await move_velocity(ctx, vx=0.30, duration_s=max(advice.duration_s, 0.5))
    if advice.action == "retry_go_to":
        print(f"정체 복구 실행: 추가 primitive 없이 다음 로직에서 go_to를 재시도합니다. reason={advice.reason}")
        return {"status": "retry_go_to", "error": None}
    print(f"정체 복구 실행: 이동 없이 다음 cycle 재탐색으로 넘깁니다. reason={advice.reason}")
    return {"status": "stop_retry_next_cycle", "error": None}


async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    print(f"좌표 이동: 추정 world target=({x:+.2f}, {y:+.2f})")
    task = asyncio.create_task(
        ctx.invoke(
            "go_to",
            {
                "target": {
                    "kind": "pose",
                    "pose": {"frame_id": "world", "position": [x, y, 0]},
                }
            },
            timeout_s=300,
        )
    )
    last_xy = await current_xy(ctx)
    stuck_started_at: float | None = None

    while not task.done():
        await asyncio.sleep(0.5)
        try:
            xy = await current_xy(ctx)
        except Exception as exc:
            print(f"좌표 이동 감시 경고: robot_status 조회 실패. 감시는 계속합니다. 원인={exc}")
            continue

        moved = math.dist(last_xy, xy)
        if moved >= GOTO_STUCK_MOVE_EPS_M:
            last_xy = xy
            stuck_started_at = None
            continue

        now = asyncio.get_running_loop().time()
        if stuck_started_at is None:
            stuck_started_at = now
            continue
        if now - stuck_started_at >= GOTO_STUCK_SECONDS:
            print(
                f"좌표 이동 정체 감지: {GOTO_STUCK_SECONDS:.1f}초 이상 위치 변화가 작습니다. "
                f"현재=({xy[0]:+.2f},{xy[1]:+.2f}), target=({x:+.2f},{y:+.2f})"
            )
            try:
                await cancel_action(ctx)
            except Exception as exc:
                print(f"cancel 경고: 실패했지만 go_to task를 중단하고 AI 복구를 계속합니다. 원인={exc}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"go_to 종료 경고: cancel 이후 응답 오류를 무시합니다. 원인={exc}")
            advice = await ask_stuck_vlm_advice(ctx, target_xy=(x, y), current_xy_value=xy)
            advice_result = await execute_stuck_vlm_advice(ctx, advice)
            return {
                "status": "stuck_vlm_recovered",
                "error": "go_to_stuck_no_position_change",
                "target_xy": (x, y),
                "advice": {
                    "action": advice.action,
                    "duration_s": advice.duration_s,
                    "reason": advice.reason,
                },
                "advice_result": result_summary(advice_result),
            }

    result = await task
    return result


async def pick_nearest_cube(ctx: Any) -> Any:
    print("집기 실행: nearest cube pick_entity 호출")
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    print("놓기 실행: nearest zone place_entity 호출")
    return await ctx.invoke("place_entity", {}, timeout_s=300)


async def attempt_place(
    ctx: Any,
    tracker: CompletionTracker | None,
    *,
    delivered_before: int,
    label: str,
) -> PlaceAttempt:
    print(label)
    result = await place_nearest_zone(ctx)
    await asyncio.sleep(0.5)
    held_after = await get_held_cube_color(ctx)
    delivered_after = await get_delivered_count(ctx, tracker)
    released = held_after is None
    scored = delivered_after > delivered_before
    placed = released and scored
    return PlaceAttempt(
        result=result,
        held_after=held_after,
        delivered_after=delivered_after,
        released=released,
        scored=scored,
        placed=placed,
    )


def result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


def result_ok(result: Any) -> bool:
    status = str(getattr(result, "status", "")).lower()
    return status in {"done", "success", "succeeded", "completed", "ok"}


def result_status(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("status", ""))
    return str(getattr(result, "status", ""))


def is_stuck_recovered(result: Any) -> bool:
    return result_status(result) in {"stuck_recovered", "stuck_vlm_recovered"}


def robot_xy_yaw(robot_status: Any) -> tuple[float, float, float]:
    pose = robot_status.robot.pose
    return float(pose.position[0]), float(pose.position[1]), float(pose.yaw_deg)


async def current_xy(ctx: Any) -> tuple[float, float]:
    x, y, _yaw = robot_xy_yaw(await get_robot_status(ctx))
    return x, y


def yaw_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def pose_close(
    current_pose: tuple[float, float, float],
    cached_pose: tuple[float, float, float],
    *,
    max_xy_delta_m: float = LETTER_SCAN_CACHE_MAX_XY_DELTA_M,
    max_yaw_delta_deg: float = LETTER_SCAN_CACHE_MAX_YAW_DELTA_DEG,
) -> bool:
    dx = current_pose[0] - cached_pose[0]
    dy = current_pose[1] - cached_pose[1]
    return math.hypot(dx, dy) <= max_xy_delta_m and yaw_delta_deg(current_pose[2], cached_pose[2]) <= max_yaw_delta_deg


def body_bearing_from_image(*, image_angle_deg: float, head_yaw_rad: float) -> float:
    # Image angle is screen-space and tends to oversteer in the world projection.
    # Keep the previous sign convention, but damp the image contribution.
    return round(math.degrees(head_yaw_rad) - image_angle_deg * IMAGE_BEARING_GAIN, 1)


def pad_bearing_from_detection(detection: LetterDetection) -> float:
    # Destination signs are far from the camera and sit near the image edges.
    # A larger gain matches the world-space bearing observed in the top-down viewer.
    return round(math.degrees(detection.frame.yaw) - detection.angle_deg * PAD_IMAGE_BEARING_GAIN, 1)


def pad_scales_for_distance(distance_m: float) -> tuple[float, float, str]:
    if distance_m <= PAD_CLOSE_DISTANCE_M:
        return PAD_CLOSE_TRAVEL_SCALE, PAD_CLOSE_ALIGN_SCALE, "close"
    return PAD_TRAVEL_SCALE, PAD_ALIGN_SCALE, "normal"


def xy_from_bearing(
    robot_status: Any,
    *,
    bearing_deg: float,
    distance_m: float,
    standoff_m: float = 0.35,
    travel_scale: float = 1.0,
) -> tuple[float, float]:
    rx, ry, yaw_deg = robot_xy_yaw(robot_status)
    travel_m = max(0.35, (distance_m - standoff_m) * travel_scale)
    theta = math.radians(yaw_deg + bearing_deg)
    x = rx + math.cos(theta) * travel_m
    y = ry + math.sin(theta) * travel_m
    print(
        f"좌표 추정식: robot=({rx:+.2f},{ry:+.2f}), yaw={yaw_deg:+.1f}도, "
        f"body_bearing={bearing_deg:+.1f}도, distance={distance_m:.2f}m, "
        f"standoff={standoff_m:.2f}m, scale={travel_scale:.2f} -> target=({x:+.2f},{y:+.2f})"
    )
    return x, y


def estimate_cube_distance(det: ScannedColor) -> float:
    return max(0.45, min(4.0, CUBE_DISTANCE_K / math.sqrt(max(det.blob_area, 1))))


def estimate_sign_distance(det: LetterDetection) -> float:
    _x, _y, width, height = det.bbox
    size = max(width, height, 1)
    return max(0.7, min(5.0, SIGN_DISTANCE_K / size))


def estimate_pad_sign_distance(det: LetterDetection) -> float:
    _x, _y, width, height = det.bbox
    size = max(width, height, 1)
    return max(0.8, min(6.0, PAD_SIGN_DISTANCE_K / size))


def _cv2_np() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    return cv2, np


@lru_cache(maxsize=16)
def render_letter_template(letter: str, size: int = LETTER_TEMPLATE_SIZE) -> Any:
    cv2, np = _cv2_np()
    image = np.zeros((size, size), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.35
    thickness = 5
    (text_width, text_height), _baseline = cv2.getTextSize(letter, font, scale, thickness)
    x = max(0, (size - text_width) // 2)
    y = max(text_height, (size + text_height) // 2)
    cv2.putText(image, letter, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    _ok, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    return image


def normalize_letter_crop(mask_crop: Any, size: int = LETTER_TEMPLATE_SIZE) -> Any:
    cv2, np = _cv2_np()
    height, width = mask_crop.shape[:2]
    output = np.zeros((size, size), dtype=np.uint8)
    if height <= 0 or width <= 0:
        return output
    scale = min((size - 18) / max(width, 1), (size - 18) / max(height, 1))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(mask_crop, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    x = (size - resized_width) // 2
    y = (size - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def letter_template_score(mask_crop: Any, target_letter: str) -> float:
    cv2, _np = _cv2_np()
    normalized = normalize_letter_crop(mask_crop)
    template = render_letter_template(target_letter)
    return float(cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0, 0])


def letter_template_scores(mask_crop: Any) -> dict[str, float]:
    return {letter: letter_template_score(mask_crop, letter) for letter in "ABCDE"}


def sign_background_color_counts(image: Any, bbox: tuple[int, int, int, int]) -> dict[str, int]:
    cv2, np = _cv2_np()
    height, width = image.shape[:2]
    x, y, bbox_width, bbox_height = bbox
    # Keep the ROI close to the white letter. A wide ROI pulls in sky/conveyor colors
    # around small signs and makes green A/C signs look blue.
    x_pad = max(8, int(bbox_width * 0.8))
    y_pad = max(8, int(bbox_height * 0.55))
    roi = image[
        max(0, y - y_pad) : min(height, y + bbox_height + y_pad),
        max(0, x - x_pad) : min(width, x + bbox_width + x_pad),
    ]
    if roi.size == 0:
        return {"red": 0, "green": 0, "blue": 0, "yellow": 0}
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 75, 70]), np.array([10, 255, 255]))
    red_mask += cv2.inRange(hsv, np.array([170, 75, 70]), np.array([180, 255, 255]))
    return {
        "red": int(cv2.countNonZero(red_mask)),
        "green": int(cv2.countNonZero(cv2.inRange(hsv, np.array([38, 65, 65]), np.array([92, 255, 255])))),
        "blue": int(cv2.countNonZero(cv2.inRange(hsv, np.array([88, 65, 45]), np.array([138, 255, 255])))),
        "yellow": int(cv2.countNonZero(cv2.inRange(hsv, np.array([17, 65, 70]), np.array([42, 255, 255])))),
    }


def color_mask_for_name(hsv: Any, color: str) -> Any:
    cv2, np = _cv2_np()
    if color == "red":
        mask = cv2.inRange(hsv, np.array([0, 75, 70]), np.array([10, 255, 255]))
        mask += cv2.inRange(hsv, np.array([170, 75, 70]), np.array([180, 255, 255]))
        return mask
    if color == "green":
        return cv2.inRange(hsv, np.array([38, 65, 65]), np.array([92, 255, 255]))
    if color == "blue":
        return cv2.inRange(hsv, np.array([88, 65, 45]), np.array([138, 255, 255]))
    if color == "yellow":
        return cv2.inRange(hsv, np.array([17, 65, 70]), np.array([42, 255, 255]))
    return np.zeros(hsv.shape[:2], dtype=np.uint8)


def detect_letter_by_colored_background(frame: ScanFrame, target_letter: str, image: Any) -> LetterDetection | None:
    """Fallback for small signs where the white letter contour is fragmented."""
    cv2, np = _cv2_np()
    if target_letter == "A":
        return None
    expected_color = SIGN_BACKGROUND_COLORS.get(target_letter)
    if expected_color is None:
        return None

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = color_mask_for_name(hsv, expected_color)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: LetterDetection | None = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (240 <= area <= 45000):
            continue
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        aspect = bbox_width / max(bbox_height, 1)
        if not (14 <= bbox_width <= 240 and 14 <= bbox_height <= 240 and 0.45 <= aspect <= 1.85):
            continue

        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + bbox_width + pad)
        y1 = min(height, y + bbox_height + pad)
        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(roi_hsv, np.array([0, 0, 160]), np.array([180, 105, 255]))
        white_pixels = cv2.countNonZero(white_mask)
        if white_pixels < 35:
            continue

        scores = letter_template_scores(white_mask)
        score = scores[target_letter]
        best_letter, best_letter_score = max(scores.items(), key=lambda item: item[1])
        if best_letter != target_letter and best_letter_score > score + 0.08:
            continue
        if score < 0.36:
            continue

        cx = x + bbox_width / 2
        image_angle = (cx - width / 2) / (width / 2) * 30.0
        detection = LetterDetection(
            letter=target_letter,
            score=max(score, 0.43),
            angle_deg=round(image_angle, 1),
            full_bearing_deg=body_bearing_from_image(image_angle_deg=image_angle, head_yaw_rad=frame.yaw),
            bbox=(x, y, bbox_width, bbox_height),
            frame=frame,
        )
        print(
            f"OpenCV 배경색 fallback 승인: target={target_letter}, bbox={(x, y, bbox_width, bbox_height)}, "
            f"score={score:.2f}, best_letter={best_letter}:{best_letter_score:.2f}, "
            f"color={expected_color}, image={frame.path}"
        )
        if best is None or detection.score > best.score or detection.area > best.area:
            best = detection
    return best


def detect_letter(frame: ScanFrame, target_letter: str) -> LetterDetection | None:
    cv2, np = _cv2_np()
    try:
        image = decode_jpeg(frame.jpeg)
    except Exception:
        return None
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 90, 255]))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    expected_color = SIGN_BACKGROUND_COLORS.get(target_letter)
    best: LetterDetection | None = None
    for contour in contours:
        if cv2.contourArea(contour) < 45:
            continue
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        aspect = bbox_width / max(bbox_height, 1)
        if not (8 <= bbox_width <= 280 and 12 <= bbox_height <= 300 and 0.12 <= aspect <= 1.65):
            continue
        crop = white_mask[y : y + bbox_height, x : x + bbox_width]
        scores = letter_template_scores(crop)
        score = scores[target_letter]
        if score < LETTER_MIN_SCORE:
            continue
        counts = sign_background_color_counts(image, (x, y, bbox_width, bbox_height))
        best_color, best_count = max(counts.items(), key=lambda item: item[1])
        expected_count = counts.get(expected_color or "", 0)
        best_letter, best_letter_score = max(scores.items(), key=lambda item: item[1])
        best_letter_color = SIGN_BACKGROUND_COLORS.get(best_letter)
        if (
            best_letter != target_letter
            and best_letter_color == expected_color
            and best_letter_score > score + 0.035
        ):
            print(
                f"OpenCV 후보 제외: target={target_letter}, bbox={(x, y, bbox_width, bbox_height)}는 "
                f"같은 배경색 후보 {best_letter} 점수가 더 높습니다. "
                f"scores={{{', '.join(f'{k}:{v:.2f}' for k, v in scores.items())}}}, counts={counts}"
            )
            continue
        if (
            best_letter != target_letter
            and best_letter_color is not None
            and best_letter_color == best_color
            and best_letter_score > score - 0.02
        ):
            print(
                f"OpenCV 후보 제외: target={target_letter}, bbox={(x, y, bbox_width, bbox_height)}는 "
                f"배경색 {best_color}와 {best_letter} 후보가 더 일치합니다."
            )
            continue
        if expected_color and (expected_count < 80 or expected_count < best_count * 0.70):
            print(
                f"OpenCV 후보 제외: target={target_letter}, bbox={(x, y, bbox_width, bbox_height)} "
                f"배경색 불일치 expected={expected_color}, counts={counts}"
            )
            continue
        cx = x + bbox_width / 2
        angle = (cx - width / 2) / (width / 2) * 30.0
        detection = LetterDetection(
            letter=target_letter,
            score=score,
            angle_deg=round(angle, 1),
            full_bearing_deg=body_bearing_from_image(image_angle_deg=angle, head_yaw_rad=frame.yaw),
            bbox=(x, y, bbox_width, bbox_height),
            frame=frame,
        )
        print(
            f"OpenCV 후보 승인: target={target_letter}, bbox={(x, y, bbox_width, bbox_height)}, "
            f"score={score:.2f}, best_letter={best_letter}:{best_letter_score:.2f}, "
            f"best_color={best_color}, counts={counts}, image={frame.path}"
        )
        if best is None or detection.score > best.score:
            best = detection
    fallback = detect_letter_by_colored_background(frame, target_letter, image)
    if fallback is not None and (best is None or fallback.score >= best.score - 0.08):
        return fallback
    return best


async def capture_frame(ctx: Any, *, index: int, yaw: float, cycle: int, label: str) -> ScanFrame:
    await set_head(ctx, yaw=yaw, pitch=SCAN_PITCH)
    await asyncio.sleep(HEAD_SETTLE_S)
    robot_pose = robot_xy_yaw(await get_robot_status(ctx))
    jpeg = await get_camera_frame(ctx)
    path = None
    if SAVE_DEBUG_IMAGES:
        save_dir = DEBUG_DIR / f"cycle_{cycle:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        yaw_name = f"yaw_{yaw:+.2f}".replace("+", "p").replace("-", "m")
        path = save_dir / f"{label}_frame_{index}_{yaw_name}.jpg"
        path.write_bytes(jpeg)
        print(f"사진 저장: {path}")
    return ScanFrame(
        index=index,
        yaw=yaw,
        pitch=SCAN_PITCH,
        jpeg=jpeg,
        path=path,
        robot_x=robot_pose[0],
        robot_y=robot_pose[1],
        robot_yaw_deg=robot_pose[2],
    )


async def scan_colors(ctx: Any, *, cycle: int) -> list[ScannedColor]:
    found: list[ScannedColor] = []
    for index, yaw in enumerate(SCAN_YAWS, start=1):
        frame = await capture_frame(ctx, index=index, yaw=yaw, cycle=cycle, label="color_scan")
        detections = detect_color_blobs(frame.jpeg, min_area=250)
        preview = ", ".join(
            f"{item.color}:{item.blob_area}@{item.angle_deg:+.1f}"
            for item in detections[:5]
        )
        print(f"색상 스캔 요약: frame={index}, yaw={yaw:+.2f}, count={len(detections)}, top=[{preview}]")
        for detection in detections:
            found.append(
                ScannedColor(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    full_bearing_deg=body_bearing_from_image(
                        image_angle_deg=detection.angle_deg,
                        head_yaw_rad=yaw,
                    ),
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    frame=frame,
                )
            )
    await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
    return sorted(found, key=lambda item: item.blob_area, reverse=True)


def cached_letter_detection(
    memory: AgentMemory | None,
    target_letter: str,
    current_pose: tuple[float, float, float],
) -> LetterDetection | None:
    if memory is None:
        return None
    cached = memory.letter_scan_cache.get(target_letter)
    if cached is None:
        return None
    cached_pose = (cached.frame.robot_x, cached.frame.robot_y, cached.frame.robot_yaw_deg)
    if not pose_close(current_pose, cached_pose):
        return None
    dx = current_pose[0] - cached_pose[0]
    dy = current_pose[1] - cached_pose[1]
    print(
        f"표지판 캐시 사용: target={target_letter}, image={cached.frame.path}, "
        f"pose_delta=({dx:+.2f}m,{dy:+.2f}m,{yaw_delta_deg(current_pose[2], cached_pose[2]):.1f}도), "
        f"bearing={cached.full_bearing_deg:+.1f}도, bbox={cached.bbox}"
    )
    return cached


def remember_letter_detection(memory: AgentMemory | None, detection: LetterDetection) -> None:
    if memory is None:
        return
    memory.letter_scan_cache[detection.letter] = detection


def remember_letter_frames(memory: AgentMemory | None, target_letter: str, frames: list[ScanFrame]) -> None:
    if memory is None or not frames:
        return
    memory.letter_frame_cache[target_letter] = frames


def cached_letter_from_frames(
    memory: AgentMemory | None,
    target_letter: str,
    current_pose: tuple[float, float, float],
) -> tuple[bool, LetterDetection | None]:
    if memory is None:
        return False, None
    frames = memory.letter_frame_cache.get(target_letter)
    if not frames:
        return False, None
    close_frames = [
        frame
        for frame in frames
        if pose_close(current_pose, (frame.robot_x, frame.robot_y, frame.robot_yaw_deg))
    ]
    if not close_frames:
        return False, None
    print(
        f"표지판 사진 캐시 사용: target={target_letter}, frames={len(close_frames)}개, "
        "현재 pose 변화가 작아 새 두리번 탐색을 생략합니다."
    )
    best: LetterDetection | None = None
    for frame in close_frames:
        detection = detect_letter(frame, target_letter)
        if detection is not None and (best is None or detection.score > best.score):
            best = detection
    if best is None:
        print(f"표지판 사진 캐시 결과: target={target_letter} 미검출. 새 촬영 없이 실패로 처리합니다.")
        return True, None
    print(
        f"표지판 사진 캐시 결과: {target_letter} 발견, bearing={best.full_bearing_deg:+.1f}도, "
        f"bbox={best.bbox}, score={best.score:.2f}, image={best.frame.path}"
    )
    remember_letter_detection(memory, best)
    return True, best


async def scan_for_letter(
    ctx: Any,
    target_letter: str,
    *,
    cycle: int,
    max_body_turns: int = 3,
    memory: AgentMemory | None = None,
) -> LetterDetection | None:
    current_pose = robot_xy_yaw(await get_robot_status(ctx))
    cached = cached_letter_detection(memory, target_letter, current_pose)
    if cached is not None:
        return cached
    used_frame_cache, cached_from_frames = cached_letter_from_frames(memory, target_letter, current_pose)
    if used_frame_cache:
        return cached_from_frames

    print(f"표지판 탐색: target={target_letter}")
    captured_frames: list[ScanFrame] = []
    for body_turn in range(max_body_turns):
        print(f"표지판 탐색 단계: target={target_letter}, body_scan={body_turn + 1}/{max_body_turns}")
        for index, yaw in enumerate(SCAN_YAWS, start=1):
            frame = await capture_frame(ctx, index=index, yaw=yaw, cycle=cycle, label=f"sign_{target_letter}_scan_{body_turn + 1}")
            captured_frames.append(frame)
            detection = detect_letter(frame, target_letter)
            if detection is not None:
                print(
                    f"표지판 발견: {target_letter}, bearing={detection.full_bearing_deg:+.1f}도, "
                    f"bbox={detection.bbox}, score={detection.score:.2f}, image={detection.frame.path}"
                )
                remember_letter_detection(memory, detection)
                remember_letter_frames(memory, target_letter, captured_frames)
                await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
                return detection
        if body_turn < max_body_turns - 1:
            await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
            print(f"OpenCV 탐색 실패: target={target_letter}, {SEARCH_TURN_DEGREES}도 회전 후 다시 확인합니다.")
            await turn_scan(ctx, direction="left", degrees=SEARCH_TURN_DEGREES)
    await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
    remember_letter_frames(memory, target_letter, captured_frames)
    print(f"표지판 탐색 실패: {target_letter}")
    return None


async def scan_front_for_letter(ctx: Any, target_letter: str, *, cycle: int, label: str) -> LetterDetection | None:
    frame = await capture_frame(ctx, index=1, yaw=0.0, cycle=cycle, label=label)
    detection = detect_letter(frame, target_letter)
    if detection is not None:
        print(
            f"정면 재검출 성공: {target_letter}, bearing={detection.full_bearing_deg:+.1f}도, "
            f"bbox={detection.bbox}, score={detection.score:.2f}, image={detection.frame.path}"
        )
    else:
        print(f"정면 재검출 실패: target={target_letter}, image={frame.path}")
    await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
    return detection


async def scan_for_letter_front_aligned(
    ctx: Any,
    target_letter: str,
    *,
    cycle: int,
    max_body_turns: int = 3,
) -> LetterDetection | None:
    """Find a sign, but only return detections captured with the head facing front."""
    print(f"정면 기준 표지판 탐색: target={target_letter}")
    for body_turn in range(max_body_turns):
        print(f"정면 기준 표지판 탐색 단계: target={target_letter}, body_scan={body_turn + 1}/{max_body_turns}")
        front = await scan_front_for_letter(
            ctx,
            target_letter,
            cycle=cycle,
            label=f"sign_{target_letter}_front_scan_{body_turn + 1}",
        )
        if front is not None:
            return front

        side_detection: LetterDetection | None = None
        for index, yaw in enumerate(SCAN_YAWS[1:], start=2):
            frame = await capture_frame(
                ctx,
                index=index,
                yaw=yaw,
                cycle=cycle,
                label=f"sign_{target_letter}_side_probe_{body_turn + 1}",
            )
            detection = detect_letter(frame, target_letter)
            if detection is not None:
                side_detection = detection
                print(
                    f"측면 probe 발견: {target_letter}, head_yaw={yaw:+.2f}, "
                    f"bearing={detection.full_bearing_deg:+.1f}도, bbox={detection.bbox}, image={frame.path}"
                )
                break

        await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
        if side_detection is not None:
            direction = "left" if side_detection.frame.yaw < 0 else "right"
            wz = 0.5 if direction == "left" else -0.5
            print(
                f"정면 재획득 회전: {target_letter}가 {direction} 측면에 있으므로 "
                "좌표 계산 전에 몸을 돌려 정면으로 맞춥니다."
            )
            await move_velocity(ctx, vx=-0.2, wz=wz, duration_s=1.6)
            front = await scan_front_for_letter(
                ctx,
                target_letter,
                cycle=cycle,
                label=f"sign_{target_letter}_front_after_side_{body_turn + 1}",
            )
            if front is not None:
                return front

        if body_turn < max_body_turns - 1:
            print(f"정면 기준 탐색 실패: target={target_letter}, {SEARCH_TURN_DEGREES}도 회전 후 다시 확인합니다.")
            await turn_scan(ctx, direction="left", degrees=SEARCH_TURN_DEGREES)

    print(f"정면 기준 표지판 탐색 실패: {target_letter}")
    await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
    return None


async def recenter_side_head_detection(
    ctx: Any,
    target_letter: str,
    detection: LetterDetection,
    *,
    cycle: int,
) -> LetterDetection | None:
    head_yaw = detection.frame.yaw
    if abs(head_yaw) <= SIDE_HEAD_YAW_EPS:
        return detection

    # In this simulator yaw=-0.8 is the left-looking head frame and yaw=+0.8 is right.
    direction = "left" if head_yaw < 0 else "right"
    duration = min(2.2, max(1.0, abs(head_yaw) / 0.5))
    wz = 0.5 if direction == "left" else -0.5
    print(
        f"목적지 재정렬: {target_letter}를 머리 {direction} 프레임(yaw={head_yaw:+.2f})에서 발견했습니다. "
        "좌표 계산 전에 몸을 그 방향으로 돌리고 정면에서 다시 확인합니다."
    )
    await set_head(ctx, yaw=0.0, pitch=SCAN_PITCH)
    await move_velocity(ctx, vx=-0.2, wz=wz, duration_s=duration)
    return await scan_front_for_letter(
        ctx,
        target_letter,
        cycle=cycle,
        label=f"sign_{target_letter}_front_reacquire",
    )


async def observe_world(ctx: Any, memory: AgentMemory, tracker: CompletionTracker | None = None) -> Observation:
    robot_status = await get_robot_status(ctx)
    held_color = await get_held_cube_color(ctx)
    delivered_count = await get_delivered_count(ctx, tracker)
    if held_color is None and memory.last_pick_xy is not None:
        print(
            f"관찰 최적화: 저장된 집기 좌표가 있어 색상 스캔을 생략합니다. "
            f"pick_xy=({memory.last_pick_xy[0]:+.2f}, {memory.last_pick_xy[1]:+.2f})"
        )
        colors = []
    else:
        colors = await scan_colors(ctx, cycle=memory.cycle)
    memory.held_color = held_color
    memory.delivered_count = delivered_count
    memory.stage = "have_cube" if held_color else "need_cube"
    return Observation(
        robot_status=robot_status,
        held_color=held_color,
        delivered_count=delivered_count,
        colors=colors,
    )


def choose_cube_detection(colors: list[ScannedColor]) -> ScannedColor | None:
    cube_colors = [item for item in colors if item.color in DESTINATION_SIGN_RULES]
    if not cube_colors:
        return None
    # 큐브가 가까울수록 bbox가 화면 아래쪽에 있고 area가 큽니다.
    return max(cube_colors, key=lambda item: item.blob_area + item.bbox[1] * 12)


async def pick_here_and_record(
    ctx: Any,
    memory: AgentMemory,
    *,
    label: str,
) -> tuple[bool, str | None, Any]:
    result = await pick_nearest_cube(ctx)
    await asyncio.sleep(0.5)
    held = await get_held_cube_color(ctx)
    if held:
        xy = await current_xy(ctx)
        memory.last_pick_xy = xy
        memory.active_color = held
        print(
            f"{label}: 집기 성공 held_color={held}, pick_xy=({xy[0]:+.2f}, {xy[1]:+.2f}) 저장, "
            "다음 배송 후 재집기 좌표로 사용합니다."
        )
        return True, held, result
    print(f"{label}: 집기 실패 또는 미확인 result={result_summary(result)}")
    return False, None, result


async def pick_here_without_record(
    ctx: Any,
    memory: AgentMemory,
    *,
    label: str,
) -> tuple[bool, str | None, Any]:
    result = await pick_nearest_cube(ctx)
    await asyncio.sleep(0.5)
    held = await get_held_cube_color(ctx)
    if held:
        memory.active_color = held
        print(f"{label}: 집기 성공 held_color={held}. 이 특수 집기 좌표는 저장하지 않습니다.")
        return True, held, result
    print(f"{label}: 집기 실패 또는 미확인 result={result_summary(result)}")
    return False, None, result


async def pick_from_saved_location(
    ctx: Any,
    memory: AgentMemory,
    *,
    label: str,
) -> tuple[bool, str | None, Any | None]:
    if memory.last_pick_xy is None:
        print(f"{label}: 저장된 집기 좌표가 없습니다.")
        return False, None, None
    pick_xy = memory.last_pick_xy
    print(
        f"{label}: 저장된 집기 좌표로 바로 이동합니다. "
        f"xy=({pick_xy[0]:+.2f}, {pick_xy[1]:+.2f})"
    )
    nav = await go_to_xy(ctx, *pick_xy)
    print(f"{label}: 저장 집기 좌표 이동 결과: {result_summary(nav)}")
    if is_stuck_recovered(nav):
        print(f"{label}: 이동 중 정체 복구가 발생해서 집기를 생략합니다.")
        memory.last_pick_xy = None
        return False, None, nav
    picked, held, result = await pick_here_and_record(ctx, memory, label=label)
    if not picked:
        print(f"{label}: 저장된 집기 좌표에서 실패했으므로 좌표를 버립니다.")
        memory.last_pick_xy = None
    return picked, held, result


async def finish_place_attempt(
    ctx: Any,
    memory: AgentMemory,
    tracker: CompletionTracker | None,
    *,
    attempt: PlaceAttempt,
    target_letter: str,
    target_xy: tuple[float, float],
    held_color: str,
    used_saved_xy: bool,
    allow_post_place_pick: bool,
) -> dict[str, Any]:
    if attempt.placed:
        memory.pad_estimates[target_letter] = target_xy
        print(f"목적지 좌표 저장: sign={target_letter}, xy=({target_xy[0]:+.2f}, {target_xy[1]:+.2f})")
    elif not used_saved_xy:
        memory.pad_estimates.pop(target_letter, None)
    if attempt.placed and held_color not in memory.completed_colors:
        memory.completed_colors.append(held_color)
    if not attempt.placed:
        memory.pad_estimates.pop(target_letter, None)
        print(
            f"놓기 실패 보정: sign={target_letter} 추정 좌표를 버리고 다음 cycle에서 재탐색합니다. "
            f"released={attempt.released}, scored={attempt.scored}"
        )
    action_result = {
        "action": "place_cube",
        "placed": attempt.placed,
        "released": attempt.released,
        "scored": attempt.scored,
        "held_color_before": held_color,
        "delivered_after": attempt.delivered_after,
        "target_xy": target_xy,
        "result": result_summary(attempt.result),
    }
    print(
        f"놓기 확인: placed={attempt.placed}, released={attempt.released}, "
        f"scored={attempt.scored}, delivered={attempt.delivered_after}, held_after={attempt.held_after}"
    )

    if allow_post_place_pick and attempt.placed:
        if target_letter == "C":
            print("C 배송 후 즉시 집기 시도: C 앞 컨베이어 근처 큐브를 바로 집어봅니다.")
            picked, picked_color, post_pick = await pick_here_without_record(ctx, memory, label="C 배송 직후")
            action_result["post_c_pick_result"] = result_summary(post_pick)
            action_result["post_c_picked_color"] = picked_color
            if picked and picked_color:
                print(
                    f"C 배송 직후 재집기 성공: held_color={picked_color}. "
                    "다음 cycle에서 LLM에게 다음 목적지를 다시 묻습니다."
                )
            else:
                print("C 배송 직후 재집기 실패: 다음 cycle에서 LLM에게 다음 행동을 묻습니다.")
        else:
            print(f"{target_letter} 배송 성공: 다음 cycle에서 LLM에게 다음 행동을 다시 묻습니다.")
    return action_result


async def approach_and_pick(ctx: Any, observation: Observation, memory: AgentMemory) -> dict[str, Any]:
    print("큐브 단계: 색상 blob 좌표 대신 source sign 'A'를 기준으로 접근합니다.")
    memory.source_estimate = None
    if memory.last_pick_xy is not None:
        picked, held, result = await pick_from_saved_location(
            ctx,
            memory,
            label="이전 집기 좌표",
        )
        if picked:
            return {"action": "pick_cube", "picked": True, "held_color": held, "source_xy": memory.last_pick_xy}
        print("이전 집기 좌표에서 집기 실패: 메모리를 버리고 A/source를 다시 탐색합니다.")

    target_xy: tuple[float, float] | None = None
    detection = await scan_for_letter(ctx, "A", cycle=memory.cycle, memory=memory)
    if detection is None:
        print("source 탐색 실패: A 표지판을 못 찾아 복구 회전합니다.")
        return {"action": "search_source", "status": "failed", "reason": "A 미검출"}

    if detection.area >= 10000 and abs(detection.full_bearing_deg) <= 35:
        print(
            f"source 도착 판단: A가 충분히 큽니다. area={detection.area}, "
            f"bearing={detection.full_bearing_deg:+.1f}도. 바로 pick을 시도합니다."
        )
    else:
        current_status = await get_robot_status(ctx)
        distance = estimate_sign_distance(detection)
        target_xy = xy_from_bearing(
            current_status,
            bearing_deg=detection.full_bearing_deg,
            distance_m=distance,
            standoff_m=SOURCE_APPROACH_STANDOFF_M,
            travel_scale=SOURCE_TRAVEL_SCALE,
        )
        print(
            f"source 좌표 추정: sign=A, bearing={detection.full_bearing_deg:+.1f}도, "
            f"distance~{distance:.2f}m, xy=({target_xy[0]:+.2f}, {target_xy[1]:+.2f})"
        )

    if target_xy is not None:
        nav = await go_to_xy(ctx, *target_xy)
        print(f"source 접근 결과: {result_summary(nav)}")
        if is_stuck_recovered(nav):
            print("source 접근 정체 복구: 이번 cycle 집기는 생략하고 다음 cycle에서 A를 다시 탐색합니다.")
            return {
                "action": "pick_cube",
                "picked": False,
                "recover": "retry_source_without_turn",
                "result": result_summary(nav),
                "source_xy": target_xy,
            }
        await asyncio.sleep(0.5)

    picked, held, result = await pick_here_and_record(ctx, memory, label="source 위치")
    if picked:
        return {"action": "pick_cube", "picked": True, "held_color": held, "source_xy": target_xy}

    print("집기 1차 실패: source 방향으로 조금 더 전진 후 pick을 한 번 더 시도합니다.")
    await move_velocity(ctx, vx=0.35, duration_s=SOURCE_FORWARD_RETRY_S)
    await asyncio.sleep(0.3)
    picked, held, retry_result = await pick_here_and_record(ctx, memory, label="source 전진 보정")
    if picked:
        return {"action": "pick_cube", "picked": True, "held_color": held, "source_xy": target_xy}

    memory.source_estimate = None
    print("집기 실패 보정: source 좌표 메모리를 버리고 다음 cycle에서 A를 다시 탐색합니다.")
    print(f"집기 실패 또는 미확인: first={result_summary(result)}, retry={result_summary(retry_result)}")
    return {
        "action": "pick_cube",
        "picked": False,
        "recover": "retry_source_without_turn",
        "result": result_summary(retry_result),
        "source_xy": target_xy,
    }


async def navigate_and_place(
    ctx: Any,
    observation: Observation,
    memory: AgentMemory,
    tracker: CompletionTracker | None = None,
    *,
    allow_post_place_pick: bool = True,
    route_retry_count: int = 0,
) -> dict[str, Any]:
    held_color = observation.held_color or memory.held_color
    if held_color not in DESTINATION_SIGN_RULES:
        return {"action": "place_cube", "status": "failed", "reason": "held color 없음"}

    target_letter = DESTINATION_SIGN_RULES[held_color]
    align_xy: tuple[float, float] | None = None
    used_saved_xy = target_letter in memory.pad_estimates
    if used_saved_xy:
        target_xy = memory.pad_estimates[target_letter]
        print(f"목적지 메모리 사용: sign={target_letter}, xy=({target_xy[0]:+.2f}, {target_xy[1]:+.2f})")
    else:
        detection = await scan_for_letter(ctx, target_letter, cycle=memory.cycle, memory=memory)
        if detection is None:
            key = f"scan_{target_letter}"
            memory.failed_attempts[key] = memory.failed_attempts.get(key, 0) + 1
            return {"action": "search_pad", "status": "failed", "reason": f"{target_letter} 미검출"}

        current_status = await get_robot_status(ctx)
        distance = estimate_pad_sign_distance(detection)
        pad_bearing = pad_bearing_from_detection(detection)
        target_scale, align_scale, pad_distance_mode = pad_scales_for_distance(distance)
        target_xy = xy_from_bearing(
            current_status,
            bearing_deg=pad_bearing,
            distance_m=distance,
            standoff_m=PAD_APPROACH_STANDOFF_M,
            travel_scale=target_scale,
        )
        align_xy = xy_from_bearing(
            current_status,
            bearing_deg=pad_bearing,
            distance_m=distance,
            standoff_m=PAD_APPROACH_STANDOFF_M,
            travel_scale=align_scale,
        )
        print(
            f"목적지 좌표 추정: sign={target_letter}, raw_bearing={detection.full_bearing_deg:+.1f}도, "
            f"pad_bearing={pad_bearing:+.1f}도, "
            f"distance~{distance:.2f}m, mode={pad_distance_mode}, "
            f"target_scale={target_scale:.2f}, align_scale={align_scale:.2f}, "
            f"image_angle={detection.angle_deg:+.1f}도, "
            f"head_yaw={detection.frame.yaw:+.2f}, image={detection.frame.path}, "
            f"xy=({target_xy[0]:+.2f}, {target_xy[1]:+.2f})"
        )
        print(
            f"목적지 정렬 좌표: scale={target_scale:.2f} 접근 후 "
            f"scale={align_scale:.2f} 지점으로 한 번 더 이동해 목적지 방향을 바라봅니다. "
            f"align_xy=({align_xy[0]:+.2f}, {align_xy[1]:+.2f})"
        )

    nav = await go_to_xy(ctx, *target_xy)
    print(f"목적지 이동 결과: {result_summary(nav)}")
    if is_stuck_recovered(nav):
        memory.pad_estimates.pop(target_letter, None)
        if route_retry_count < 1:
            print(f"목적지 이동 정체 복구: sign={target_letter} 좌표를 버리고 즉시 다시 탐색합니다.")
            return await navigate_and_place(
                ctx,
                observation,
                memory,
                tracker,
                allow_post_place_pick=allow_post_place_pick,
                route_retry_count=route_retry_count + 1,
            )
        print(f"목적지 이동 정체 복구: sign={target_letter} 재탐색도 한 번 사용했으므로 다음 cycle로 넘깁니다.")
        return {
            "action": "place_cube",
            "status": "retry_next_cycle",
            "reason": "go_to_stuck_recovered",
            "target_letter": target_letter,
            "target_xy": target_xy,
        }
    await asyncio.sleep(0.7)

    delivered_before = observation.delivered_count
    if align_xy is not None:
        first_attempt = await attempt_place(
            ctx,
            tracker,
            delivered_before=delivered_before,
            label=(
                f"1차 놓기 시도: scale={target_scale:.2f} 접근 좌표에서 먼저 place를 호출합니다. "
                f"target_xy=({target_xy[0]:+.2f}, {target_xy[1]:+.2f})"
            ),
        )
        print(
            f"1차 놓기 결과: placed={first_attempt.placed}, released={first_attempt.released}, "
            f"scored={first_attempt.scored}, held_after={first_attempt.held_after}"
        )
        if first_attempt.placed or first_attempt.released:
            return await finish_place_attempt(
                ctx,
                memory,
                tracker,
                attempt=first_attempt,
                target_letter=target_letter,
                target_xy=target_xy,
                held_color=held_color,
                used_saved_xy=used_saved_xy,
                allow_post_place_pick=allow_post_place_pick,
            )
        print(
            f"1차 놓기 실패: 큐브가 아직 잡혀 있으므로 scale={align_scale:.2f} 정렬 좌표로 이동한 뒤 다시 시도합니다."
        )

    if align_xy is not None:
        align_nav = await go_to_xy(ctx, *align_xy)
        print(f"목적지 방향 정렬 이동 결과: {result_summary(align_nav)}")
        if is_stuck_recovered(align_nav):
            memory.pad_estimates.pop(target_letter, None)
            if route_retry_count < 1:
                print(f"목적지 정렬 이동 정체 복구: sign={target_letter} 좌표를 버리고 즉시 다시 탐색합니다.")
                return await navigate_and_place(
                    ctx,
                    observation,
                    memory,
                    tracker,
                    allow_post_place_pick=allow_post_place_pick,
                    route_retry_count=route_retry_count + 1,
                )
            return {
                "action": "place_cube",
                "status": "retry_next_cycle",
                "reason": "align_go_to_stuck_recovered",
                "target_letter": target_letter,
                "target_xy": align_xy,
            }
        target_xy = align_xy
        await asyncio.sleep(0.4)

    if used_saved_xy:
        print(f"목적지 보정 생략: 저장된 sign={target_letter} 좌표를 사용했으므로 바로 place를 시도합니다.")
    else:
        refine_detection = await scan_for_letter(ctx, target_letter, cycle=memory.cycle, max_body_turns=1, memory=memory)
        if refine_detection is not None:
            if refine_detection.area >= 8500:
                print(
                    f"목적지 보정 생략: sign={target_letter}이 이미 충분히 큽니다. "
                    f"area={refine_detection.area}, bbox={refine_detection.bbox}"
                )
            else:
                refine_status = await get_robot_status(ctx)
                refine_distance = estimate_pad_sign_distance(refine_detection)
                refine_bearing = pad_bearing_from_detection(refine_detection)
                refine_target_scale, _refine_align_scale, refine_mode = pad_scales_for_distance(refine_distance)
                refine_xy = xy_from_bearing(
                    refine_status,
                    bearing_deg=refine_bearing,
                    distance_m=refine_distance,
                    standoff_m=PAD_REFINE_STANDOFF_M,
                    travel_scale=refine_target_scale,
                )
                print(
                    f"목적지 보정 좌표: sign={target_letter}, raw_bearing={refine_detection.full_bearing_deg:+.1f}도, "
                    f"pad_bearing={refine_bearing:+.1f}도, "
                    f"distance~{refine_distance:.2f}m, mode={refine_mode}, scale={refine_target_scale:.2f}, "
                    f"xy=({refine_xy[0]:+.2f}, {refine_xy[1]:+.2f})"
                )
                refine_nav = await go_to_xy(ctx, *refine_xy)
                print(f"목적지 보정 이동 결과: {result_summary(refine_nav)}")
                if is_stuck_recovered(refine_nav):
                    memory.pad_estimates.pop(target_letter, None)
                    print(f"목적지 보정 이동 정체 복구: sign={target_letter} place를 생략하고 다음 cycle에서 재탐색합니다.")
                    return {
                        "action": "place_cube",
                        "status": "retry_next_cycle",
                        "reason": "refine_go_to_stuck_recovered",
                        "target_letter": target_letter,
                        "target_xy": refine_xy,
                    }
                target_xy = refine_xy
                await asyncio.sleep(0.5)
        else:
            print("목적지 보정: 현재 위치에서 표지판 재검출 실패. 기존 위치에서 place를 시도합니다.")

    final_attempt = await attempt_place(
        ctx,
        tracker,
        delivered_before=delivered_before,
        label="최종 놓기 시도: 보정/정렬 이후 place를 호출합니다.",
    )
    return await finish_place_attempt(
        ctx,
        memory,
        tracker,
        attempt=final_attempt,
        target_letter=target_letter,
        target_xy=target_xy,
        held_color=held_color,
        used_saved_xy=used_saved_xy,
        allow_post_place_pick=allow_post_place_pick,
    )


async def recover(ctx: Any, memory: AgentMemory) -> dict[str, Any]:
    print("복구 동작: 후진 후 왼쪽 180도 회전")
    await move_velocity(ctx, vx=-0.35, duration_s=1.0)
    await turn_scan(ctx, direction="left", degrees=180)
    memory.failed_attempts["recover"] = memory.failed_attempts.get("recover", 0) + 1
    return {"action": "recover", "status": "done"}


def prompt_round_completion_config(
    level: int = 1,
    existing: CompletionConfig | None = None,
) -> CompletionConfig:
    round_seconds = {
        "round1": 5 * 60,
        "round2": 10 * 60,
        "round3": 15 * 60,
    }
    try:
        selected = input("Round 선택 [round1/round2/round3/manual] (Enter=round2): ").strip().lower()
    except EOFError:
        if existing is not None:
            print("Round 선택 입력을 받을 수 없어 CLI completion 설정을 그대로 사용합니다.")
            return existing
        selected = ""
    selected = selected or "round2"
    if selected == "manual":
        while True:
            raw_seconds = input("Manual 제한 시간(초)을 입력하세요: ").strip()
            try:
                max_elapsed_s = float(raw_seconds)
            except ValueError:
                print("숫자로 다시 입력하세요.")
                continue
            if max_elapsed_s <= 0:
                print("0보다 큰 시간을 입력하세요.")
                continue
            break
        round_label = f"manual {max_elapsed_s:.0f}s"
    else:
        if selected not in round_seconds:
            print(f"알 수 없는 round={selected!r}. 기본 round2를 사용합니다.")
            selected = "round2"
        max_elapsed_s = float(round_seconds[selected])
        round_label = selected

    print(
        f"평가 라운드 설정: {round_label}, time_limit={max_elapsed_s:.1f}s, "
        "max_delivered_cubes=12, level1 scoring=20 points/cube"
    )
    return CompletionConfig(
        level=level,
        max_elapsed_s=max_elapsed_s,
        max_delivered_cubes=12,
    )


async def maybe_prepare_evaluation_setup(ctx: Any, *, level: int = 1) -> None:
    try:
        option = input("Evaluation setup option [1-50] (Enter=현재 scene/pose 유지): ").strip()
    except EOFError:
        print("Evaluation setup 입력을 받을 수 없어 현재 scene과 robot pose를 그대로 사용합니다.")
        return
    if not option:
        print("Evaluation setup: 현재 scene과 robot pose를 그대로 사용합니다.")
        return
    try:
        option_num = int(option)
    except ValueError:
        print(f"Evaluation setup option={option!r}은 숫자가 아니므로 setup을 생략합니다.")
        return
    if not 1 <= option_num <= 50:
        print(f"Evaluation setup option={option_num}은 1~50 범위를 벗어나 setup을 생략합니다.")
        return

    setup_seed = os.environ.get("EVAL_SETUP_SEED", f"option-{option_num:02d}")
    clearance_m = float(os.environ.get("EVAL_OBSTACLE_CLEARANCE_M", DEFAULT_OBSTACLE_CLEARANCE_M))
    setup = choose_evaluation_setup(level, setup_seed)
    setup = await apply_clear_start_from_layout(ctx, setup, clearance_m=clearance_m)

    print("=" * 60)
    print("Evaluation setup")
    print(f"level: {setup.level}")
    print(f"option: {option_num}")
    print(f"setup_seed: {setup.setup_seed}")
    print(f"cube_color_order_key: {setup.cube_color_order_key}")
    print(f"start_xy: ({setup.start_x:+.3f}, {setup.start_y:+.3f})")
    print(f"obstacle_clearance_m: {clearance_m:.2f}")
    print("=" * 60)
    await reload_current_scene(ctx)
    try:
        input(
            "viewer seed box에 cube_color_order_key를 입력하고 apply/reset한 뒤 Enter를 누르세요..."
        )
    except EOFError:
        print("viewer seed 확인 입력을 받을 수 없어 현재 상태에서 시작 위치 이동을 진행합니다.")
    await go_to_start_position(ctx, setup)
    print("Evaluation setup complete.")


async def prepare_level_1_run_options(
    ctx: Any,
    completion: CompletionConfig | None,
) -> CompletionConfig:
    if os.environ.get("MENLO_LEVEL1_SKIP_RUN_OPTIONS", "").lower() in {"1", "true", "yes"}:
        if completion is not None:
            return completion
        return CompletionConfig(level=1, max_elapsed_s=10 * 60, max_delivered_cubes=12)

    level = completion.level if completion is not None else 1
    configured_completion = prompt_round_completion_config(level=level, existing=completion)
    await maybe_prepare_evaluation_setup(ctx, level=level)
    return configured_completion


async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
    prompt_run_options: bool = True,
) -> AgentMemory:
    if prompt_run_options:
        completion = await prepare_level_1_run_options(ctx, completion)

    memory = AgentMemory()
    tracker = CompletionTracker(completion) if completion is not None else None

    for cycle in range(1, max_cycles + 1):
        memory.cycle = cycle
        print(f"\n[my_real_level_1] Cycle {cycle}")
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

        observation = await observe_world(ctx, memory, tracker)
        print(
            f"관찰: stage={memory.stage}, held_color={observation.held_color}, "
            f"delivered={observation.delivered_count}, color_detections={len(observation.colors)}"
        )

        if tracker is None and observation.delivered_count >= MAX_DELIVERIES:
            print("목표 배송 수 도달")
            break

        decision = await ask_llm_agent_decision(ctx, observation, memory)
        print(f"Decision: {decision}")

        if decision.action == "stop":
            print(f"LLM 판단으로 중지합니다. reason={decision.reason}")
            break
        if decision.action == "recover":
            action_result = await recover(ctx, memory)
        elif decision.action == "deliver_held_cube":
            action_result = await navigate_and_place(ctx, observation, memory, tracker)
        else:
            action_result = await approach_and_pick(ctx, observation, memory)

        memory.last_result = action_result
        if action_result.get("status") == "failed":
            await recover(ctx, memory)
        elif action_result.get("picked") is False:
            print("집기 실패: 180도 회전 복구는 생략하고 다음 cycle에서 A/source를 다시 탐색합니다.")

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
    print("Level 1 coordinate-guided agent 실행")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=None,
        prompt_run_options=True,
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
