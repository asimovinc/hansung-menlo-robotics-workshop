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
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTimeout, CompletionTracker
from menlo_runner.config import load_config
from menlo_runner.llm import ask_vlm, call_llm
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
    # 런타임에 추정한 컨베이어 벨트/레일 색(제외가 아니라 획득 시 '후순위' 용도).
    belt_color: str | None = None
    # ★run14 belt_color 플랩 방어★: sticky belt_color를 넘보는 '도전색'과 그 연속 관측 수.
    # 벨트 색은 한 run 동안 불변인데 _detect_belt_color가 매 프레임 max-area blob 색을 돌려주는
    # 탓에 노란 랙이 이기는 프레임에 파랑-벨트 방어(D2/선제우회/A3)가 통째로 꺼졌다(run14 4:4).
    belt_challenge_color: str | None = None
    belt_challenge_count: int = 0
    # pick 연속 실패 횟수: 같은 큐브에 막히면 더 크게 relocate하기 위한 카운터.
    pick_fail_streak: int = 0
    # A3: '전진 0 + too-far' pick 차단의 연속 횟수 — PICK_BYPASS_AFTER_N 도달 시 벨트 우회로 전환.
    pick_blocked_streak: int = 0
    # A2: 직전 navpad 도착 기록 {letter,x,y} — 이동 PAD_ARRIVED_STICKY_M 이내면 navpad 재실행 생략.
    pad_arrived: dict[str, Any] | None = None
    # 마지막으로 실제 집은 색(get_held_cube_info ground truth).
    last_grabbed_color: str | None = None
    # 최근 pick 실패 메모(수명 ttl). 존재하면 recover가 더 크게 회전해 stuck 큐브를 벗어납니다.
    recent_pick_fail: dict[str, Any] | None = None
    # --- 경로 기억(성공 경험 greedy 재사용; 엄밀한 RL이 아닌 online heuristic 최적화) ---
    # pad_memory[color] = {"last_seen", "anchor", "successful_routes", "failed_routes",
    # "best_route"}. 전부 로봇 자신의 odometry pose(고유수용성) + 카메라/VLM 관찰에서 유도한
    # 학생 추정치이며 scene_state/entity ID가 아닙니다. 한 run 안에서만 축적됩니다(영속 캐시 없음).
    pad_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 현재 배송(pick 성공 → place 성공)의 step 단위 이동 기록. 성공 시 pad_memory로 승격.
    route_trace: list[dict[str, Any]] = field(default_factory=list)
    # 현재 배송의 누적 통계(t0, start_pose, vlm_calls, stalls, path_len_m).
    route_stats: dict[str, Any] = field(default_factory=dict)
    # stall(전진 병진 실패)이 났던 pose(x, y, 당시 yaw) 기록 — 같은 지점·같은 방향 재돌진 방지.
    stall_spots: list[dict[str, float]] = field(default_factory=list)
    # 우회가 실제로 병진을 확보한 지점·방향("성공한 우회") 기록 — 같은 지점 재방문 시 그 방향 우선.
    detour_wins: list[dict[str, float]] = field(default_factory=list)
    # --- pad 지도(survey-first): 표지 letter별 관측 ray 축적 + 삼각측량 동결 좌표 ---
    # sign_rays[letter] = [{"x","y","bearing_deg","conf"}, ...] — 각 VLM 목격의 (관측 pose,
    # world 방위각) ray. 스폰 서베이가 부트스트랩하고 이후 항법 중 목격이 계속 쌓입니다.
    # sign_goals[letter] = {"x","y"} — 기선·교각 조건을 만족한 두 ray의 교점(삼각측량)으로
    # '동결'된 pad world 좌표. 동결 후에는 원거리 bbox 거리 추정(요동 2.9~6.0m 실측)이 목표를
    # 끌고 다니지 못합니다. 전부 자기 odometry + 카메라/VLM 유도 추정치(Level 2 합법),
    # 한 run 안에서만 유효합니다.
    sign_rays: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    sign_goals: dict[str, dict[str, float]] = field(default_factory=dict)
    # M1: 동결 폐기(_drop_sign_map) 직후 그 letter의 재동결을 SIGN_REFREEZE_COOLDOWN_LOOKS 회의
    # freeze 시도 동안 막는 쿨다운 카운터(letter→남은 시도 수). 폐기 직후 같은 데코이가 즉시
    # 재응집하는 루프를 차단합니다. 감소 단위 = _maybe_freeze_sign_goal 호출(= 유효 ray 추가). run 스코프.
    sign_refreeze_block: dict[str, int] = field(default_factory=dict)
    # place-probe 성공 시 실측 반경(letter→m). 기록·로그·발표 전용 — 어떤 제어 로직도 이 값을 읽지
    # 않습니다(§8-5: 표지-팔레트 오프셋은 랙 face 방향 종속이라 되먹임하면 오배송). run 스코프.
    place_radius: dict[str, float] = field(default_factory=dict)
    # source-seek 우선순위 ①: 서베이 스윕에서 관측한 clean cube의 world 방위·면적 스냅샷
    # {"bearing_deg","area","x","y"} 리스트(상한 CUBE_SIGHTINGS_KEEP). 서베이 시점 스냅샷이라 이동 후
    # stale해질 수 있으나 방위 힌트 용도라 무해. place·navpad는 읽지 않습니다(획득 전용). run 스코프.
    cube_sightings: list[dict[str, float]] = field(default_factory=list)
    # source-seek 폴백(소스 단서 전무) 누계 턴 수 — SRC_FALLBACK_MAX_ROUNDS 초과 시 기존 아크-스윕 위임.
    source_fallback_rounds: int = 0
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

    이 구조는 특정 strategy에 묶이지 않도록 의도적으로 중립적입니다. 
    Level 1 팀은 coordinate estimate에 full bearing을 사용할 수 있고, 
    Level 2 팀은 closed-loop visual centering에 사용할 수 있습니다. 
    필요하면 confidence, target type, depth field를 추가하세요.
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
    return await ctx.invoke("set_head", args, timeout_s=30)


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
        timeout_s=300,
    )


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Code가 robot을 시각적으로 충분히 위치시킨 뒤 nearest cube를 집습니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=900,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """Matching pad에 도달한 뒤 nearest zone에 place합니다."""
    return await ctx.invoke("place_entity", {}, timeout_s=900)


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


def _too_far_m(error_message: str | None) -> float | None:
    """pick/place 실패 메시지에서 'X.XXm > 1.20m' 꼴의 실거리(좌변)를 뽑습니다(순수 — pytest로 잠급니다).

    SDK의 too-far 실패는 최근접 대상까지의 실거리를 담습니다(라이브 실측 두 포맷:
    'too far from cube (2.60m > 1.20m)' / '거리 초과 1.57m>1.20m'). 이는 area 신기루(병합
    blob·구조물)에 속은 도착 판정을 사후 보정할 수 있는 유일한 정밀 range 피드백입니다(run6:
    area 17241 '도착' → 실거리 2.60m). '>' 좌변이 'm'으로 끝나면 그 직전 숫자 토큰을 후방
    스캔으로 뽑고(re 미사용 — _parse_place_error와 같은 코드베이스 관례), 우변에도 숫자+'m'
    (반경)이 있어야 too-far 계약으로 인정합니다. 포맷 불일치·too-far 아님·None이면 None을
    돌려주어 호출부가 기존 실패 경로를 그대로 타게 합니다. scene_state·좌표·엔티티 조회가
    아니라 허용 skill 호출의 결과 메시지(스칼라)만 사용합니다(§0/§7 합법).
    """
    if not isinstance(error_message, str):
        return None
    gt = error_message.find(">")
    if gt < 0:
        return None
    left = error_message[:gt].rstrip()
    if not left.endswith("m"):
        return None
    end = len(left) - 1  # 'm' 위치. 그 앞의 숫자 토큰(좌변 마지막 수 = 실거리)을 후방 스캔.
    start = end
    seen_dot = False
    while start > 0:
        ch = left[start - 1]
        if ch.isdigit():
            start -= 1
        elif ch == "." and not seen_dot:
            seen_dot = True
            start -= 1
        else:
            break
    if start == end:
        return None
    right = error_message[gt + 1:]
    if _first_float_in(right) is None or "m" not in right:
        return None
    try:
        return float(left[start:end])
    except ValueError:
        return None


def _pad_arrival_fresh(
    entry: dict[str, Any] | None, letter: str | None, pose: dict[str, Any], max_move_m: float
) -> bool:
    """직전 navpad 도착 기록이 이 letter에 아직 유효한지 판정합니다(순수 — pytest로 잠급니다).

    A2(자기완결 place): run8에서 place 실패 → LLM → navigate_to_pad 재실행 → 도착 재증명(17~104s,
    재확인 VLM 플랩 90~144s 포함) → place … 핑퐁이 라운드를 태웠습니다. 도착 기록(letter,x,y)
    이후 이동 거리가 max_move_m 이내면 '아직 그 pad 앞'이므로 navpad를 다시 돌리지 않고 진입
    push+place로 직행합니다. 기록 없음·다른 letter·이동 초과·포즈 파손이면 False(기존 navpad 경로).
    odometry 상대 이동량만 쓰므로 §0/§7 합법입니다.
    """
    if not entry or letter is None or entry.get("letter") != letter:
        return False
    try:
        dx = float(pose["x"]) - float(entry["x"])
        dy = float(pose["y"]) - float(entry["y"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.hypot(dx, dy) <= max_move_m


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

    system_prompt = (
        "당신은 humanoid warehouse robot의 상위 단계 결정을 담당합니다. "
        "목표: source area(A)의 cube를 집어 색상과 일치하는 destination pad(B red, C green, D blue, E yellow)에 놓기. "
        f"다음 행동 중 정확히 하나만 고르세요: {', '.join(sorted(ALLOWED_NEXT_ACTIONS))}. "
        "raw 속도 명령은 출력하지 말고 상위 단계 결정만 내리세요. "
        "채점은 색 무관입니다(정확히 분류된 큐브당 동일 점수). 특정 색을 고집하지 마세요. "
        "규칙 — held_color가 null이면 '획득' 단계: clean_cubes에 큐브가 보이면 pick_cube를 고르세요 "
        "(pick_entity가 각도·거리와 무관하게 최근접 큐브를 스스로 접근·파지하므로 별도 정렬이 필요 없습니다). "
        "보이는 clean cube가 전혀 없을 때만 search_cube로 탐색하세요. 어떤 색이든 좋으니 target_color는 null로 두면 됩니다. "
        "특히 note에 소스 힌트(source=/src_cube=/source_ray=)가 있으면 반드시 search_cube를 선택하세요 — "
        "실행부가 소스(A)로 전진해 큐브를 시야에 넣습니다(랜덤 스폰 대응). "
        "held_color가 있으면 '배송' 단계: target_color를 반드시 held_color로 두고 그 색 pad로 "
        "navigate_to_pad 또는 place_cube 하세요. "
        "clean_cubes 힌트가 있으면 그 목록의 cube를 우선 노리고, belt_color는 후순위로 두세요. "
        "failed_attempts가 3 이상 쌓인 색은 skip_target 하세요. "
        "설명 없이 아래 schema의 JSON 하나만 반환하세요: "
        '{"next_action": "<action>", "target_color": "<color 또는 null>", "reason": "<짧은 이유>"}.'
    )
    user_prompt = json.dumps(decision_context, ensure_ascii=False)

    decision: AgentDecision | None = None
    try:
        config = load_config(require_tokamak=True)
        raw = call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=config.tokamak_api_key,
        )
        decision = parse_agent_decision(raw)
    except Exception:
        decision = None

    # LLM은 상위 시퀀서/ratifier로만 씁니다. 저수준 제어와 색 정합성은 코드가 강제합니다.
    if decision is not None:
        if memory.held_color:
            # 배송 단계: 실제 든 색으로 강제해 잘못된 pad 배송을 막습니다.
            decision.target_color = memory.held_color
        # 획득 단계에서는 target_color를 강제하지 않습니다(색맹 pick과 정합; 참고용).
        return decision

    # LLM 실패 시 rule-based로 degrade합니다(Tokamak 장애에도 동작을 이어갑니다).
    if memory.held_color:
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=memory.held_color,
            reason="대체 동작: LLM 실패, 든 색 pad로 이동.",
        )
    if decision_context["visible_targets"]:
        return AgentDecision(
            next_action="pick_cube",
            target_color=None,
            reason="대체 동작: LLM 실패, 보이는 cube를 pick(pick_entity가 최근접 큐브를 자체 접근·파지).",
        )
    return AgentDecision(next_action="search_cube", reason="대체 동작: LLM 실패, 보이는 target이 없어 탐색.")


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

    # cube를 찾는 단계에서는 head를 넓게 훑어 주변 target을 최대한 많이 봅니다.
    raw_detections = await scan_head(ctx)
    # scan이 끝나면 head를 정면으로 되돌려 이후 body-frame 판단이 흔들리지 않게 합니다.
    await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)
    # LLM에 넘기기 전에 perception 노이즈(컨베이어 레일/바닥 등 비현실적 blob)를 배제합니다.
    # 이렇게 하지 않으면 build_decision_context가 초대형 레일 blob을 "가장 큰 cube"로
    # LLM에 전달해 잘못된 target을 고르게 만듭니다.
    arrival_area = PAD_ARRIVAL_AREA if memory.held_color else CUBE_ARRIVAL_AREA
    detections = [d for d in raw_detections if _plausible_target(d, arrival_area)]
    visible_colors = sorted({d.color for d in detections})
    # 프레임을 뒤덮는 벨트/레일 색을 런타임 추정해 기억합니다(제외가 아니라 후순위 용도).
    # ★run14 플랩 방어★: 매 프레임 덮어쓰지 않고 sticky 규칙을 경유합니다. 노란 랙이 이기는
    # 단발 프레임이 파랑-벨트 방어를 껐던 결함(_detect_belt_color per-frame overwrite)을 막습니다.
    belt = _detect_belt_color(raw_detections)
    memory.belt_color, memory.belt_challenge_color, memory.belt_challenge_count, _belt_outcome = (
        _resolve_belt_color(
            memory.belt_color,
            belt,
            memory.belt_challenge_color,
            memory.belt_challenge_count,
        )
    )
    if _belt_outcome in ("challenge", "switch"):
        _trace_step(
            memory,
            action="belt_color",
            outcome=_belt_outcome,
            sticky=memory.belt_color,
            detected=belt,
            challenge=memory.belt_challenge_color,
            count=memory.belt_challenge_count,
        )
    # 획득에 쓸 '깨끗한 큐브' 후보(색:크기)를 note에 실어 LLM이 벨트/노이즈에 안 휘둘리게 합니다.
    clean = sorted(
        (d for d in detections if _is_clean_cube(d, arrival_area)),
        key=lambda d: d.blob_area,
        reverse=True,
    )

    # cube를 들고 있으면 destination pad를 찾는 단계입니다. 같은 색 pad가 아직
    # 안 보일 때만 VLM으로 signage를 읽어 불필요한 호출과 시간 낭비를 줄입니다.
    # 추가 절감: 그 색의 경로 기억(best_route)이나 last_seen이 이미 있으면 결정용
    # 프리뷰 VLM도 생략합니다 — pad-nav가 필요한 시점에 스스로 look하므로 여기서
    # 또 읽는 것은 순수 중복입니다(사유는 note의 pad_cache로 남김).
    vlm_summary = ""
    pad_cache_note = ""
    if memory.held_color is not None and memory.held_color not in visible_colors:
        cached = memory.pad_memory.get(memory.held_color) or {}
        if cached.get("best_route") or cached.get("last_seen") or cached.get("anchor"):
            pad_cache_note = "pad_cache=hit(vlm_preview_skip)"
        else:
            try:
                config = load_config(require_tokamak=True)
                prompt = build_signage_vlm_prompt(memory.held_color)
                vlm_summary = await ask_vlm_about_frame(
                    ctx,
                    prompt,
                    api_key=config.tokamak_api_key,
                    max_width=SIGNAGE_VLM_MAX_WIDTH,
                    quality=SIGNAGE_VLM_QUALITY,
                )
            except Exception:
                vlm_summary = ""

    note_parts = [f"stage={memory.stage}", f"search_turns={memory.search_turns}"]
    if pad_cache_note:
        note_parts.append(pad_cache_note)
    if memory.held_color:
        target_pad = DESTINATION_SIGN_RULES.get(memory.held_color, "?")
        note_parts.append(f"held={memory.held_color}->pad {target_pad}")
    note_parts.append("visible=" + (",".join(visible_colors) if visible_colors else "none"))
    if memory.belt_color:
        note_parts.append(f"belt_color={memory.belt_color}(deprioritize)")
    if clean:
        note_parts.append("clean_cubes=" + ",".join(f"{d.color}:{d.blob_area}" for d in clean[:5]))
    # source-seek 힌트(획득 단계·clean cube 미시야일 때만): 소스(A)로 유도해 LLM이 search_cube를
    # 고르게 합니다. 코드 강제가 아니라 결정 유도(§5.2) — 실행부는 유효한 next_action을 override하지 않습니다.
    if memory.held_color is None and not clean:
        src_pose = _pose_dict(robot_status)
        src_kind, src_payload = _source_target_priority(memory)
        if src_kind == "cube":
            note_parts.append(f"src_cube={src_payload['bearing_deg']:+.0f}° a={int(src_payload['area'])}")
        elif src_kind == "goal":
            src_dist, src_turn = _face_turn_to(src_pose, src_payload)
            note_parts.append(f"source=A@d≈{src_dist:.1f}m turn={src_turn:+.0f}°")
        elif src_kind == "ray":
            note_parts.append(f"source_ray={src_payload:+.0f}°(미동결)")
        else:
            note_parts.append("source=unknown(sweep)")
        if src_kind != "fallback":
            note_parts.append("clean cube 미시야 -> search_cube 권장")

    return Observation(
        robot_status=robot_status,
        detections=detections,
        note="; ".join(note_parts),
        vlm_summary=vlm_summary,
    )


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 observe하세요.
    - robot_status, camera evidence, SDK result status를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 정보를 반환하세요.
    """
    # 중요한 action 뒤에는 상태를 다시 읽어 실제로 성공했는지 근거를 모읍니다.
    held = await get_held_cube_info(ctx)
    # 느린 pick 직후 held가 아직 None으로 보일 수 있어 한 번 더 확인합니다(타이밍 헤지).
    if held is None and decision.next_action == "pick_cube":
        await asyncio.sleep(0.3)
        held = await get_held_cube_info(ctx)
    delivered_count = await get_delivered_count(ctx)
    robot_status = await get_robot_status(ctx)
    robot_motion = getattr(getattr(robot_status, "robot", None), "status", None)

    # 마지막 target 색이 아직 보이는지, 얼마나 크게 보이는지 다시 관찰합니다.
    target_still_visible: bool | None = None
    target_blob_area: int | None = None
    if decision.target_color is not None:
        try:
            matching = [d for d in await perceive(ctx) if d.color == decision.target_color]
            target_still_visible = len(matching) > 0
            target_blob_area = max((d.blob_area for d in matching), default=0)
        except Exception:
            target_still_visible = None

    # SDK result에 error가 실려 있으면 그대로 노출해 recovery 판단에 씁니다.
    result_error = None
    inner_result = action_result.get("result")
    if isinstance(inner_result, dict):
        result_error = inner_result.get("error")

    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": delivered_count,
        "held_cube": held,
        "held_color": held["color"] if held else None,
        "robot_motion": str(robot_motion) if robot_motion is not None else None,
        "target_still_visible": target_still_visible,
        "target_blob_area": target_blob_area,
        "result_error": result_error,
        # 경로 기억용 odometry pose(고유수용성). update_memory가 배송 시작/드롭 지점으로 씁니다.
        "pose": _pose_dict(robot_status),
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
    prev_delivered = memory.delivered_count
    new_delivered = int(verified.get("delivered_count", prev_delivered))
    new_held = verified.get("held_color")
    pose = verified.get("pose")

    action = decision.next_action
    color = decision.target_color
    action_result = verified.get("action_result") or {}

    # 경로 기록 수명주기: 획득→배송 전환(방금 집음) 시점에 새 배송 trace를 시작합니다.
    # 이 trace에 pad-nav의 모든 이동(기대/실제 이동량·stall·VLM look)이 쌓이고,
    # 배송이 성공하면 아래에서 pad_memory[색]으로 승격됩니다.
    # began_new_trace: 이 cycle에 새 배송 trace가 방금 시작됐는지(R2 커밋 가드의 입력).
    began_new_trace = memory.held_color is None and bool(new_held)
    if began_new_trace:
        memory.route_trace = []
        memory.route_stats = {
            "target_color": new_held,
            "t0": time.monotonic(),
            "start_pose": pose,
            "vlm_calls": 0,
            "stalls": 0,
            "path_len_m": 0.0,
        }

    # cube를 찾거나 접근하는 동안의 목표 색을 active_color로 기억합니다.
    if action in {"search_cube", "navigate_to_cube"} and color:
        memory.active_color = color

    # 배달 성공: delivered_count가 늘면 방금 놓은 색을 완료 목록에 넣습니다.
    if new_delivered > prev_delivered:
        placed = memory.held_color or memory.active_color
        if placed and placed not in memory.completed_colors:
            memory.completed_colors.append(placed)
        if placed:
            memory.failed_attempts.pop(placed, None)
        # 성공 경로 승격(R2 커밋 가드): 이 cycle에 커밋해도 되는 경로일 때만 저장합니다.
        #  - t0가 없으면 추적된 배송이 아님(시작 전이거나 이미 커밋됨) → 커밋 금지.
        #  - 같은 cycle에 새 배송 trace가 시작됐으면(delivered가 1-cycle 늦게 잡힌 경우)
        #    route_stats는 '방금 집은 다음 큐브' 것이므로, 커밋하면 score≈0 쓰레기 경로가
        #    best_route(min-score라 이후에 못 이김)로 굳어짐 → 커밋 금지(학습 1건 생략 감수).
        # 커밋 색 키는 placed 추정이 아니라 route_stats["target_color"](pick 시점 ground truth).
        if _should_commit_route(memory.route_stats, began_new_trace):
            route_color = memory.route_stats.get("target_color") or placed
            entry = _pad_memory_entry(memory.pad_memory, route_color)
            stats = {k: v for k, v in memory.route_stats.items() if k != "t0"}
            stats["total_time_s"] = round(time.monotonic() - memory.route_stats["t0"], 1)
            waypoints = _route_waypoints(
                memory.route_stats.get("start_pose"), memory.route_trace, pose
            )
            route = _commit_successful_route(entry, waypoints, stats, drop_pose=pose)
            print(
                f"[route] {route_color} 배송 경로 저장: score={route['score']:.1f}"
                f" (time={stats['total_time_s']}s vlm={stats.get('vlm_calls', 0)}"
                f" stall={stats.get('stalls', 0)} path={stats.get('path_len_m', 0.0):.1f}m"
                f" wp={len(waypoints)})"
            )
            memory.route_trace = []
            memory.route_stats = {}
        memory.active_color = None
        memory.pick_fail_streak = 0
        memory.recent_pick_fail = None

    # pick 결과 판정: 이제 들고 있으면 성공, 아니면 실패 횟수를 늘립니다.
    if action == "pick_cube":
        if new_held:
            memory.failed_attempts.pop(color or new_held, None)
            memory.pick_fail_streak = 0
            memory.recent_pick_fail = None
            memory.last_grabbed_color = new_held
        else:
            key = color or memory.active_color
            if key:
                memory.failed_attempts[key] = memory.failed_attempts.get(key, 0) + 1
            # 같은 큐브에 막혀 색맹 pick이 헛집는 것을 막기 위해 relocate를 유도합니다.
            memory.pick_fail_streak += 1
            memory.recent_pick_fail = {"ttl": 2}

    # place 실패 판정: 배달 수 변화 없이 아직 들고 있으면 실패로 간주합니다.
    if action == "place_cube" and new_delivered == prev_delivered and new_held:
        memory.failed_attempts[new_held] = memory.failed_attempts.get(new_held, 0) + 1

    # navigate 실패 판정: 도착하지 못했으면 실패 횟수를 늘립니다.
    if action in {"navigate_to_cube", "navigate_to_pad"} and action_result.get("reached") is False and color:
        memory.failed_attempts[color] = memory.failed_attempts.get(color, 0) + 1

    # pad 접근 실패는 통계와 함께 진단용으로 남깁니다(다음 시도 전략·발표 근거).
    if action == "navigate_to_pad" and action_result.get("reached") is False and memory.held_color:
        _record_failed_route(
            _pad_memory_entry(memory.pad_memory, memory.held_color),
            {k: v for k, v in memory.route_stats.items() if k != "t0"},
            reason="navigate_to_pad_failed",
        )

    # search 진행도: 찾았으면 0으로, 못 찾았으면 누적해 무한 탐색을 감지합니다.
    if action in {"search_cube", "search_pad"}:
        memory.search_turns = 0 if action_result.get("found") else memory.search_turns + 1
    else:
        memory.search_turns = 0

    # skip 기록: 반복 실패로 건너뛴 색을 남기고 active에서 제외합니다.
    if action == "skip_target" and color and color not in memory.skipped_colors:
        memory.skipped_colors.append(color)
        memory.failed_attempts.pop(color, None)
        if memory.active_color == color:
            memory.active_color = None

    # 실패-회피 메모 수명 관리(존재 시 recover가 더 크게 회전; 시간이 지나면 소멸).
    if memory.recent_pick_fail is not None:
        memory.recent_pick_fail["ttl"] -= 1
        if memory.recent_pick_fail["ttl"] <= 0:
            memory.recent_pick_fail = None

    # 최종 상태 반영.
    memory.delivered_count = new_delivered
    memory.held_color = new_held
    memory.stage = "deliver_cube" if new_held else "need_cube"

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
        },
        "llm_decision": decision.__dict__,
        "memory": {
            "active_color": memory.active_color,
            "completed_colors": list(memory.completed_colors),
            "skipped_colors": list(memory.skipped_colors),
            "failed_attempts": dict(memory.failed_attempts),
            "search_turns": memory.search_turns,
            "route_stats": {
                k: v for k, v in memory.route_stats.items() if k not in {"t0", "start_pose"}
            },
            "pad_routes": {
                c: len(e.get("successful_routes", [])) for c, e in memory.pad_memory.items()
            },
            "stall_spots": len(memory.stall_spots),
        },
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# LEVEL 2 학생 TODO: vision-only action 구현
# ---------------------------------------------------------------------------
# Level 2는 go_to를 호출하면 안 됩니다. Camera observation, set_head,
# set_velocity, memory, recovery behavior로 navigate하세요.

# 아래 상수는 vision-only navigation의 튜닝 값입니다. 팀 전략에 맞게 조정하세요.
CENTER_TOLERANCE_DEG = 10.0   # 이 각도 안이면 target이 화면 중앙에 있다고 봅니다.
CUBE_ARRIVAL_AREA = 9000      # cube blob이 이만큼 크면 pick할 만큼 가깝다고 봅니다.
PAD_ARRIVAL_AREA = 20000      # pad blob이 이만큼 크면 place할 만큼 가깝다고 봅니다.
MIN_TARGET_AREA = 300         # 이보다 작은 blob은 noise로 무시합니다.
NAV_MAX_STEPS = 14            # navigation 한 번에서 최대 servo step 수.
SEARCH_MAX_ROTATIONS = 8      # search에서 body를 회전시키는 최대 횟수(대략 한 바퀴).
# --- M2: 헤드 pitch 정책(용도별 분리) + 근접 표지 재확인 ---
# 표지 추적은 0.15로 충분하고(라이브 확정), 바닥/free-space 확인은 관절 포화점 0.45가 실질 max-down.
# 두 pitch는 용도가 다르고 광학 전제도 달라(RACK_GROUND_A/B·FREE_*는 0.15 전제) 분리 배선합니다.
HEAD_PITCH_TRACK = 0.15  # M2: 표지 추적/전방 서보 헤드 pitch(옛 리터럴 pitch=0.15 8곳 치환, 값 불변).
                         # 변경 금지 — RACK_GROUND_*/FREE_* 지면 투영이 이 pitch를 전제(G6).
HEAD_PITCH_FLOOR = 0.45  # M2: 바닥/free-space 확인 전용(관절 포화 실측 0.45=0.75 동일 프레임 = 실질 max-down).
                         # 관측 전용 — _probe_free_space/_rack_map_from_frame에 배선 금지(Q6 미해결, G6).
HEAD_PITCH_SIGN_RECHECK = -0.15  # 잠정(M3 Q5 보정): d<1m 표지 상실 시 위쪽 1회 재확인(프레임 상단 잘림 대비).
PAD_SIGN_LOST_NEAR_M = 1.0        # 잠정(M3 Q5 보정): 근접 재확인 발동 반경(PAD_PLACE_NEAR_M=1.0 동치 수치).
PAD_SIGN_NEAR_RECHECK = False     # 세션5 정찰(핸드오프 §3)이 프레임아웃 가설 반증: d≈0.85m 사인은
                                  # 완전 프레임 내이고 근접 상실은 네트워크 플랩. 위 피치(-0.15)는 눈높이
                                  # 이하 사인을 오히려 하단 이탈시켜 net-loss → 재확인 비활성(근거·플럼빙 보존, 가역).

# --- perception noise 방어용 상수 ---
# 주의: 이 값들은 "이 씬의 좌표"가 아니라 "cube/pad라는 물체의 성질"에서 유도합니다.
# 그래서 시작 위치가 바뀌는 히든 평가에서도 성립하고, hardcoding에 해당하지 않습니다.
MAX_AREA_ARRIVAL_MULT = 4.0  # 단일 target은 도착 크기의 이 배수를 넘을 수 없음(컨베이어 레일/바닥 배제).
MIN_TARGET_ASPECT = 0.4      # cube/pad는 대략 정사각. 길고 얇은 blob(레일)은 배제.
MAX_TARGET_ASPECT = 2.5      # 정상 target이 걸러지면 이 범위를 넓히세요.
MAX_TARGET_WIDTH_FRAC = 0.6  # cube/pad는 프레임 폭의 이 비율을 넘지 않음(가로로 긴 컨베이어 레일 밴드 배제).

# --- "깨끗한 큐브" 판별 상수 ---
# cube면은 bbox를 꽉 채운 정사각으로 보입니다. 관측상 실제 큐브면 fill 0.92~0.93, aspect~1.0;
# 벨트/반사/구조물은 fill 0.45~0.67, aspect 1.3~1.5. 아래 값은 그 물리적 간극에서 유도합니다(씬 좌표 아님).
CLEAN_FILL_MIN = 0.80        # 이 이상 채워져야 단단한 큐브면으로 인정.
CLEAN_ASPECT_MIN = 0.7       # 큐브면은 거의 정사각. 가로로 긴 레일 밴드(≈1.3~1.5)를 배제.
CLEAN_ASPECT_MAX = 1.4
CLEAN_CUBE_MAX_MULT = 2.5    # ★D1a(run6 전도)★ clean '큐브' 면적 상한 배수(9000×2.5=22500). 레일
                             # 배제 상한(36000) 아래로 들어온 belt-scale 정사각 blob(33772)이 clean
                             # 큐브로 통과해 pick 타겟이 됐음. 실측 pick 성공 area는 ~8k(1.2m)라
                             # 실큐브가 이 상한에 걸릴 일이 없고, 이보다 크면 구조물/병합 덩어리.
NAV_RESELECT_AREA_RATIO = 1.3  # 획득 nav에서 다른 큐브로 target을 바꾸려면 이 배수 이상 커야 함(진동 방지).
PICK_FAIL_RELOCATE = 2       # pick이 이만큼 연속 실패하면 크게 relocate해 다른 큐브가 최근접이 되게 함.
PICK_READY_AREA = 7000       # clean 큐브가 이만큼 크면(≈충분히 가까우면) 중앙정렬 없이 pick 발사.
# ★사용자(정지 큐브 우선): 컨베이어 위 큐브는 접근하는 사이 벨트가 실어 옮겨 1.2m 반경에 못 넣고
# 실패·relocate 순환에 빠집니다(run2/4 확정). 로봇 정지 상태 2프레임 차로 '정지' 큐브만 인정 — 좌표
# 0(카메라 angle_deg/area만). 벨트 위 이동 큐브는 프레임 사이 angle_deg가 크게 바뀌어 제외됩니다.
STATIONARY_DT_S = 0.4         # 움직임 감지 2프레임 간격(초). 벨트 이동이 드러날 만큼.
STATIONARY_MOVE_DEG = 3.0     # ★잠정(E2E 보정)★: 프레임 사이 angle_deg 변화가 이 이상이면 이동으로 제외.
STATIONARY_AREA_FRAC = 0.30   # ★잠정★: blob_area가 이 비율 이상 변하면(전후 이동) 이동으로 제외.
                             # pick_entity는 각도 무관 최근접 큐브를 스스로 파지함(라이브 -26°/area~8000 즉시 성공).
                             # 학습 정책이 짧은 회전으로 큐브를 중앙 정렬하지 못하므로(ramp-up) 정렬을 요구하지 않음.

# --- 회전/pad-nav 상수 (학습 locomotion 정책 실측 기반) ---
# 이 로봇의 학습 보행 정책은 제자리 yaw 스핀이 포화되어 거의 안 돕니다(실측 ~3°/s, 큰 wz는 무효).
# 방향전환은 반드시 아크(vx>0 + wz)로 합니다(실측 vx0.3/wz1.2 ≈ 34°/s). 아래 값은 그 실측에서 유도합니다.
ARC_VX = 0.25                # 아크 조향 시 전진 속도(회전을 살리기 위한 최소 vx).
ARC_WZ = 1.2                 # 아크 조향 시 yaw rate(양수=좌회전).
SWEEP_VX = 0.2               # target을 잃었을 때 걸으며 시야를 훑는 아크의 전진 속도.
SWEEP_WZ = 1.0               # 스윕 아크의 yaw rate.
# ★run14 전도 방어(구조물-근접 회전)★: recover의 회전 아크는 vx>0 전진을 동반하므로 벨트/랙에
# 붙은 채 돌면 구조물로 밀고 들어가며 전도한다(오늘 전도 6/6이 구조물 접촉 중 회전·측면 기동).
# 회전 전에 후퇴로 이탈부터 시킨다. 후퇴 vx는 아크 실효 하한(|vx|≥0.2, 기존 -0.15는 미달로 안 걸음)에 맞춘다.
RECOVER_BACKUP_VX = -0.2         # 회전 전 구조물 이탈용 후퇴 속도(실효 하한 0.2 정합).
RECOVER_BACKUP_DUR = 0.8         # 평시 후퇴 시간(≈0.27×0.8≈0.22m).
RECOVER_BACKUP_DUR_BLOCKED = 1.4 # 전방 막힘(구조물 접촉 의심) 시 후퇴 시간(≈0.38m 이탈).
FORWARD_VX = 0.4             # target이 중앙일 때 직진 속도.
VLM_MIN_CONFIDENCE = 0.5     # 이 미만 신뢰도의 sign은 무시하고 계속 탐색합니다.
# VLM 지연(6~32s/회)이 pad-nav 병목입니다. signage는 큰 글자(A~E)만 읽으면 되어 저해상도에
# 견고하므로, signage 판독 프레임을 기본(800px/q70)보다 줄여 업로드+vision 토큰을 낮춰 호출
# 시간을 단축합니다. 단 far C(작게 잡힘)를 너무 낮추면 놓칠 수 있어 중간값에서 시작 — 라이브로
# 지연 대 검출률을 측정해 조정하세요(더 내리면 빠르지만 C 놓칠 위험, compress_jpeg는 수정 가능).
SIGNAGE_VLM_MAX_WIDTH = 640  # 기본 800 → 640 (프레임 면적 ~36%↓).
SIGNAGE_VLM_QUALITY = 60     # 기본 70 → 60 (JPEG 품질↓로 payload↓).
PAD_SEARCH_TURN_DEG = 55.0   # head 스캔에서 sign을 못 찾으면 body를 이만큼 돌려 새 구역을 스캔.
PAD_TURN_TOL_DEG = 8.0       # 폐루프 회전이 목표 yaw 이 오차 안에 들면 종료.
# 라이브 실측(2026-07-04): center 프레임에서도 VLM이 target 'C'를 position=Left로 정확히 읽어,
# 파서 키 정규화 수정 후엔 center 1회로 흔히 잡힙니다. 다만 로봇이 이동하면 표지가 프레임 밖으로
# 벗어날 수 있어, center를 먼저 보고(확신 검출 시 조기 종료) 좌·우를 보조로 훑습니다 — 조기 종료가
# 있어 흔한 경우 VLM 1회, 정면에 없을 때만 측면 비용을 지불합니다(사용자 요청: 카메라 좌우 스캔).
HEAD_SCAN_YAWS_RAD = (0.0, -0.6, 0.6)
VLM_MAX_RETRIES = 2          # Tokamak fallback(상위 모델 미가용)·빈 응답 시 같은 프레임 재시도 횟수.
                             # 라이브 실측: 2회 중 1회가 fallback 문장 → '표지 없음'(conf=0.00)으로
                             # 오인돼 acquisition 붕괴(런3). 재시도로 일시적 provider 플랩을 흡수.
PAD_POS_OFFSET_DEG = 15.0    # VLM의 left/right를 대략 이 각도(도)로 환산해 body-bearing 계산.
PAD_FACE_TOL_DEG = 8.0       # 목표 방향이 이 안이면 이미 마주봤다고 보고 접근으로 넘어감.
                             # 반드시 PAD_POS_OFFSET_DEG보다 작아야 함: 같으면(둘 다 15) 'left'/'right'
                             # 검출의 face_turn=±15가 tol을 못 넘어(>15 거짓) 영영 회전하지 않음(라이브 확인).
PAD_OUTER_MAX = 18           # look→(전진/회전 or face+접근) 시도 최대 횟수. anchor goal-seek는
                             # VLM을 생략해 반복이 싸므로(~5s), 벽 통과 탐색 여유를 넉넉히 둡니다.
PAD_ADVANCE_DUR = 1.4        # 전진 한 청크의 길이(초). 시야 변경·pad 접근 공용.
PAD_FWD_BEFORE_TURN = 3      # 못 찾을 때 이 횟수-1 만큼 전진하고 매 이 횟수째에 회전.
TURN_MAX_ARCS = 5            # 폐루프 회전 한 번에서 최대 아크 명령 수(무한루프 방지).

# --- VLM 응답 정규화·비례 조향 상수 ---
# qwen 계열 VLM은 같은 프롬프트에도 응답 키 스키마가 흔들립니다(라이브 실측 5형:
# letter / sign_letter / label / text / text_content, label에는 글자 대신 'sign letter'
# 같은 서술어가 오기도 함). 아래 키 후보를 순서대로 훑되 "단일 알파벳" 값만 글자로 인정합니다.
VLM_LETTER_KEYS = ("letter", "sign_letter", "text_content", "text", "label")
# 라이브 실측: qwen이 글자를 위 키가 아니라 서술 문장에만 싣기도 함
# (label="sign", description="green square sign with white letter 'C'"). 이 키의 문장에서
# 'letter X' 뒤 글자나 단일 대문자 토큰을 별도로 추출합니다(_letter_from_phrase).
VLM_LETTER_DESC_KEYS = ("description", "desc", "caption")
VLM_DEFAULT_CONFIDENCE = 0.75  # 글자+bbox는 있는데 confidence 결측인 응답(라이브 실측)에 부여.
                               # 게이트(0.5)는 넘기되 명시적 high(0.9)보다는 낮게.
VLM_BBOX_SCALE = 1000.0        # qwen bbox_2d는 0~1000 정규화 좌표(640px 프레임 응답에서
                               # x=723 관측 → 픽셀 좌표 아님). 범위를 벗어나면 무시하고 fallback.
CAMERA_HFOV_DEG = 60.0         # perception.py의 angle 규약(±30° half-FOV)과 동일한 수평 화각.
                               # bbox 중심 x → 방위각 비례 환산에 사용(±15° 양자화 제거).

# --- 전진 stall(막힘) 감지·우회 상수 (라이브 실측 기반) ---
# fwd 0.4/1.4s의 정상 병진 ≈0.37m(실측 → 실효 속도 FORWARD_EFF_SPEED_MPS). 학습 정책은
# 구조물(source ledge 등)에 막혀도 명령을 수용해 병진 0·회전만 남고, navpad는 이를 몰라
# 제자리 배회했습니다(라이브 확정: x≈1.1 고착). 전진 전후 odometry 거리(상대량)를
# 기대 이동량(속도×시간 운동 모델)과 비교한 '이동 효율'로 stall을 감지해 우회합니다.
FORWARD_EFF_SPEED_MPS = 0.27  # FORWARD_VX=0.4의 실효 병진 속도(실측 0.37m/1.4s ≈ 0.27m/s).
STALL_EFF_RATIO = 0.3         # 실제/기대 이동 효율이 이 미만이면 stall(옛 0.11m/0.37m 게이트와 동치).
STALL_ABS_FLOOR_M = 0.04      # odometry 노이즈 하한(이 미만 병진은 어떤 기대치든 stall).
PAD_STALL_BACKUP_S = 0.8      # 선제 우회(들이받지 않음) 시 회전 공간 확보용 짧은 고정 후진 시간.
PAD_STALL_DETOUR_DEG = 50.0   # stall 시 우회 회전각. 다음 look의 VLM 재보정을 전제로 크게 꺾음.
# 충돌 후퇴(collision back-off): 선제 free-space 인지가 놓친 장애물에 실제로 부딪혀 stall(제자리
# 걸음)이 나면, 고정 시간 후진은 밀고 들어간 양과 무관해 과/부족 후퇴가 됩니다. 대신 '전진한
# 만큼' 폐루프로 되돌립니다 — 매 후진 청크의 odometry 변위를 재어 누적 후퇴가 전진 변위에 닿을
# 때까지 반복(속도 신뢰 아님, 실측 이동량 기준). 선제 우회로 애초에 안 들이받은 경우엔 적용 안 함.
STALL_REVERSE_VX = 0.15       # 후퇴 속도(기존 고정 후진과 동일 크기).
STALL_REVERSE_CHUNK_S = 0.6   # 후퇴 폐루프의 한 청크 길이(초).
STALL_REVERSE_MAX_CHUNKS = 5  # 후퇴 청크 안전 상한(무한 후진 방지).
# 측면 우회(lateral bypass): 짧은 detour로도 못 뚫는 선형 구조물(벨트 등)에 반복해 막히면,
# 표지 재조준을 잠시 멈추고 목표 쪽으로 ~90° 꺾어 여러 청크를 '따라 이동'해 구조물의 끝/틈을
# 지나갑니다(표준 bug-following, 카메라·odometry만). 라이브 확정: pad가 벨트 너머라 직진
# 접근만으론 x≈1.1에서 영구 고착 -> R6 비수렴 신호와 함께 escalate.
PAD_BYPASS_STALL_TRIGGER = 2  # 연속 hard-stall(직진+detour 모두 실패)이 이만큼이면 측면 우회 발동.
PAD_BYPASS_TURN_DEG = 80.0    # 측면 우회 시 목표 쪽으로 꺾는 각(구조물과 대략 평행하게 이동).
PAD_BYPASS_MAX_CHUNKS = 4     # 한 번의 측면 우회에서 따라 이동할 최대 전진 청크 수(escalate 상한).
BELT_FOLLOW_CHUNKS = 6        # ★사용자 제안(능동 벨트 우회)★: 벨트를 카메라로 선제 식별해 우회할 때
                              # 따라 이동할 청크 수. 반응형 4청크보다 길게 커밋 — 램→wedge 전에 꺾어
                              # free-space 쪽으로 벨트 끝/틈까지 따라갑니다(run3: 반응형 우회가 wedge 후
                              # 양측막힘 4/6 실패). stall이면 _lateral_bypass가 조기 종료(반대쪽 재시도).
STALL_SPOT_RADIUS_M = 0.45    # 기억된 stall 지점 반경 — 이 안에서 같은 방향 전진이면 선제 우회.
STALL_HEADING_TOL_DEG = 60.0  # stall 지점의 '같은 방향' 판정 폭(도). 다른 방향 접근은 허용.

# --- 경로 기억(route memory)·last_seen·VLM 절감 상수 ---
# 성공한 배송 경로를 점수화해 저장하고 다음 같은 색 배송에서 greedy 재사용하는 online
# heuristic 최적화입니다(엄밀한 RL 학습이 아니라 성공/실패 경험 축적형 탐욕 선택).
# 모든 좌표는 로봇 자신의 odometry(고유수용성) 기준 — scene_state가 아니며, 한 run
# 안에서만 유효합니다(프로세스 간 영속 캐시 없음 = 특정 setup hardcoding 아님).
ADVANCE_MIN_S = 0.7            # 램프업 미달로 거의 안 걷는 초단시간 전진 명령 방지 하한.
WAYPOINT_TOL_M = 0.4           # waypoint 도달 판정 반경(유클리드 거리).
ROUTE_REPLAY_CHUNKS_PER_WP = 4 # waypoint당 최대 전진 청크 수(초과 시 replay 중단).
ROUTE_MIN_WAYPOINT_GAP_M = 0.5 # waypoint 압축 최소 간격(기록 폭증·미세 진동 방지).
LAST_SEEN_MAX_REUSE = 2        # last_seen 연속 재조준 상한(초과 시 VLM 재확인 강제).
LAST_SEEN_MAX_DRIFT_M = 2.5    # 목격 pose에서 이만큼 멀어지면 ray(방향) 가정을 불신.
ROUTE_SCORE_VLM_W = 10.0       # 경로 점수: score = time + vlm*10 + stall*5 + path*1.5.
ROUTE_SCORE_STALL_W = 5.0      #   (낮을수록 좋음. VLM 호출이 지배 비용이라 가장 무겁게.)
ROUTE_SCORE_PATH_W = 1.5
FAILED_ROUTES_KEEP = 3         # 실패 경로 기록 보관 상한(진단용).
DETOUR_WIN_KEEP = 20           # 성공 우회 방향 기록 보관 상한.

# --- R6: 접근 수렴 판정(변화율 proxy) 상수 — 현재 '관측 모드'(로그·기록만, 행동 불변) ---
# 단안이라 pad 거리 d를 직접 못 재므로, 원근 투영의 면적 ∝ 1/d² 관계로 색블롭 면적의
# 변화율을 거리 변화율의 단조 대용으로 씁니다. 라이브(R4)에서 임계를 보정한 뒤에만
# 행동(전략 변경) 트리거로 승격합니다.
APPROACH_MIN_SAMPLES = 4         # 수렴 판정 최소 표본 수(전/후반 중앙값 비교가 성립하는 최소).
APPROACH_AREA_GROWTH_MIN = 1.15  # 후반 면적 중앙값이 전반의 이 배수 이상이면 '접근 중'.
                                 # 면적 ∝ 1/d²: 5m 거리에서 한 청크(≈0.37m) 전진 시 (5/4.63)² ≈ 1.17
                                 # — 먼 거리에서도 청크당 증가분을 감지하는 하한에서 유도.

# --- pad anchor(목격 기반 대략적 위치 기억) 상수 ---
# last_seen ray(방향만)의 한계: 목격 pose에서 벗어나거나 몸이 많이 회전하면 재조준각이
# ±160° 같은 발산 회전을 낳습니다(라이브 확정: x=2.63까지 진출 후 스핀→전도). bbox 면적으로
# 거리까지 추정해 sign을 world '점'으로 기억하면 재조준각을 매 반복 '현재 pose'에서 새로
# 계산하므로(폐루프) 회전·이동으로 낡지 않습니다 — 사람이 목적지를 한 번 보면 대략적 위치를
# 기억해 두고, 그 뒤로는 안 보여도 장애물을 우회하며 그 방향으로 가는 방식(고전 Bug 알고리즘의
# goal + 국소 회피)과 같습니다. 점 좌표는 카메라 관찰(방위+크기)을 자기 odometry 프레임에
# 투영한 학생 추정치일 뿐 scene_state가 아닙니다(Level 2 합법).
PAD_SIGN_DIST_K = 0.34          # d ≈ K/√area_frac(면적 ∝ 1/d²의 역산). 라이브 실측 far-C bbox
                                # (area_frac≈0.0093, 당시 pad까지 ~3.5m 추정)에서 유도한 잠정값.
                                # 과소추정(짧게 멈춤→가까운 재목격으로 자가보정)이 과대추정(구조물
                                # 돌진)보다 안전해 낮은 쪽을 택함 — 라이브 로그의 d 추정치로 보정하세요.
PAD_ANCHOR_MIN_D = 0.8          # 거리 추정 하한(과대 bbox가 0m 근처 추정을 내는 것 방지).
PAD_ANCHOR_MAX_D = 6.0          # 거리 추정 상한(원거리 미소 bbox의 노이즈 폭주 방지).
PAD_ANCHOR_MIN_CONF = 0.6       # anchor 융합 신뢰도 게이트 — nav 게이트(0.5)보다 엄격하게 잡아
                                # 낮은 확신의 오검출(A를 C로 오독 등)이 점 추정을 오염시키지 않게 함.
PAD_ANCHOR_OUTLIER_M = 2.0      # n≥2로 자리잡은 anchor 평균에서 이 이상 벗어난 새 목격은 기각
                                # (한 번의 오독이 평균을 끌고 가는 것 방지; Mahalanobis-lite).
PAD_ANCHOR_NEAR_M = 1.2         # anchor 근접 반경 — 이 안이면 goal-seek 대신 VLM 확인으로 전환
                                # (점 추정 오차 안이므로 더 걸어봤자 목표를 지나칠 뿐).
PAD_ANCHOR_MAX_REUSE = 5        # anchor 연속 goal-seek 상한 — 초과 시 VLM 재확인(오염 방지).
PAD_ANCHOR_NEAR_MISS_LIMIT = 3  # anchor 근접 VLM 연속 미검출 → anchor 폐기(오염 자가치유).
PAD_ANCHOR_MAX_REAIM_DEG = 90.0  # 한 반복의 anchor 재조준각 상한 — 점 기반이라 발산하진 않지만
                                 # 큰 후방 회전을 두 반복에 나눠(회전→전진→재평가) 전도 위험을 줄임.
PAD_ANCHOR_W_CAP = 6.0          # 융합 가중치 누적 상한 — 옛 목격 더미가 새(더 가까운) 목격을
                                # 압도하지 못하게 해 잘못 초기화된 anchor도 재목격으로 씻겨나감.
PAD_PLACE_NEAR_M = 1.0          # 초근접 도착 판정: anchor 추정 거리가 이 안이고 sign을 VLM으로
                                # 정면 확인했는데 전진이 막혀 더 못 다가가면, 색블롭 도착 크기
                                # (≥PAD_ARRIVAL_AREA)를 못 채워도 도착으로 봅니다 — 초근접에서
                                # 바닥 pad가 카메라 하단 밖으로 나가 blob=0이 되는 것을 place
                                # 실패로 오판하던 결함 해소(라이브 확정: (2.69,1.53)에서 'C'
                                # center conf 0.99인데 area=0으로 18 attempt 소진 후 전도).

# --- pad 지도(survey-first) 상수: 스폰 서베이 + ray 삼각측량 목표 동결 ---
# 라이브 확정(2026-07-04): 단안 bbox 거리 추정이 같은 C 표지에 2.9/3.4/6.0m로 요동쳐, 원거리
# 오추정 목격 1회가 anchor를 (+3.44,+0.01)→(+5.15,-1.68)로 끌고 가 로봇을 랙 미로로 보내
# 전도시켰습니다. 대책: (1) 런 시작 시 아크 회전 서베이로 보이는 모든 표지의 관측 ray를 지도에
# 부트스트랩하고(후보-게이트형이라 표지가 안 보이는 스폰이면 VLM 0회, ~20s로 끝남 — setup 1~50
# 랜덤 스폰에도 비용 자동 조절), (2) 기선·교각이 충분한 두 ray의 교점(삼각측량)으로 pad world
# 좌표를 '동결'해 이후 bbox 거리 노이즈가 목표를 끌고 다니지 못하게 합니다. 전부 자기 odometry
# + 카메라/VLM 유도이며 좌표 하드코딩이 없습니다(Level 2 합법, 히든 setup 대응).
SURVEY_STEPS = 8              # 아크 회전 서베이 방위 수(45°씩 한 바퀴).
SURVEY_STEP_DEG = 45.0        # 서베이 스텝 회전각. FOV 60°와 겹쳐 사각 없이 커버.
SURVEY_VLM_MAX_CALLS = 3      # 서베이 VLM 판독 상한 — 시간 예산 보호(후보 방위에서만 호출).
SURVEY_CAND_MIN_AREA = 2500   # 표지 후보 blob 최소 면적(far-C 실측 8334의 여유 하한).
SURVEY_CAND_MIN_FILL = 0.55   # 표지는 단색 사각(실측 fill 93~94%) — 랙 슬래브(fill<50%) 배제.
SURVEY_CAND_MAX_CY_FRAC = 0.72  # 표지는 공중 부양(수평선 부근) — 바닥 큐브(cy_frac≈0.88+) 배제.
SURVEY_CAND_ASPECT = (0.5, 2.0)  # 표지는 대략 정사각 — 가로로 긴 레일/랙 밴드 배제.
SURVEY_DEDUPE_BEARING_DEG = 18.0  # 이미 판독한 world 방위 ±이 각 안의 후보는 재판독 생략.
TRI_MIN_BASELINE_M = 1.2      # 삼각측량 최소 기선 — 짧으면 방위 노이즈가 거리 오차로 폭주.
TRI_MIN_ANGLE_DEG = 12.0      # 두 ray 최소 교각 — 준평행 ray의 불안정 교점 배제.
TRI_MAX_RANGE_M = 12.0        # 교점까지 거리 상한(창고 규모 밖 교점은 오독으로 기각).
SIGN_RAYS_KEEP = 12           # letter별 보관 ray 수 상한.
# --- M1: 표지 목표 오염 차단(클러스터 합의 동결) + VLM 조기 포기 상수 ---
# 동일 문자 표지가 다수 실재하고('B'≥2~3개, 'D' 바닥 팔레트 데코이) 삼각측량 쌍의 재투영 잔차는
# 구성상 0이라(오염 쌍도 0), 오염 분리는 '다중 쌍 교점의 클러스터 합의'로만 가능합니다: 같은 실물을
# 겨눈 쌍들의 교점은 밀집하고, 실물을 섞은 쌍의 교점은 군집에서 이탈합니다(spec §2-3).
SIGN_CLUSTER_MAX_RADIUS_M = 0.8   # 잠정(M3 보정): 교점 군집 반경. 오염 이탈 실측 1.6m·place 반경
                                  # 1.2m보다 작게, 3° 방위 노이즈 산개(~1.4m 꼬리)와 절충(유효 창 협소, R11).
SIGN_CLUSTER_TIE_RATIO = 0.5      # 잠정(M3 보정): 2위 군집이 1위의 이 비율 이상이면 다수 미확정 →
                                  # 동결 보류(freeze_hold). 데코이 4:2 시나리오에서 실타깃 군집 방어(다수결 함정 G2).
SIGN_REFREEZE_COOLDOWN_LOOKS = 3  # 잠정(M3 보정): 동결 폐기 직후 같은 데코이 재응집 차단 카운터.
                                  # PAD_ANCHOR_NEAR_MISS_LIMIT(3)과 동형. 감소 단위 = freeze 시도(유효 ray 추가)
                                  # 이지 VLM look 총수 아님 — 플랩으로 ray가 안 쌓이면 실시간 쿨다운은 길어지며
                                  # 이는 안전측입니다(상수명 _LOOKS를 'look 총수'로 오독 금지). 좌표 blacklist
                                  # 대안은 odometry drift 누적으로 기각·백로그.
PAD_VLM_ABANDON_N = 2             # 잠정(M1 라이브~M3): navpad 탐색 look 연속 빈응답 임계. 1회는 우연
                                  # (세션 내 회복 실측), 2연속(≥170~380s)이면 파동 → round1(300s) 과반 소각.
                                  # 루프 헤드 승격으로 anchor 존재 시에만 탐색 look 잠금(confirm look·anchor
                                  # 근접(≤1.2m) look은 예외 — 오배송 방지·자가치유 신호원 유지).
# --- M4: 잔여거리 도착 게이트 + free-space orbit 상수 ---
# 접근 면(approach-face) 문제: pad_B place 원은 남/동 통로에서만 도달 가능해 북측 정면 접근으로는
# 목표를 완벽히 동결해도 반경 진입이 불가능합니다. 면적 단독 도착 판정에 '동결 목표 잔여거리 AND'를
# 얹어 오도착을 막고, 반복 실패 시 폐기 경로로 degrade하며, 선형 우회로도 못 뚫으면 orbit으로 다른 면을
# 찾습니다. ★M3 실측(pad_B 남/동 실주행) 전제 상수는 잠정값이며 E2E 중 보정합니다 — 사용자 승인 하에
# M3 데이터 게이트 미완 상태로 잠정 진행(§4 hard gate override).★
PAD_ARRIVAL_RESIDUAL_M = 1.6      # ★잠정(M3 미측정 — E2E 보정)★: 도착 선언 잔여거리 상한. 제약: ≥
                                 # PLACE_PROBE_START_M(1.35) — 도착 게이트가 place-probe 진입 게이트(1.35)보다
                                 # 작으면 도착하고도 probe 미충족으로 헛돕니다(직렬 이중 벽). 라이브 실측
                                 # 도착 1.47m를 포함하도록 1.6 채택(place 반경 1.2m + 접근 여유).
PAD_ARRIVAL_RESIDUAL_FAIL_N = 3  # 잠정: area_ok인데 잔여 초과가 이 횟수 반복되면 도달 불가 오염 goal로
                                 # 판단해 폐기(_drop_sign_map degrade). PAD_ANCHOR_NEAR_MISS_LIMIT(3) 동형.
                                 # ★폐기 경로와 잔여 AND는 같은 diff — 분리 반입 금지(G4 무한 grind 방지).★
ORBIT_TRIGGER_BYPASS_ROUNDS = 2  # 잠정: 측면 우회(2·3·4청크)가 이 횟수 발동해도 미해결이면 선형 우회
                                 # 무력(코너 웨지) → orbit 발동. orbit는 nav당 1회.
ORBIT_MAX_CHUNKS = 10            # ★잠정(M3 미측정 — pad_B 둘레 실주행 ÷ ~0.37m/청크로 보정)★: orbit 순회
                                 # 청크 상한(~3.7m 이동). 소진 시 무조건 포기(전도 방지).
ORBIT_STALL_GIVEUP = 2           # 잠정: orbit 순회도 연속 stall이면 무의미 → 중단(더 밀지 않음).
ORBIT_OPEN_MIN_GAIN_M = 0.1      # 잠정: 개방 검사에서 goal 재조준 후 전진이 이만큼 가까워져야 '열림'.
                                 # 단순 '이동'은 detour로 옆으로 새도 True라 false-open이 나므로, goal
                                 # 잔여 d의 실제 감소로 판정(§OQ-C 근사 강화, 자체검토 반영). E2E 보정.
SURVEY_MAX_WALL_S = 90.0      # 서베이 실시간(wall-clock) '소프트' 캡. 서베이는 라운드 타이머 밖(타이머는
                             # 첫 cycle에서 시작)이라 예산 보호가 아니라 '첫 pick 착수 지연·VLM 쿼터
                             # 소모 방지'가 목적. 매 스텝 착수 전 검사해 초과 시 새 스텝을 시작하지 않습니다
                             # (spec §5.1 "조기 종료"). 이미 시작된 판독은 완료되므로 최악 오버슈트 ≈ 진행
                             # 중인 signage 판독 1회분입니다(단, VLM 판독은 SURVEY_VLM_MAX_CALLS=3회로 이미
                             # 하드 캡되어 총량이 유계). 인flight 판독까지 끊으려면 ask_vlm(동기 호출)을
                             # asyncio 스레드+취소로 감싸야 해 라이브 하드닝으로 이관합니다(Codex MAJOR 3).
# --- source(A)-seek(랜덤 스폰 획득) 상수 ---
# setup 1~50은 랜덤 스폰이라 clean cube가 시야에 없을 수 있습니다. 소스(A 표지/컨베이어)를 국소화·
# 접근해 'clean cube가 보이는 상태'를 만들어 기존 획득 흐름에 넘깁니다(신규 action 없이 기존
# search_cube 실행부 강화). 종료는 'clean cube 가시화' — 먼 큐브 조기 종료는 의도된 정상 동작이고
# 접근은 navigate_to_cube 소관입니다. 전부 카메라+자기 odometry 유도, source-seek 자체는 VLM 0회.
SRC_SEEK_MAX_CHUNKS = 4        # source-seek 턴당 전진 청크 상한(각 PAD_ADVANCE_DUR=1.4s). 매 턴 LLM 복귀.
SRC_FALLBACK_ADVANCE_M = 2.0   # 소스 단서 전무(cube/goal/ray 모두 없음) 시 열린 방향으로 전진할 거리(m).
SRC_FALLBACK_MAX_ROUNDS = 3    # 폴백(열린 방향 전진+무료 재스윕) 누계 캡 — 초과 시 기존 아크-스윕에 위임.
CUBE_SIGHTINGS_KEEP = 8        # cube_sightings 보관 상한(오래된 것부터 폐기, _add_sign_ray와 동형).
# 마무리 접근(place 반경 진입): 표지 blob은 크고 높이 떠 있어 1.4~1.5m 밖에서도 도착 면적
# (PAD_ARRIVAL_AREA)을 채웁니다 — 라이브 확정: blob 49k·'D' conf 0.98로 도착 선언한 지점이
# 동결 목표에서 1.47m라 place('팔레트 1.20m 이내' 요구)가 실패. 도착 선언 후 동결 목표까지
# 남은 거리를 소회전+짧은 전진으로만 마저 좁힙니다(협소 구역 큰 아크는 전도 위험 — 라이브에서
# 수동 3s 아크 직후 전도). stall이면 팔레트에 닿은 것이므로 더 밀지 않고 place를 시도합니다.
PAD_CLOSE_ENOUGH_M = 1.05     # 동결 목표까지 이 거리면 place 반경(1.2m) 안으로 판단.
PAD_CLOSE_MAX_CHUNKS = 3      # 마무리 접근 전진 청크 상한.
PAD_CLOSE_CHUNK_S = 0.8       # 마무리 전진 한 청크 길이(짧게 — 팔레트 충돌 완충).
PAD_CLOSE_MAX_TURN_DEG = 40.0 # 마무리 재조준 회전 상한(협소 구역 대회전 금지).
# --- §1(핸드오프): pick/place 반경(1.2m) 진입 push-through — 좌표 0, blob 서보 + odometry ---
# 세션5 정찰 실측: 표지 blob 도착 면적은 1.34~1.47m 밖에서도 충족(area_ok)이나 그 지점은 palette
# place 반경(1.2m) 밖이라 place가 헛돕니다. 약추진(vx≤0.4)은 place-zone 바닥 빈 팔레트에 1.34m에서
# stall, vx0.5 강추진이 0.87m(원 내부) upright 돌파. 동결 goal이 없으면(run1) 잔여거리 게이트·
# place-probe·_close_to_goal이 전부 우회돼 1.34m 단일 place→0배달. push-through는 goal 없이 pad
# 색블롭을 서보하며 vx를 단계 상향해 반경으로 진입합니다(모든 pad 공용 — pad별 방위 하드코딩 금지 §7).
PAD_PUSH_VX_LADDER = (0.5, 0.6)  # ★잠정(E2E 보정)★: 정찰 vx0.5 돌파. stall이면 다음 단계로 상향.
PAD_PUSH_MAX_CHUNKS = 4          # ★잠정★: push-through 전진 청크 상한(전도 방지·직렬 1개).
PAD_PUSH_MAX_ADVANCE_M = 0.7     # ★잠정★: 누적 전진 상한(1.34m→~0.6m). 팔레트 없는 pad 오버슈트 차단.
# ★J(run15 전도)★ push 서보 blob이 이 면적을 넘으면 카메라가 구조물(랙/벽)에 밀착됐다는 뜻 —
# 실제 pad blob은 근접(≤0.9m)에서도 ~55k에 그침(라이브 실측 40k~55k). run15는 벽 밀착 213k에서
# vx0.5→0.6로 stall grind 2청크 후 전도(전형적 구조물-접촉 grind 전도). flooding이면 밀지 않고 즉시
# 중단해 grind torque 전도를 차단한다(Fix H의 정상 근접 오버사이즈 추격 40~55k은 그대로 통과).
PAD_PUSH_FLOOD_AREA = 130000     # ★J★ 서보 blob ≥ 이 값이면 구조물 밀착(flooding) → push 즉시 중단.
PICK_RETRY_MAX_DIST_M = 3.0      # ★D1b(run6)★ pick 실패 실거리가 이 이하일 때만 거리-피드백 재전진.
                                 # 그보다 멀면 쫓던 blob이 큐브가 아닐 공산(구조물) — 전진하지 않음.
PICK_RETRY_TARGET_M = 1.0        # 재전진 후 목표 잔거리(1.2m pick 반경 안쪽 여유).
PICK_RETRY_MAX_CHUNKS = 8        # 거리-피드백 재전진의 청크 상한(~0.3m/청크 × 8 ≈ 최대 2.4m 커버).
PICK_MIN_CYCLE_ADVANCE_M = 0.05  # ★E2(run7 마비)★ 한 pick cycle 전진이 이 미만 + too-far 실패면
                                 # '이 자리에선 불가'로 보고 재시도 대신 recover(재배치)로 끊습니다.

# ── Option A(데모 단순화, 2026-07-06 사용자 승인): 미검증 경로 봉인 + 자기완결 place ──
# 8런 트레이스 근거: freeze_commit 0/8, place-probe 진입 0/8, orbit 트리거 0/8. 라이브에서 한 번도
# 밟히지 않은 경로가 데모 중 처음 발화하는 것 자체가 최대 비결정성이므로, 삭제 대신 스위치로
# 잠급니다(1줄 복원 가능·회귀 없음). 순수 함수와 pytest는 그대로 두고 콜사이트만 게이트합니다.
SIGN_FREEZE_ENABLED = False   # 표지 삼각측량 동결 커밋(M1). run8은 동결 없이 anchor 조향으로 도달.
PLACE_PROBE_ENABLED = False   # place-probe 분기 — 동결 goal 전제라 라이브 진입 0회.
ORBIT_ENABLED = False         # M4 궤도 우회 — M3 미보정 ★잠정★ 상수뿐, 트리거 0회.
PAD_ARRIVED_STICKY_M = 0.8    # A2(자기완결 place): navpad 도착 뒤 이동이 이 이하면 도착 유효로 보고
                              # navpad 재실행(도착 재증명 + 재확인 VLM 90~144s/사이클, run8)을 생략.
PICK_BELT_BYPASS_ENABLED = True  # A3(run9): pick이 '전진 0' 차단으로 연속 실패하면 국소 recover 대신
PICK_BYPASS_AFTER_N = 2          # navpad의 능동 벨트 우회와 동형으로 벨트를 따라 건너갑니다
                                 # (run9: 후퇴+회전 recover ×14로는 520s 내내 벨트 못 건넘).
PAD_ARRIVAL_ANCHOR_MAX_M = 1.8   # ★I(run12 전도)★ 도착 선언·진입 push 전 anchor 잔여 sanity.
                                 # area 게이트는 같은 색 거대 blob에 속습니다 — blue 배달에서 벨트
                                 # (blue)가 area 30886로 '도착' 판정됐는데 anchor는 4.4m라 반대
                                 # 증언(run12) → F(무가드)+H(무상한) push가 벨트를 갈다 전도.
                                 # anchor가 이보다 멀면 도착 거부·재관찰(run8 정상 도착 1.65~1.77m
                                 # 는 통과). anchor 미형성(None)이면 게이트 생략(기존 동작).
# --- place-probe(첫 배달 갭 해소) 상수 ---
# 마무리 접근(_close_to_goal)이 동결 목표 1.05m까지 좁혀도, 표지 blob 도착 지점(라이브 확정 1.47m)과
# palette place 반경(1.20m) 사이 갭이 남을 수 있습니다. place-probe는 place를 반복 시도하되 성공
# 판정을 '오직 delivered 증가'로 하고(status 불신 — pad 부재 시 status=done인데 큐브가 바닥에 떨어져
# delivered 불변인 라이브 2회), 결정적으로 반복 실패하면 lateral 탈출 1회로 접근각을 바꿉니다.
PLACE_PROBE_START_M = 1.35     # 진입 게이트 잔여 거리 상한(= place 반경 1.20 + 접근 청크 1회분 여유 0.15).
PLACE_PROBE_MAX_TRIES = 4      # place 루프 최대 반복(각 반복: delivered 읽기 → place → nudge).
PLACE_PROBE_STEP_S = 0.8       # nudge 전진 시간. ADVANCE_MIN_S(0.7) 위 — 0.5s는 램프업 미달로 물리적
                             # 으로 안 걷습니다(라이브 확정). 이 하한 없이는 nudge 병진이 0이 됩니다.
PLACE_PROBE_MAX_WALL_S = 8.0   # place 루프 wall-clock 캡. place 호출 '사이'에서만 체크 — 초과 시 종료가
                             # 아니라 lateral 전환점(캡=탈출구 진입 신호). place 즉시성(~1ms) 전제.
LATERAL_MAX_WALL_S = 5.0       # lateral 국면의 '분리 계상 예산'(설계 목표) — place 루프 8s와 따로 계상해
                             # place가 8s를 다 써도 lateral이 반드시 1회 실행되게 하는 개념적 분리이지,
                             # 재조준을 막는 런타임 게이트가 아닙니다(하위 단계가 자기유계라 구조적으로 유한).
                             # 실측 lateral 왕복(80° 회전 2회+2청크≈6-8s)은 이 값을 넘을 수 있으나 재조준은
                             # 필수라 건너뛰지 않고, lat_elapsed는 관측 로그로만 씁니다(§8-10, Codex 재검토).

# --- 선제 장애물 인지(바닥 free-space) 상수 — get_vision 1프레임 밝기(V)로 발 앞 바닥 판정 ---
# 반응형(odometry stall)은 '부딪힌 뒤'에만 작동해 새 구조물마다 헛돌격 1청크를 버립니다. 바닥은
# 어둡고(반사 광택의 짙은 청회색, 실측 V≈55) 구조물(회색 ledge·밝은 시안 레일·표지, 실측 V≈100+)
# 은 밝다는 라이브 관찰을 이용해, 전진 직전 프레임 하단(발 앞) 밝기로 정면 막힘을 선제 판정합니다.
# floor_ref를 매 프레임 발밑 strip에서 percentile로 뽑아 자기보정하므로 색 하드코딩이 없고(히든
# 평가 조명 변화에 견딤), scene_state를 전혀 안 쓰는 카메라 기반 판정이라 Level 2에서 합법입니다.
FREE_FEET_TOP_FRAC = 0.85   # 발 앞 strip = 프레임 하단 15%(로봇 바로 앞 지면).
FREE_FLOOR_PCT = 20.0       # strip의 이 백분위 밝기를 '바닥' 기준으로(어두운 바닥이 항상 일부 존재).
FREE_FLOOR_MARGIN = 40.0    # 바닥 기준보다 이만큼(V) 밝으면 '바닥 아님'(구조물). 실측 바닥/구조물
                            # 밝기 간극(≈55 vs ≈100+)에서 유도.
FREE_CENTER_LO = 0.35       # 정면 판정 밴드(발 앞 중앙 x 범위 시작).
FREE_CENTER_HI = 0.65       # 정면 판정 밴드 끝.
FREE_SIDE_LO = 0.05         # 좌/우 여유 비교 밴드 안쪽 경계(양끝 대칭: 좌=[LO,HI], 우=[1-HI,1-LO]).
FREE_SIDE_HI = 0.35         # 좌/우 여유 비교 밴드 바깥 경계.
FREE_BLOCK_FRAC = 0.40      # 중앙 발밑의 '바닥 아님' 비율이 이 이상이면 정면 막힘. 실측: 접근 가능
                            # (0.27) < 0.40 < 막힘(0.45~0.72) — 오차단(멀쩡한 접근 차단)이 더
                            # 해로우므로 clear 여유를 크게 두는 값.
FREE_SIDE_MARGIN = 0.15     # 좌/우 막힘 비율차가 이 이상일 때만 더 열린 쪽을 우회 방향으로 채택.

# --- sector map(시각 장애물 지도, M2) 상수 ---
# egocentric 8섹터(각 45°, 정면=중앙) "가장 가까운 랙까지 거리 m" 배열로 원거리 진입을 억제하고
# 우회 방향을 고릅니다. 지면 투영 d=ground_a/(y_bottom_frac-ground_b)는 pitch=HEAD_PITCH_TRACK 전제의 카메라
# '광학 상수'이지 씬 좌표가 아닙니다(setup이 바뀌어도 카메라 내부 파라미터는 불변 → 하드코딩 아님).
# ⚠ RACK_GROUND_A/B는 잠정값입니다 — 라이브 S1 sim에서 알려진 거리 3점(≈1.0/2.0/3.5m, odometry
# 실측)의 y_bottom_frac을 2미지수 최소제곱으로 적합하고 4번째 점(≈2.7m) 예측 오차 ≤0.5m로 검증한 뒤
# 확정해야 합니다(§5.7 보정 프로토콜). 미보정 상태에서도 랙 blob 0이면 전 섹터 inf로 안전 degrade하고
# 발밑 free_space_profile이 최종 방어라 오보정이 치명적이지 않습니다.
RACK_GROUND_A = 0.6           # 지면 투영 분자(잠정 — 라이브 최소제곱 보정 필요).
RACK_GROUND_B = 0.4           # 지면 투영 지평선 오프셋(잠정 — y_bottom_frac이 이보다 커야 전방 유효 거리).
RACK_BLOCK_NEAR_M = 1.8       # navpad 전진 전 목표 방위 섹터 랙이 이보다 가까우면 선제 우회.
RACK_SIDE_MARGIN_M = 0.6      # 좌/우 섹터 거리차가 이 이상이어야 더 열린 쪽을 freer_side로 채택.
RACK_STAGE_DIST_M = 1.5       # staging waypoint 경유점 거리(목표 막히고 인접 섹터 열림 시).
# ⚠ sector map 활성 게이트. RACK_GROUND_A/B가 라이브 미보정인 동안은 False로 두어 순수 함수·배선을
# 완비하되 live-verified 항법에는 영향을 주지 않습니다(미보정 투영의 오억제 방지). 라이브 S1에서
# 3+1점 지면 투영 보정(§5.7)을 마치고 RACK_GROUND_A/B를 확정한 뒤 True로 활성화하세요.
RACK_SECTOR_MAP_ENABLED = False


def _frame_width_from(detection: Any) -> float | None:
    """detection의 centroid.x와 angle_deg로 카메라 프레임 폭(px)을 역산합니다.

    perception은 angle_deg = (cx - W/2)/(W/2)*HFOV_HALF_DEG(30) 로 각도를 계산하므로,
    이를 뒤집으면 W = 60*cx/(angle_deg + 30) 입니다. 해상도를 하드코딩하지 않고
    관찰값에서 유도하므로 시작 포즈가 바뀌는 히든 평가에서도 성립합니다.
    """
    cx = detection.centroid[0]
    denom = detection.angle_deg + 30.0  # == HFOV_HALF_DEG
    if denom <= 0 or cx <= 0:
        return None
    return 60.0 * cx / denom


def _plausible_target(detection: Any, arrival_area: int) -> bool:
    """color blob이 '진짜 cube/pad'로 보이는지 검사합니다(카메라 기반, scene_state 미사용).

    네 가지 물리 prior로 환경 노이즈를 배제합니다:
    - 면적 하한: MIN_TARGET_AREA 미만은 noise.
    - 면적 상한: 단일 target은 도착 크기의 몇 배를 넘길 수 없음 → 화면을 뒤덮는
      컨베이어 파란 레일/바닥 같은 초대형 blob을 배제.
    - aspect(가로/세로): cube/pad는 대략 정사각. 길고 얇은 레일은 배제.
    - 폭 비율: cube/pad는 프레임 폭의 일부만 차지. 프레임을 가로지르는 레일 밴드를 배제
      (정사각으로 잡혀 aspect·면적을 통과하는 레일 조각까지 걸러냄).
    """
    if detection.blob_area < MIN_TARGET_AREA:
        return False
    if detection.blob_area > arrival_area * MAX_AREA_ARRIVAL_MULT:
        return False
    _, _, width, height = detection.bbox
    if height <= 0:
        return False
    aspect = width / height
    if not (MIN_TARGET_ASPECT <= aspect <= MAX_TARGET_ASPECT):
        return False
    frame_width = _frame_width_from(detection)
    if frame_width is not None and width / frame_width > MAX_TARGET_WIDTH_FRAC:
        return False
    return True


def _fill_ratio(detection: Any) -> float:
    """blob_area / bbox 넓이. 꽉 찬 정사각 큐브면은 1에 가깝고, 벨트/반사/구조물은 낮습니다."""
    _, _, width, height = detection.bbox
    if width <= 0 or height <= 0:
        return 0.0
    return detection.blob_area / (width * height)


def _is_clean_cube(detection: Any, arrival_area: int) -> bool:
    """detection이 '고립된 단단한 큐브면'으로 보이는지 검사합니다(색 무관).

    _plausible_target(노이즈/레일 배제)에 더해 두 물리 prior를 요구합니다:
    - fill(면적/bbox) >= CLEAN_FILL_MIN: 큐브면은 bbox를 꽉 채움. 벨트/반사는 성깁니다.
    - aspect가 거의 정사각(CLEAN_ASPECT_MIN..MAX): 큐브면은 ~1.0, 가로로 긴 레일은 1.3~1.5.
    """
    if not _plausible_target(detection, arrival_area):
        return False
    # ★D1a(run6 전도)★ 큐브 전용 면적 상한: _plausible_target의 레일 상한(×4.0)은 벨트 '세그먼트'
    # (정사각·고fill, area 33772)를 통과시켰고, 그걸 clean 큐브로 믿은 pick push가 벨트에 올라
    # 전도했음. 큐브는 pick 반경(1.2m)에서도 area ~8k이므로 ×2.5(22500)면 실큐브 손실 없이
    # 구조물/병합 덩어리만 걸러짐.
    if detection.blob_area > arrival_area * CLEAN_CUBE_MAX_MULT:
        return False
    _, _, width, height = detection.bbox
    if height <= 0:
        return False
    aspect = width / height
    if not (CLEAN_ASPECT_MIN <= aspect <= CLEAN_ASPECT_MAX):
        return False
    return _fill_ratio(detection) >= CLEAN_FILL_MIN


def _detect_belt_color(detections: list[Any]) -> str | None:
    """프레임을 뒤덮는 초대형 수평 구조물(컨베이어 벨트/레일)의 색을 런타임에 추정합니다.

    벨트는 큐브 도착 크기의 몇 배를 넘는 가로로 긴 blob으로 나타납니다. 이 색은 '제외'가
    아니라 '후순위'로만 쓰므로(획득 시 다른 색을 먼저 치움) 오탐이 나도 치명적이지 않습니다.
    특정 씬 좌표가 아니라 '거대·수평' 물리 성질에서 유도하므로 히든 평가에서도 성립합니다.
    """
    oversized = [
        d for d in detections
        if d.blob_area > CUBE_ARRIVAL_AREA * MAX_AREA_ARRIVAL_MULT
        and d.bbox[2] >= d.bbox[3]  # 가로 >= 세로 (수평 밴드)
    ]
    if not oversized:
        return None
    return max(oversized, key=lambda d: d.blob_area).color


BELT_COLOR_SWITCH_N = 3  # ★run14 플랩 방어★: sticky belt_color를 다른 색으로 교체하려면 그 새 색이
                         #  연속 이만큼 관측돼야 함(단발 노란-랙 프레임은 무시). 벨트 색은 run 내 불변.


def _resolve_belt_color(
    current: str | None,
    detected: str | None,
    challenge_color: str | None,
    challenge_count: int,
    switch_n: int = BELT_COLOR_SWITCH_N,
) -> tuple[str | None, str | None, int, str]:
    """belt_color를 sticky하게 갱신합니다(run14 blue↔yellow 4:4 플랩 방어).

    _detect_belt_color는 매 프레임 max-area blob 색을 돌려주므로, 노란 랙이 이기는 프레임에서
    파랑-벨트 방어(D2 가드/선제 우회/A3)가 통째로 꺼진다. 벨트 색은 한 run 동안 불변이므로 최초
    확정색을 sticky로 잡고, 다른 색은 '연속 switch_n회 동일' 관측돼야만 교체한다(최초 확정이 오탐일
    가능성에 대비한 안전밸브). 관측이 없는(None) 프레임은 중립이라 상태를 보존한다.

    반환 (belt_color, challenge_color, challenge_count, outcome). outcome ∈
    {none, establish, reinforce, challenge, switch} — challenge/switch를 호출부가 trace로 남긴다.
    순수 함수(ctx·robot 무관) — pytest로 잠근다.
    """
    if detected is None:
        return current, challenge_color, challenge_count, "none"
    if current is None:
        return detected, None, 0, "establish"
    if detected == current:
        # 확정색 재확인 → 진행 중이던 도전 무효화.
        return current, None, 0, "reinforce"
    # detected != current: 플랩 시도.
    if detected == challenge_color:
        count = challenge_count + 1
        if count >= switch_n:
            return detected, None, 0, "switch"
        return current, detected, count, "challenge"
    # 새로운 도전색 등장 → 카운트 1로 시작(직전 도전색은 연속성이 끊겼으므로 폐기).
    return current, detected, 1, "challenge"


def _select_acquire_target(
    candidates: list[Any],
    belt_color: str | None,
    locked_color: str | None,
) -> Any | None:
    """획득 모드에서 다가갈 큐브를 고릅니다(벨트색 후순위 + 진동 방지 hysteresis)."""
    if not candidates:
        return None
    # 벨트색이 아닌 큐브를 우선, 그다음 큰(가까운) 순.
    best = max(candidates, key=lambda d: (d.color != belt_color, d.blob_area))
    # 이미 어떤 색을 추적 중이면, 새 후보가 충분히(NAV_RESELECT_AREA_RATIO배) 더 커야만
    # target을 바꿔 등거리 큐브 사이에서 프레임마다 흔들리는 것을 막습니다.
    if locked_color is not None and best.color != locked_color:
        locked_now = [d for d in candidates if d.color == locked_color]
        if locked_now:
            locked_best = max(locked_now, key=lambda d: d.blob_area)
            if best.blob_area < locked_best.blob_area * NAV_RESELECT_AREA_RATIO:
                return locked_best
    return best


async def _clean_cube_ready(ctx: Any) -> tuple[bool, str | None]:
    """pick 준비 여부(ready)와 color-blind pick이 실제로 집을 큐브 색을 함께 반환합니다.

    color-blind pick_entity는 부채꼴(콘)과 무관하게 3D 최근접 큐브 엔티티를 잡습니다. 그래서
    '집을 색'은 콘을 무시하고 전체에서 가장 크게(=가장 가깝게) 보이는 '깨끗한' 큐브의 색으로
    리포트해 실제 pick 결과와 일치시킵니다(옛 콘-내 최댓값 리포트는 pick이 콘 밖 큐브를 잡을 때
    라벨이 어긋났음 — 배송엔 무해했지만 진단이 오해를 부름). ready 게이트는 '충분히 가까운
    (area >= PICK_READY_AREA) 깨끗한 큐브'가 보이는지만 봅니다(콘/중앙 정렬 요구 없음 — 아래 주석).
    실제 잡은 색은 pick 뒤 get_held_cube_info로 최종 확정하므로, 이 색은 어디까지나 예측/진단용입니다.
    """
    clean = [d for d in await perceive(ctx) if _is_clean_cube(d, CUBE_ARRIVAL_AREA)]
    if not clean:
        return (False, None)
    grab = max(clean, key=lambda d: d.blob_area)  # 콘 무관 최근접 ≈ pick_entity가 잡을 큐브.
    # 중앙 정렬을 요구하지 않습니다: pick_entity는 각도 무관하게 최근접 큐브를 스스로 파지하고
    # (라이브에서 -26°/area~8000 즉시 성공), 이 학습 정책은 짧은 회전으로 큐브를 중앙에 정렬하지
    # 못합니다(ramp-up으로 제자리 맴돌기만 함). 그래서 '충분히 가까운(area) clean 큐브'면 준비 완료.
    ready = grab.blob_area >= PICK_READY_AREA
    return (ready, grab.color)


def _stationary_from_frames(
    f1: list[Any], f2: list[Any], *, move_deg: float, area_frac: float
) -> list[Any]:
    """두 프레임 detection에서 '정지'한 큐브(f2 원소)만 골라 반환합니다(순수 — pytest로 잠급니다).

    로봇이 정지한 채 찍은 두 프레임에서, f2의 각 큐브를 같은 색 f1 후보 중 angle_deg가 가장 가까운
    것과 매칭합니다. angle_deg 변화(수평 이동)와 blob_area 변화율(전후 이동)이 둘 다 임계 이하면
    '정지'로 봅니다. 벨트 위 큐브는 프레임 사이 크게 이동해 제외됩니다. f1에 같은 색이 없으면(방금
    시야로 들어옴) 이동으로 간주해 제외 — 좌표·엔티티 0(카메라 각도/면적만, §0/§7).
    """
    out: list[Any] = []
    for d2 in f2:
        same = [d1 for d1 in f1 if d1.color == d2.color]
        if not same:
            continue
        d1 = min(same, key=lambda d: abs(d.angle_deg - d2.angle_deg))
        moved = abs(d1.angle_deg - d2.angle_deg)
        af = abs(d1.blob_area - d2.blob_area) / max(d1.blob_area, d2.blob_area, 1.0)
        if moved <= move_deg and af <= area_frac:
            out.append(d2)
    return out


async def _stationary_clean_cubes(
    ctx: Any, arrival_area: int, *, memory: AgentMemory | None = None
) -> list[Any]:
    """로봇 정지 상태 2프레임(~STATIONARY_DT_S)을 비교해 '정지'한 clean 큐브만 반환합니다(VLM 0).

    호출부는 locomotion 없이(정지 상태) 호출해야 ego-motion 오검출이 없습니다. 벨트 위 이동 큐브를
    쫓아 1.2m 반경에 못 넣던 실패(run2/4)를 상류에서 차단 — 정지 큐브가 없으면 호출부가 이동 큐브를
    쫓는 대신 재배치합니다. perceive 2장(각 ~30ms) + STATIONARY_DT_S 대기라 VLM 대비 저비용.
    """
    f1 = [d for d in await perceive(ctx) if _is_clean_cube(d, arrival_area)]
    await asyncio.sleep(STATIONARY_DT_S)
    f2 = [d for d in await perceive(ctx) if _is_clean_cube(d, arrival_area)]
    out = _stationary_from_frames(
        f1, f2, move_deg=STATIONARY_MOVE_DEG, area_frac=STATIONARY_AREA_FRAC
    )
    _trace_step(memory, action="stationary_scan", total=len(f2), stationary=len(out))
    return out


def _detect_rack_color(detections: list[Any]) -> str | None:
    """초대형 blob(area > CUBE_ARRIVAL_AREA×MAX_AREA_ARRIVAL_MULT=36000) 중 최빈 색을 랙 색으로
    런타임 추정합니다(색 하드코딩 금지 — _detect_belt_color와 동형). 초대형 blob이 없으면 None.

    씬 좌표가 아니라 '초대형'이라는 물리 성질에서 유도하므로 히든 setup에서도 성립합니다. sector map은
    이 색의 blob만 랙으로 취급하고, None이면 rack_sector_map이 전 섹터 inf로 안전 degrade합니다.
    """
    thresh = CUBE_ARRIVAL_AREA * MAX_AREA_ARRIVAL_MULT
    colors = [d.color for d in detections if getattr(d, "blob_area", 0) > thresh]
    if not colors:
        return None
    return max(set(colors), key=colors.count)


def rack_sector_map(
    detections: list[Any],
    frame_h: float,
    *,
    ground_a: float,
    ground_b: float,
    n_sectors: int = 8,
    hfov_deg: float = CAMERA_HFOV_DEG,
    block_near_m: float = RACK_BLOCK_NEAR_M,
    side_margin_m: float = RACK_SIDE_MARGIN_M,
) -> dict[str, Any]:
    """랙 blob들의 하단행 지면 투영으로 egocentric 8섹터 blocked-거리 배열을 만듭니다(순수).

    반환 {"sectors":[d0..d7|inf], "blocked_front":bool, "freer_side":+1좌/-1우/0}. 각 섹터 45°,
    로봇 정면=중앙 섹터(index n//2). 값은 그 섹터 최근접 랙까지 거리 m(없으면 inf). 지면 투영:
    d=ground_a/(y_bottom_frac-ground_b)(pitch=HEAD_PITCH_TRACK 전제 광학 상수). 방위는 blob의 full_bearing_deg
    (head yaw 포함, 없으면 angle_deg)로 섹터를 고릅니다. 랙 blob 0(또는 _detect_rack_color None)이면
    전 섹터 inf(장애 없음)로 안전 degrade합니다 — 발밑 free_space_profile이 최종 방어라 오보정 무해.
    """
    inf = float("inf")
    sectors = [inf] * n_sectors
    empty = {"sectors": sectors, "blocked_front": False, "freer_side": 0.0}
    if not detections or frame_h <= 0 or n_sectors < 2:
        return empty
    rack_color = _detect_rack_color(detections)
    if rack_color is None:
        return empty
    sector_width = 360.0 / n_sectors
    for d in detections:
        if getattr(d, "color", None) != rack_color:
            continue
        bbox = getattr(d, "bbox", None)
        if not bbox:
            continue
        _x0, y0, _w, h = bbox
        if h <= 0:
            continue
        y_bottom_frac = (float(y0) + float(h)) / float(frame_h)
        denom = y_bottom_frac - ground_b
        if denom <= 0.0:
            continue  # 지평선 위(투영 무효).
        dist = ground_a / denom
        bearing = float(getattr(d, "full_bearing_deg", getattr(d, "angle_deg", 0.0)))
        # 반 섹터 오프셋: 정면(bearing 0)이 중앙 섹터의 '중심'에 오도록(경계가 아니라). 오프셋이 없으면
        # bearing -5°가 좌측 섹터로 새어, 중앙 섹터만 검사하는 blocked_front가 코앞(약간 좌측) 장애를
        # 놓칩니다 — 중앙 섹터 = [-sector_width/2, +sector_width/2) 대칭 전방 콘.
        idx = int(((bearing + 180.0 + sector_width / 2.0) % 360.0) // sector_width) % n_sectors
        if dist < sectors[idx]:
            sectors[idx] = dist
    center = n_sectors // 2
    left_open = min(sectors[:center]) if center > 0 else inf
    right_open = min(sectors[center + 1:]) if center + 1 < n_sectors else inf
    if left_open > right_open + side_margin_m:
        freer = 1.0
    elif right_open > left_open + side_margin_m:
        freer = -1.0
    else:
        freer = 0.0
    return {
        "sectors": sectors,
        "blocked_front": sectors[center] < block_near_m,
        "freer_side": freer,
    }


def _rack_preferred_side(rack_map: dict[str, Any] | None, fallback_side: float) -> float:
    """sector map freer_side가 결정적(±1)이면 그것을, 아니면 fallback_side를 detour side로 반환(순수).

    rack_map None(미보정·비활성·랙 0)이면 fallback을 그대로 반환 → 기존 side 선택이 유지됩니다(무영향).
    """
    if rack_map is None:
        return fallback_side
    fs = rack_map.get("freer_side", 0.0)
    return fs if fs != 0.0 else fallback_side


def _rack_front_blocked(
    rack_map: dict[str, Any] | None,
    *,
    near_m: float = PAD_ANCHOR_NEAR_M,
    block_m: float = RACK_BLOCK_NEAR_M,
) -> bool:
    """목표 방위(정면) 섹터에 랙이 near_m~block_m 대역으로 다가오면 True(원거리 진입 억제 트리거, 순수).

    near_m(1.2m) 이내는 불관여(발밑·마무리 접근·place-probe 대역) — 오억제 방지. rack_map None이면 False.
    """
    if rack_map is None:
        return False
    sectors = rack_map.get("sectors") or []
    if not sectors:
        return False
    front = sectors[len(sectors) // 2]
    return near_m < front < block_m


def _rack_staging_side(rack_map: dict[str, Any] | None) -> float:
    """정면이 막히고 좌/우 인접 섹터 중 더 열린 쪽이 있으면 staging 방향(+1좌/-1우), 아니면 0(순수)."""
    if rack_map is None:
        return 0.0
    sectors = rack_map.get("sectors") or []
    if len(sectors) < 3:
        return 0.0
    c = len(sectors) // 2
    left = sectors[c - 1]
    right = sectors[c + 1] if c + 1 < len(sectors) else float("inf")
    if left == right:
        return 0.0
    return 1.0 if left > right else -1.0


def _stage_chunk_count(
    dist_m: float, *, chunk_m: float = FORWARD_EFF_SPEED_MPS * PAD_ADVANCE_DUR
) -> int:
    """staging 목표 거리를 덮는 데 필요한 전진 청크 수(각 청크 ≈ FORWARD_EFF_SPEED_MPS×PAD_ADVANCE_DUR).

    최소 1, 상한 SRC_SEEK_MAX_CHUNKS(과도한 경유 방지). RACK_STAGE_DIST_M(1.5m)이 실제 이동에
    반영되게 합니다 — 단일 기본 청크(≈0.38m)로는 '1.5m 경유'가 성립하지 않던 결함(Codex MAJOR 1) 수정.
    """
    if dist_m <= 0.0 or chunk_m <= 0.0:
        return 1
    return min(max(1, math.ceil(dist_m / chunk_m)), SRC_SEEK_MAX_CHUNKS)


async def _rack_map_from_frame(ctx: Any) -> dict[str, Any] | None:
    """현재 POV 프레임으로 sector map을 계산합니다(활성 시). 비활성(미보정)이거나 프레임 획득/디코드
    실패면 None을 돌려주어 모든 호출부가 영향 없이 기존 동작을 유지합니다 — 이 조기 None이 M2 배선의
    회귀 방지 계약입니다(RACK_SECTOR_MAP_ENABLED=True + RACK_GROUND_A/B 보정 후에만 실제 지도 반환).
    """
    if not RACK_SECTOR_MAP_ENABLED:
        return None
    try:
        import cv2
        import numpy as np

        jpeg = await ctx.get_vision("pov")
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        dets = detect_color_blobs(jpeg)
        return rack_sector_map(dets, float(img.shape[0]), ground_a=RACK_GROUND_A, ground_b=RACK_GROUND_B)
    except Exception:
        return None


def _blob_targetable(
    detection: Any, target_color: str, arrival_area: int, *, allow_oversize: bool = False
) -> bool:
    """push 서보가 이 blob을 계속 추적해도 되는지 판정합니다(순수 — pytest로 잠급니다).

    기본은 _plausible_target(노이즈 하한 + 레일/초대형 상한 + aspect/폭). ★H(진단 2026-07-06)★
    allow_oversize=True는 place 최종 진입 push 전용: 근접 pad blob은 area가 상한(36000)을 넘어
    (run8 실측 40472) 서보 눈에서 사라지고, push가 'blob 상실'로 1청크(run11: 0.09m) 만에 끝나
    1.1m 밖 place → place_entity는 로봇 전방 ~0.5m에 큐브를 놓으므로 존 밖(팔레트 가장자리)에
    착지 → 흡수(배달 판정) 실패가 됩니다. 이 모드는 '같은 색 + 최소 면적(노이즈 차단)'만 요구해
    프레임을 채운 pad blob도 팔레트 접촉(stall 종단)까지 추적하게 합니다. pick push에는 절대 쓰지
    말 것 — pick의 상한은 벨트 세그먼트 오인(run6 전도)을 막는 방어입니다.
    """
    if detection.color != target_color:
        return False
    if allow_oversize:
        return detection.blob_area >= MIN_TARGET_AREA
    return _plausible_target(detection, arrival_area)


async def _best_color_blob(
    ctx: Any, target_color: str | None, arrival_area: int, *, allow_oversize: bool = False
) -> Any | None:
    """target_color의 추적 가능 blob 중 최대(≈최근접)를 반환합니다(없으면 None).

    _target_in_range의 인식 경로를 분리한 것 — R6 수렴 관측이 도착 게이트와 동일한
    인식으로 면적 표본을 얻게 하기 위함입니다(별도 인식이면 표본과 게이트가 어긋남).
    allow_oversize는 place 진입 push 전용(_blob_targetable 참조).
    """
    if target_color is None:
        return None
    matching = [
        d for d in await perceive(ctx)
        if _blob_targetable(d, target_color, arrival_area, allow_oversize=allow_oversize)
    ]
    return max(matching, key=lambda d: d.blob_area) if matching else None


async def _target_in_range(ctx: Any, target_color: str | None, arrival_area: int) -> bool:
    """target_color blob이 arrival_area 이상으로 크고 화면 중앙 근처에 보이면 True입니다."""
    best = await _best_color_blob(ctx, target_color, arrival_area)
    return (
        best is not None
        and best.blob_area >= arrival_area
        and abs(best.angle_deg) <= CENTER_TOLERANCE_DEG * 1.5
    )


async def visual_search(ctx: Any, target_color: str | None = None) -> bool:
    """Camera movement와 robot motion으로 cube 또는 pad를 search합니다.

    TODO:
    - set_head 또는 body rotation을 사용하는 scan pattern을 설계하세요.
    - 필요하면 cube와 pad를 어떻게 구분할지 결정하세요.
    - Visual centering에 도움이 되면 detection.full_bearing_deg를 사용하세요.
    - 유용한 target을 찾았는지 반환하세요.
    """
    # cube를 찾는지(pick) pad를 찾는지(place)에 따라 크기 상한이 달라집니다.
    held = await get_held_cube_info(ctx)
    arrival_area = PAD_ARRIVAL_AREA if held else CUBE_ARRIVAL_AREA

    # body를 조금씩 회전시키며 매 방향마다 head를 훑어 넓게 search합니다.
    for _ in range(SEARCH_MAX_ROTATIONS):
        detections = await scan_head(ctx)
        # 이후 body-frame 조향을 위해 head를 정면으로 되돌립니다.
        await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)

        if target_color is None:
            # 획득: 색 고정 없이 '깨끗한' 큐브만 후보. 벨트색은 제외가 아니라 후순위(다른 색 먼저).
            belt = _detect_belt_color(detections)
            pool = [d for d in detections if _is_clean_cube(d, arrival_area)]
        else:
            # pad 탐색 등 색이 지정된 경우: 해당 색의 plausible blob만.
            belt = None
            pool = [d for d in detections if d.color == target_color and _plausible_target(d, arrival_area)]
        if pool:
            best = max(pool, key=lambda d: (d.color != belt, d.blob_area))
            # full_bearing_deg(head yaw 포함)로 target 쪽으로 body를 대략 정렬합니다.
            # 제자리 회전은 학습 정책상 안 먹히므로 아크(vx>0+wz)로 살짝 걸으며 정렬합니다.
            bearing = best.full_bearing_deg
            if abs(bearing) > CENTER_TOLERANCE_DEG:
                wz = -ARC_WZ if bearing > 0 else ARC_WZ
                await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=min(abs(bearing) / 40.0, 1.2))
            return True

        # 이 방향에서 target을 못 찾았으면 아크-스윕(걸으며 회전)으로 시야를 옮긴 뒤 다시 훑습니다.
        await move_velocity(ctx, vx=SWEEP_VX, wz=SWEEP_WZ, duration_s=0.8)
    return False


async def visual_navigate_to_target(ctx: Any, target_color: str | None, *, verbose: bool = False) -> bool:
    """카메라 피드백만 사용해 cube 또는 pad 앞까지 폐루프 이동합니다.

    이 함수는 Level 2 규칙에 맞게 scene_state, 좌표, go_to 없이 동작합니다.
    매 step마다 현재 POV frame을 다시 인식하고, target의 화면상 각도와 blob 크기만으로
    `회전하며 전진할지`, `직진할지`, `도착으로 볼지`, `실패로 빠질지`를 결정합니다.

    동작 모드:
    - `target_color is None`: cube 획득 모드입니다. 색을 미리 고정하지 않고
      `_is_clean_cube()`를 통과한 깨끗한 큐브 후보 중 하나를 따라갑니다. 컨베이어 벨트/레일
      색은 `_detect_belt_color()`로 후순위 처리하고, `_select_acquire_target()`의
      hysteresis로 프레임마다 다른 큐브로 흔들리는 것을 막습니다.
    - `target_color is not None`: 색 지정 모드입니다. 주로 delivery pad 접근에 쓰며,
      해당 색의 `_plausible_target()` 후보 중 가장 큰 blob을 따라갑니다.

    도착 판정:
    - robot이 cube를 들고 있지 않으면 `CUBE_ARRIVAL_AREA`, 들고 있으면
      `PAD_ARRIVAL_AREA`를 기준으로 삼습니다.
    - blob 면적이 기준 이상이고, 동시에 `CENTER_TOLERANCE_DEG` 안에 들어와야
      도착으로 인정합니다. 면적만 보면 옆으로 지나가는 레일 조각도 도착으로 오판할 수
      있어서 중앙 조건을 함께 둡니다.

    반환값:
    - `True`: target이 충분히 크고 중앙에 보여서 다음 pick/place를 시도해도 되는 상태입니다.
    - `False`: target을 3회 연속 잃었거나 `NAV_MAX_STEPS` 안에 도착하지 못한 상태입니다.

    부작용:
    - `set_head()`로 시선을 정면에 맞춥니다.
    - `move_velocity()`로 짧은 전진/회전 명령을 반복합니다.
    """
    # cube를 향하는지(pick) pad를 향하는지(place)에 따라 도착 판정 크기가 다릅니다.
    held = await get_held_cube_info(ctx)
    arrival_area = PAD_ARRIVAL_AREA if held else CUBE_ARRIVAL_AREA

    # body-frame servoing을 위해 head를 정면으로 맞춥니다(이미지 각도≈몸통 방위).
    await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)

    lost_streak = 0
    locked_color: str | None = None
    for _step in range(1, NAV_MAX_STEPS + 1):
        raw = await perceive(ctx)
        if target_color is None:
            # 획득 모드: 색 고정 없이 '깨끗한' 큐브로. 벨트색은 후순위, hysteresis로 진동 방지.
            belt = _detect_belt_color(raw)
            matching = [d for d in raw if _is_clean_cube(d, arrival_area)]
            best = _select_acquire_target(matching, belt, locked_color)
            if best is not None:
                locked_color = best.color
        else:
            # 색 지정(주로 pad): 해당 색의 plausible blob 중 가장 큰 것.
            matching = [
                d for d in raw
                if d.color == target_color and _plausible_target(d, arrival_area)
            ]
            best = max(matching, key=lambda d: d.blob_area) if matching else None

        if best is None:
            # target loss: 아크-스윕으로 걸으며 재획득을 시도하고, 계속 못 찾으면 실패로 종료합니다.
            lost_streak += 1
            if verbose:
                print(f"  [nav {_step}] lost (streak={lost_streak}) -> sweep")
            if lost_streak >= 3:
                if verbose:
                    print("  [nav] FAIL: target 3연속 손실")
                return False
            await move_velocity(ctx, vx=SWEEP_VX, wz=SWEEP_WZ, duration_s=0.8)
            continue
        lost_streak = 0

        area = best.blob_area
        angle = best.angle_deg

        if area >= arrival_area and abs(angle) <= CENTER_TOLERANCE_DEG:
            # 도착: 충분히 가깝고 target이 화면 중앙에 있습니다. 중앙 조건이 없으면
            # 옆으로 지나가는 레일 밴드가 area만으로 조기 도착을 유발할 수 있습니다.
            if verbose:
                print(f"  [nav {_step}] ARRIVE {best.color} area={area} angle={angle:.1f}")
            return True

        if abs(angle) > CENTER_TOLERANCE_DEG:
            # 아직 중앙이 아니면 아크(vx>0+wz)로 target 쪽으로 선회합니다(제자리 회전 불가).
            wz = -ARC_WZ if angle > 0 else ARC_WZ
            if verbose:
                print(f"  [nav {_step}] arc  {best.color} area={area} angle={angle:.1f} wz={wz:+.1f}")
            await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=0.5)
        else:
            # 중앙에 있으면 똑바로 전진합니다.
            if verbose:
                print(f"  [nav {_step}] fwd  {best.color} area={area} angle={angle:.1f}")
            await move_velocity(ctx, vx=FORWARD_VX, duration_s=0.8)

    if verbose:
        print(f"  [nav] FAIL: {NAV_MAX_STEPS} step 내 도착 실패")
    return False


def _norm_vlm_key(k: Any) -> str:
    """VLM 응답 dict 키를 공백/언더스코어/하이픈·대소문자 차이를 무시하도록 정규화합니다.

    qwen은 같은 필드를 'sign letter'(공백)·'sign_letter'(언더스코어)·'Sign Letter'로 흔들어
    반환합니다. 라이브 실측(2026-07-04): VLM이 정확히 'C'를 읽고도 키가 'sign letter'(공백)라
    exact-match 파서(VLM_LETTER_KEYS엔 'sign_letter'만 있음)가 통째로 버려 conf=0.00으로 붕괴,
    배송이 막혔습니다. 영숫자만 남겨 모든 표기 변형을 한 표준형으로 접습니다('sign letter'·
    'sign_letter' 모두 'signletter').
    """
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _get_by_norm_key(d: dict[str, Any], *names: str) -> Any:
    """dict에서 키를 _norm_vlm_key 비교로 조회합니다(첫 일치 값, 없으면 None)."""
    wanted = {_norm_vlm_key(n) for n in names}
    for k, v in d.items():
        if _norm_vlm_key(k) in wanted:
            return v
    return None


def _looks_like_vlm_fallback(text: str) -> bool:
    """Tokamak이 상위 모델에 못 닿을 때 HTTP 200으로 돌려주는 fallback 문장을 식별합니다.

    라이브 실측(2026-07-04): 응답 본문이 "I'm having trouble reaching the model right now, but
    here's a fallback response." 로 옴 — JSON도 글자도 없어 _parse_signs가 '표지 없음'과
    구분하지 못합니다(둘 다 conf=0.00). 명시적으로 잡아 재시도하게 해, 일시적 provider 플랩이
    표지 미검출로 오인돼 acquisition이 붕괴(런3)하는 것을 막습니다.
    """
    t = text.lower()
    return "trouble reaching the model" in t or "fallback response" in t


def _parse_signs(text: str) -> list[dict[str, Any]]:
    """VLM signage 응답에서 sign 목록을 견고하게 parse합니다.

    응답은 보통 [{letter, color, position, confidence}, ...] JSON이지만 코드펜스나
    설명이 섞일 수 있어, 첫 JSON 배열/객체만 추출해 해석합니다. 실패하면 빈 목록.
    """
    stripped = text.strip()
    if "```" in stripped:
        for part in stripped.split("```"):
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                stripped = p
                break
    starts = [i for i in (stripped.find("["), stripped.find("{")) if i >= 0]
    if not starts:
        return []
    start = min(starts)
    end = max(stripped.rfind("]"), stripped.rfind("}"))
    if end <= start:
        return []
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data["signs"] if isinstance(data.get("signs"), list) else [data]
    if not isinstance(data, list):
        return []
    # VLM(qwen)은 글자 키를 'letter'가 아니라 'sign_letter'/'label'/'text'/'text_content'로,
    # 위치를 'position'이 아니라 'approximate_position'으로 반환하기도 합니다(라이브 실측).
    # 다양한 키 별칭을 표준 키('letter','position','confidence')로 정규화해 하위 소비자가
    # 일관되게 읽게 합니다. confidence 없이 bbox+글자만 주는 형식도 실측됐으므로 결측 시
    # 기본 신뢰도를 부여해 검출이 게이트에서 통째로 기각되지 않게 합니다.
    normalized: list[dict[str, Any]] = []
    for s in data:
        if not isinstance(s, dict):
            continue
        letter = _extract_sign_letter(s)
        if not letter:
            continue
        out = dict(s)
        out["letter"] = letter
        out["position"] = _get_by_norm_key(s, "position", "approximate_position") or ""
        # bbox/confidence도 키 표기 흔들림(공백/언더스코어/대소문자)을 정규화해 표준 키로 실어,
        # 하위 소비자(_sign_offset_deg·_sign_bbox_area_frac·_as_confidence)가 일관되게 읽게 합니다.
        bbox = _get_by_norm_key(s, "bbox_2d", "bbox")
        if bbox is not None:
            out["bbox_2d"] = bbox
        conf_val = _get_by_norm_key(s, "confidence")
        if conf_val is None:
            # 결측·null confidence는 방위 근거(bbox나 position)가 있는 검출만 기본 신뢰도로
            # 구제합니다. 근거가 하나도 없는 글자-단독 검출(환각 가능)은 기본값을 주면
            # _sign_offset_deg가 0°(정면)로 fallback해 엉뚱한 직진을 유발하므로 게이트가
            # 거르도록 둡니다.
            has_bearing = (
                isinstance(bbox, (list, tuple)) and len(bbox) == 4
            ) or bool(out["position"])
            if has_bearing:
                out["confidence"] = VLM_DEFAULT_CONFIDENCE
        else:
            out["confidence"] = conf_val
        normalized.append(out)
    return normalized


def _letter_from_phrase(text: str) -> str | None:
    """서술 문장에서 표지 글자 하나를 추출합니다("...white letter 'C'" → 'C').

    라이브 실측: qwen이 글자를 별도 키가 아니라 description 문장에만 싣고 인용부호로 감싸
    ('C') 단순 토큰 분리로는 못 잡는 경우가 있습니다. 인용부호·구두점을 공백으로 바꿔
    토큰화한 뒤 (1) 'letter' 바로 뒤 단일 알파벳, (2) 단일 '대문자' 토큰 순으로 봅니다.
    대문자를 요구해 관사 'a'/'an' 같은 소문자 단일 글자를 글자로 오인하지 않습니다.
    """
    for ch in "'\"-.,:;()[]":
        text = text.replace(ch, " ")
    tokens = text.split()
    for i, tok in enumerate(tokens[:-1]):
        if tok.lower() == "letter":
            nxt = tokens[i + 1]
            if len(nxt) == 1 and nxt.isalpha():
                return nxt.upper()
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha() and tok.isupper():
            return tok
    return None


def _extract_sign_letter(s: dict[str, Any]) -> str | None:
    """sign dict에서 표지 글자(단일 알파벳)를 견고하게 추출합니다.

    qwen은 글자를 VLM_LETTER_KEYS 중 아무 키에나 싣고, label에는 글자 대신 'sign letter' 같은
    서술어가 올 수도 있습니다(라이브 실측). 단일 알파벳 값을 우선 인정하고, 'sign C' 같은 혼합
    표기에서는 단일 글자 토큰을 찾으며, 서술어만 있으면 글자로 오인하지 않고 버립니다.
    글자 키가 하나도 안 잡히면 description/desc/caption 서술 문장에서 마지막으로 시도합니다.
    """
    # 키를 정규화 비교로 매칭 — qwen이 'sign letter'(공백)/'sign_letter'(언더스코어)를 오가도
    # 모두 잡습니다(라이브 실측: 공백 키를 놓쳐 정확히 읽은 'C'를 통째로 버렸던 붕괴 수정).
    wanted = {_norm_vlm_key(k) for k in VLM_LETTER_KEYS}
    values = [str(v).strip() for k, v in s.items() if v and _norm_vlm_key(k) in wanted]
    for v in values:
        if len(v) == 1 and v.isalpha():
            return v
    for v in values:
        tokens = [t for t in v.replace("-", " ").split() if len(t) == 1 and t.isalpha()]
        if tokens:
            return tokens[0]
    for k in VLM_LETTER_DESC_KEYS:
        val = _get_by_norm_key(s, k)
        if val:
            letter = _letter_from_phrase(str(val))
            if letter:
                return letter
    return None


def _as_confidence(v: Any) -> float:
    """VLM confidence를 float으로 안전 변환합니다.

    qwen은 confidence를 0.95 같은 수치뿐 아니라 'high'/'medium'/'low' 문자열로도 반환합니다
    (라이브 실측: `float('high')`가 navpad 전체를 ValueError로 크래시시킴). 문자열 등급을
    대표 수치로 매핑하고, 파싱 불가한 값은 0.0으로 떨궈 안전하게 무시합니다.
    """
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    named = {"high": 0.9, "medium": 0.6, "med": 0.6, "low": 0.3}
    if s in named:
        return named[s]
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _find_target_sign(signs: list[dict[str, Any]], letter: str) -> dict[str, Any] | None:
    """parse된 sign 목록에서 목표 글자와 일치하는 sign을 confidence 최고로 고릅니다."""
    cands = [s for s in signs if str(s.get("letter", "")).strip().upper() == letter.upper()]
    if not cands:
        return None
    return max(cands, key=lambda s: _as_confidence(s.get("confidence", 0)))


def _sign_offset_deg(target: dict[str, Any]) -> float:
    """target sign의 화면상 수평 위치를 카메라 기준 방위 오프셋(도, +=우측)으로 환산합니다.

    bbox_2d(qwen 0~1000 정규화)가 있으면 중심 x의 화면 비율로 비례 환산합니다 — left/right
    ±PAD_POS_OFFSET_DEG 양자화는 far-left 표지(실측 방위 ~-27°)를 한 번에 15°만 보정해
    영영 정면을 못 맞춥니다(라이브 확정). bbox가 없거나 정규화 범위를 벗어나면(픽셀 좌표 등)
    기존 left/center/right 양자화로 fallback합니다.
    """
    bbox = target.get("bbox_2d") or target.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, _, x2, _ = (float(v) for v in bbox)
        except (TypeError, ValueError):
            x1, x2 = -1.0, -1.0
        if 0.0 <= x1 < x2 <= VLM_BBOX_SCALE:
            frac = ((x1 + x2) / 2.0) / VLM_BBOX_SCALE
            return (frac - 0.5) * CAMERA_HFOV_DEG
    position = str(target.get("position", "center")).strip().lower()
    if "left" in position:
        return -PAD_POS_OFFSET_DEG
    if "right" in position:
        return PAD_POS_OFFSET_DEG
    return 0.0


# ---------------------------------------------------------------------------
# 학생 TODO: 경로 기억(route memory) 순수 헬퍼 — unit test 대상
# ---------------------------------------------------------------------------
# 아래 헬퍼들은 로봇 자신의 odometry pose(고유수용성)와 카메라/VLM 관찰에서 유도한
# "학생 추정치"만 다룹니다. scene_state, entity ID, coordinate go_to는 일절 쓰지 않고
# 이동은 언제나 set_velocity 폐루프(_turn_by_deg/_advance_or_detour)로만 수행하므로
# Level 2에서 합법입니다(발표에서 명시할 것). ctx가 없는 순수 함수라 unit test로 잠급니다.


def _pose_dict(robot_status: Any) -> dict[str, float]:
    """robot_status에서 {x, y, yaw_deg}만 뽑은 간결한 pose dict를 만듭니다(결측은 0.0)."""
    pose = getattr(getattr(robot_status, "robot", None), "pose", None)
    pos = getattr(pose, "position", None) or (0.0, 0.0, 0.0)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "yaw_deg": float(getattr(pose, "yaw_deg", 0.0) or 0.0),
    }


def _face_turn_to(pose: dict[str, float], target: dict[str, float]) -> tuple[float, float]:
    """현재 pose에서 target 지점까지의 (유클리드 거리 m, 필요 회전각 도)를 계산합니다.

    벡터 to_target=(dx,dy)의 세계 방위각 atan2(dy,dx)와 현재 yaw의 차를 (-180,180]로
    정규화한 값이 face_turn입니다(양수=좌회전, _turn_by_deg 규약과 동일).
    """
    dx = float(target["x"]) - float(pose["x"])
    dy = float(target["y"]) - float(pose["y"])
    dist = math.hypot(dx, dy)
    bearing_deg = math.degrees(math.atan2(dy, dx))
    return dist, _angle_diff_deg(bearing_deg, float(pose.get("yaw_deg", 0.0)))


def _movement_efficiency(expected_m: float, actual_m: float) -> float:
    """실제 이동량/기대 이동량(속도×시간 운동 모델). 기대가 0 이하면 0.0(판단 불가)."""
    if expected_m <= 0.0:
        return 0.0
    return actual_m / expected_m


def _is_stalled(expected_m: float, actual_m: float) -> bool:
    """이동 효율 기반 stall 판정: 실제 병진이 기대의 STALL_EFF_RATIO 미만이면 막힌 것.

    기대 이동량에 비례하므로 거리 기반으로 짧아진 전진 청크에도 같은 기준이 성립하고,
    STALL_ABS_FLOOR_M 하한이 odometry 노이즈 오탐을 막습니다.
    """
    return actual_m < max(STALL_ABS_FLOOR_M, expected_m * STALL_EFF_RATIO)


def free_space_profile(
    value: Any,
    *,
    feet_top: float = FREE_FEET_TOP_FRAC,
    floor_pct: float = FREE_FLOOR_PCT,
    floor_margin: float = FREE_FLOOR_MARGIN,
    center: tuple[float, float] = (FREE_CENTER_LO, FREE_CENTER_HI),
    side_lo: float = FREE_SIDE_LO,
    side_hi: float = FREE_SIDE_HI,
    block_frac: float = FREE_BLOCK_FRAC,
    side_margin: float = FREE_SIDE_MARGIN,
) -> dict[str, Any]:
    """카메라 밝기(V=채널 최대) 배열로 로봇 발 앞 바닥의 free-space를 판정합니다(순수 함수).

    value는 HxW 밝기 배열(0~255). 프레임 하단 strip(발 앞 지면)에서 floor_pct 백분위를 '바닥'
    기준으로 잡고, 그보다 floor_margin 이상 밝은 화소를 '바닥 아님(구조물)'으로 봅니다. 중앙
    밴드의 '바닥 아님' 비율이 block_frac 이상이면 정면 막힘(clear=False)입니다. 좌/우 밴드의
    막힘 비율을 비교해 더 열린(덜 막힌) 쪽을 freer_side(+1=좌, -1=우, 0=차이 미미)로 돌려줍니다.

    바닥이 어둡고 구조물이 밝다는 라이브 관찰(nav 프레임 실측)에 기반한 자기보정 방식이라 색을
    하드코딩하지 않으며(히든 평가 조명 변화에 견딤), get_vision 1프레임만 쓰므로 VLM처럼 비싸지
    않습니다. ctx가 없는 순수 함수라 합성 배열로 unit test에 잠급니다. 완전히 균일하게 밝은
    발밑(대비 0)은 floor_ref 자체가 높아져 '바닥'으로 읽힐 수 있으나, 유한한 구조물 앞에서는 발밑
    가장자리에 늘 어두운 바닥이 남아 대비가 생기고, 완전히 둘러싸인 드문 경우는 반응형 stall이
    잡습니다(선제 인지는 반응형을 대체하지 않고 보완).
    """
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    default = {
        "clear": True, "center": 0.0, "left": 0.0, "right": 0.0,
        "freer_side": 0.0, "floor_ref": 0.0, "threshold": 0.0,
    }
    if arr.ndim != 2 or arr.size == 0:
        return default
    h, w = arr.shape
    feet = arr[int(h * feet_top):, :]
    if feet.size == 0:
        feet = arr
    floor_ref = float(np.percentile(feet, floor_pct))
    threshold = floor_ref + floor_margin
    blocked = feet > threshold

    def _frac(a: float, b: float) -> float:
        lo, hi = int(w * a), int(w * b)
        if hi <= lo:
            return 0.0
        return float(blocked[:, lo:hi].mean())

    c = _frac(center[0], center[1])
    left = _frac(side_lo, side_hi)
    right = _frac(1.0 - side_hi, 1.0 - side_lo)
    if left + side_margin < right:
        freer = 1.0        # 좌측이 덜 막힘 -> 좌회전(+1)이 열린 쪽.
    elif right + side_margin < left:
        freer = -1.0       # 우측이 덜 막힘 -> 우회전(-1)이 열린 쪽.
    else:
        freer = 0.0        # 차이 미미 -> 선호 없음(호출부가 이력/표지 부호로 결정).
    return {
        "clear": c < block_frac,
        "center": c,
        "left": left,
        "right": right,
        "freer_side": freer,
        "floor_ref": floor_ref,
        "threshold": threshold,
    }


def _advance_duration_s(dist_m: float) -> float:
    """남은 거리 기반 전진 시간(속도×시간 모델의 역산: t = d / v_실효).

    [ADVANCE_MIN_S, PAD_ADVANCE_DUR]로 클램프합니다 — 너무 짧으면 램프업 미달로 안 걷고,
    너무 길면 waypoint를 지나쳐 폐루프 재보정 기회를 잃습니다.
    """
    if dist_m <= 0.0:
        return ADVANCE_MIN_S
    return min(max(dist_m / FORWARD_EFF_SPEED_MPS, ADVANCE_MIN_S), PAD_ADVANCE_DUR)


def _route_score(stats: dict[str, Any]) -> float:
    """낮을수록 좋은 경로 점수 = 시간 + VLM×10 + stall×5 + 경로길이×1.5 (결측 키는 0)."""
    return (
        float(stats.get("total_time_s", 0.0))
        + float(stats.get("vlm_calls", 0)) * ROUTE_SCORE_VLM_W
        + float(stats.get("stalls", 0)) * ROUTE_SCORE_STALL_W
        + float(stats.get("path_len_m", 0.0)) * ROUTE_SCORE_PATH_W
    )


def _select_best_route(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """score 최소(=최선) 경로를 고릅니다. waypoints가 없는 항목은 재사용 불가라 제외."""
    valid = [r for r in routes if r.get("waypoints")]
    if not valid:
        return None
    return min(valid, key=lambda r: float(r.get("score", math.inf)))


def _nearest_waypoint_index(waypoints: list[dict[str, float]], x: float, y: float) -> int:
    """현재 위치에서 유클리드 최근접 waypoint 인덱스(동률이면 경로 후반 쪽)를 반환합니다.

    후반 쪽 동률 우선은 이미 지나온 앞부분으로 되돌아가는 낭비를 막기 위함입니다.
    """
    best_i, best_d = 0, math.inf
    for i, wp in enumerate(waypoints):
        d = math.hypot(float(wp["x"]) - x, float(wp["y"]) - y)
        if d <= best_d:
            best_i, best_d = i, d
    return best_i


def _compress_waypoints(
    points: list[dict[str, float] | None],
    min_gap_m: float = ROUTE_MIN_WAYPOINT_GAP_M,
) -> list[dict[str, float]]:
    """연속 waypoint 간격을 min_gap 이상으로 솎아냅니다(첫 점과 마지막 점은 반드시 유지).

    마지막 점은 드롭(place) 지점이라 간격 미달이어도 버리지 않고 직전 점을 대체합니다.
    """
    pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in points if p is not None]
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p["x"] - out[-1]["x"], p["y"] - out[-1]["y"]) >= min_gap_m:
            out.append(p)
    last = pts[-1]
    if (out[-1]["x"], out[-1]["y"]) != (last["x"], last["y"]):
        if len(out) > 1 and math.hypot(last["x"] - out[-1]["x"], last["y"] - out[-1]["y"]) < min_gap_m:
            out[-1] = last
        else:
            out.append(last)
    return out


def _route_waypoints(
    start_pose: dict[str, float] | None,
    trace: list[dict[str, Any]],
    drop_pose: dict[str, float] | None,
) -> list[dict[str, float]]:
    """배송 trace에서 실제 병진에 성공한 step의 pose만 뽑아 waypoint 목록으로 압축합니다."""
    points: list[dict[str, float] | None] = [start_pose]
    for step in trace:
        if step.get("stall"):
            continue
        if step.get("action") in {"advance", "replay_advance", "detour_advance"}:
            points.append(step.get("pose"))
    points.append(drop_pose)
    return _compress_waypoints(points)


def _make_last_seen(
    pose: dict[str, float],
    face_turn_deg: float,
    *,
    confidence: float = 0.0,
    position: str = "",
    source: str = "vlm",
) -> dict[str, Any]:
    """sign 목격 기록: 목격 pose와 sign의 world 방위(yaw+face_turn)를 저장합니다.

    단안 VLM이라 거리를 모르므로 sign을 '위치'가 아니라 '목격 pose에서의 방향(ray)'으로
    기억합니다. 재조준은 그 world 방위를 현재 yaw와 비교해 계산하고, 목격 pose에서
    멀어질수록 ray 가정의 기하 오차가 커지므로 LAST_SEEN_MAX_DRIFT_M 밖에서는 불신합니다.
    """
    return {
        "pose": dict(pose),
        "world_heading_deg": _angle_diff_deg(float(pose.get("yaw_deg", 0.0)) + face_turn_deg, 0.0),
        "confidence": confidence,
        "position": position,
        "source": source,
    }


def _last_seen_face_turn(last_seen: dict[str, Any] | None, pose: dict[str, float]) -> float | None:
    """last_seen 기반 재조준 각(도, 양수=좌회전). 신뢰반경 밖이면 None(=VLM 필요)."""
    if not last_seen:
        return None
    seen_pose = last_seen.get("pose") or {}
    drift = math.hypot(
        float(pose["x"]) - float(seen_pose.get("x", 0.0)),
        float(pose["y"]) - float(seen_pose.get("y", 0.0)),
    )
    if drift > LAST_SEEN_MAX_DRIFT_M:
        return None
    return _angle_diff_deg(float(last_seen.get("world_heading_deg", 0.0)), float(pose.get("yaw_deg", 0.0)))


def _estimate_sign_distance(area_frac: float | None) -> float | None:
    """bbox 면적 비율 → 대략적 거리(m): d = K/√area_frac (면적 ∝ 1/d²의 역산).

    단안 카메라의 조야한 sensor model이라 ±수십% 오차를 전제합니다 — anchor는 '대략 그
    부근'만 맞으면 충분하고(최종 도착 판정은 어차피 VLM+색블롭 게이트가 책임), 가까운
    목격일수록 큰 융합 가중치가 걸려 접근할수록 점점 정밀해집니다. 결측/0이면 None.
    """
    if not area_frac or area_frac <= 0.0:
        return None
    return min(max(PAD_SIGN_DIST_K / math.sqrt(float(area_frac)), PAD_ANCHOR_MIN_D), PAD_ANCHOR_MAX_D)


def _project_point(pose: dict[str, float], face_turn_deg: float, dist_m: float) -> dict[str, float]:
    """현재 pose에서 face_turn 방향(양수=좌)으로 dist만큼 떨어진 지점의 world 좌표.

    world 방위 = yaw + face_turn (_make_last_seen의 world_heading 규약과 동일),
    좌표계는 로봇 자신의 odometry 프레임입니다.
    """
    heading_rad = math.radians(float(pose.get("yaw_deg", 0.0)) + float(face_turn_deg))
    return {
        "x": float(pose["x"]) + float(dist_m) * math.cos(heading_rad),
        "y": float(pose["y"]) + float(dist_m) * math.sin(heading_rad),
    }


def _anchor_weight(confidence: float, dist_m: float) -> float:
    """anchor 융합 가중치 = confidence / max(거리, 1m).

    가까운 목격일수록 bbox가 커서 거리 추정의 상대 오차가 작으므로 더 크게 신뢰합니다.
    confidence 하한 0.1은 conf 결측(0.0) 목격도 미미하게나마 반영되게 합니다.
    """
    return max(float(confidence), 0.1) / max(float(dist_m), 1.0)


def _fuse_anchor(
    anchor: dict[str, Any] | None,
    point: dict[str, float],
    weight: float,
) -> dict[str, Any]:
    """목격 지점 추정치를 가중 평균으로 누적해 anchor를 갱신합니다(없으면 초기화).

    w_sum을 PAD_ANCHOR_W_CAP으로 클램프해 옛 목격 더미가 새 목격을 압도하지 못하게
    합니다 — 초기 오추정 anchor도 재목격 몇 번이면 실제 위치 쪽으로 씻겨 갑니다.
    이상치 기각: 목격 2회 이상으로 자리잡은 평균에서 PAD_ANCHOR_OUTLIER_M 넘게 벗어난
    새 점은 오독(다른 표지를 목표 글자로 착각)일 공산이 커 기각합니다 — pad는 움직이지
    않으므로 정상 목격은 항상 같은 부근에 모입니다.
    """
    w_new = max(float(weight), 0.0)
    has_anchor = anchor is not None and float(anchor.get("w_sum", 0.0)) > 0.0
    if has_anchor and w_new <= 0.0:
        return anchor  # 0-가중 갱신은 정보가 없으므로 기존 anchor 유지.
    if not has_anchor:
        return {"x": float(point["x"]), "y": float(point["y"]), "w_sum": w_new, "n": 1}
    if int(anchor.get("n", 0)) >= 2:
        off = math.hypot(float(point["x"]) - float(anchor["x"]), float(point["y"]) - float(anchor["y"]))
        if off > PAD_ANCHOR_OUTLIER_M:
            return anchor  # 자리잡은 평균에서 크게 벗어난 목격은 오염으로 보고 기각.
    w_old = min(float(anchor["w_sum"]), PAD_ANCHOR_W_CAP)
    w_total = w_old + w_new
    return {
        "x": (float(anchor["x"]) * w_old + float(point["x"]) * w_new) / w_total,
        "y": (float(anchor["y"]) * w_old + float(point["y"]) * w_new) / w_total,
        "w_sum": min(w_total, PAD_ANCHOR_W_CAP),
        "n": int(anchor.get("n", 0)) + 1,
    }


def _triangulate_rays(
    r1: dict[str, float],
    r2: dict[str, float],
    *,
    min_baseline_m: float = TRI_MIN_BASELINE_M,
    min_angle_deg: float = TRI_MIN_ANGLE_DEG,
    max_range_m: float = TRI_MAX_RANGE_M,
) -> dict[str, float] | None:
    """두 관측 ray(관측 지점 + world 방위각)의 교점으로 표지의 world 좌표를 추정합니다.

    단안 bbox 거리 추정(±수십% 요동, 라이브 실측 2.9~6.0m)을 신뢰하지 않고, 서로 다른 두
    지점에서 잰 '방위각만'으로 위치를 확정하는 고전 bearing-only 삼각측량입니다. 기하가
    불량하면(짧은 기선, 준평행 교각, 관측 뒤쪽 교점, 창고 규모 밖 거리) None을 반환해
    호출부가 동결하지 않게 합니다 — 특히 t<0(관측 뒤쪽) 기각은 서로 모순인 ray 쌍(한쪽이
    오독)을 자연히 걸러냅니다(라이브의 (+2.58,-3.69) 가짜 목격 쌍이 정확히 이 경우).
    """
    bx = float(r2["x"]) - float(r1["x"])
    by = float(r2["y"]) - float(r1["y"])
    if math.hypot(bx, by) < min_baseline_m:
        return None
    if abs(_angle_diff_deg(float(r1["bearing_deg"]), float(r2["bearing_deg"]))) < min_angle_deg:
        return None
    a1 = math.radians(float(r1["bearing_deg"]))
    a2 = math.radians(float(r2["bearing_deg"]))
    d1x, d1y = math.cos(a1), math.sin(a1)
    d2x, d2y = math.cos(a2), math.sin(a2)
    # p1 + t1*d1 = p2 + t2*d2 의 2x2 선형계. 행렬 [d1 | -d2]의 행렬식.
    det = d1x * (-d2y) - (-d2x) * d1y
    if abs(det) < 1e-9:
        return None
    t1 = (bx * (-d2y) - (-d2x) * by) / det
    t2 = (d1x * by - d1y * bx) / det
    if t1 <= 0.0 or t2 <= 0.0:  # 교점이 어느 한 관측의 뒤쪽 → 모순 쌍.
        return None
    if t1 > max_range_m or t2 > max_range_m:
        return None
    return {"x": float(r1["x"]) + t1 * d1x, "y": float(r1["y"]) + t1 * d1y}


def _cluster_sign_intersections(
    points: list[dict[str, float]],
    *,
    max_radius_m: float = SIGN_CLUSTER_MAX_RADIUS_M,
) -> dict[str, Any] | None:
    """유효 ray쌍 교점들의 '클러스터 합의'로 진짜 목표를 뽑습니다(오염 교점 기각). 순수 함수.

    동일 문자 표지가 다수 실재하면 같은 실물을 겨눈 쌍들의 교점은 밀집하고, 실물을 섞은 쌍의
    교점은 군집에서 이탈합니다. 각 점 i를 중심 후보로 반경 max_radius_m 안의 점 집합(자기 포함)을
    후보 군집으로 세어, inlier 최대를 1위(동수면 낮은 index 우선 — 결정적, pytest 잠금 전제),
    1위 inlier를 제외한 나머지 점에서 최대를 2위로 잡습니다(O(n²), n ≤ C(12,2)=66). tie 판단은
    이 함수가 하지 않고 n1/n2만 보고합니다(호출부 몫). 재투영 잔차는 구성상 0이라(오염 쌍도 0)
    어떤 경로에도 잔차 개념을 넣지 않습니다(§2-3).

    반환: {"goal": 1위 inlier들의 성분별 중앙값, "inliers", "n1", "n2", "outliers"} 또는
    None(len<2, 또는 1위 군집 크기<2 — 단일 점은 합의가 아님 → 군집 미형성).
    """
    n = len(points)
    if n < 2:
        return None
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]

    def _members(center_idx: int, allowed: range | set[int]) -> list[int]:
        cx, cy = xs[center_idx], ys[center_idx]
        return [j for j in allowed if math.hypot(xs[j] - cx, ys[j] - cy) <= max_radius_m]

    # 1위 군집: 낮은 index부터 검사해 동수 tie에서 낮은 index가 이깁니다(결정성 — > 비교라 첫 최대 유지).
    best1: list[int] = []
    for i in range(n):
        c = _members(i, range(n))
        if len(c) > len(best1):
            best1 = c
    if len(best1) < 2:
        return None
    inlier_set = set(best1)
    remaining = set(range(n)) - inlier_set
    # 2위 군집: 1위 inlier를 뺀 나머지 점에서만(중심·이웃 모두 remaining) 최대. 없으면 n2=0.
    best2: list[int] = []
    for i in sorted(remaining):
        c = _members(i, remaining)
        if len(c) > len(best2):
            best2 = c
    outliers = [{"x": xs[j], "y": ys[j]} for j in range(n) if j not in inlier_set]
    return {
        "goal": {
            "x": statistics.median(xs[j] for j in best1),
            "y": statistics.median(ys[j] for j in best1),
        },
        "inliers": [{"x": xs[j], "y": ys[j]} for j in best1],
        "n1": len(best1),
        "n2": len(best2),
        "outliers": outliers,
    }


def _add_sign_ray(
    sign_rays: dict[str, list[dict[str, float]]],
    letter: str,
    ray: dict[str, float],
    *,
    keep: int = SIGN_RAYS_KEEP,
) -> None:
    """표지 관측 ray를 letter별로 축적합니다(오래된 것부터 keep 초과분 폐기)."""
    rays = sign_rays.setdefault(letter, [])
    rays.append({
        "x": float(ray["x"]), "y": float(ray["y"]),
        "bearing_deg": float(ray["bearing_deg"]), "conf": float(ray.get("conf", 0.0)),
    })
    del rays[:-keep]


def _maybe_freeze_sign_goal(
    sign_rays: dict[str, list[dict[str, float]]],
    sign_goals: dict[str, dict[str, float]],
    letter: str,
    *,
    cooldown: dict[str, int] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """letter의 ray들을 '클러스터 합의'로 동결합니다(오염 교점 기각). 이미 동결이면 sticky 불변.

    선행 스펙의 '첫 유효 교점' 동결은 서로 다른 실물의 ray를 실물 구분 없이 짝지어 오염 좌표를
    굳힐 수 있었습니다(run2 anchor 1.6m 오차). 재투영 잔차는 구성상 0이라 오염을 못 걸러내므로,
    다중 쌍 교점의 클러스터 합의로만 분리합니다(§2-3). 판정 순서(위→아래, 하나 성립 시 종료):

    1) frozen 존재 → 그대로 반환(sticky — 기존 계약 유지).
    2) 쿨다운 > 0 → 1 감소 후 None(폐기 직후 데코이 재응집 차단). 감소 단위 = 이 함수 호출
       (= 유효 ray 추가 이벤트)이지 VLM look 총수가 아닙니다.
    3) ray < 3 → 클러스터 호출조차 하지 않고 None(pending_rays). 2-ray 단발 교점 동결은
       phantom을 굳히므로 절대 금지 — 임시 조준점은 기존 last_seen/비frozen anchor 경로가 담당합니다
       (신규 필드·저장 없음: 'frozen 미설치'가 곧 임시 조준점 모드).
    4) ray ≥ 3 → 모든 쌍의 유효 교점을 _cluster_sign_intersections에 넘겨:
       - 군집 미형성 → None(no_cluster, ray 계속 수집).
       - 2위 군집이 1위의 TIE_RATIO 이상 → 동결 보류(hold, 다수 미확정 — 다수결 함정 방어).
       - 그 외 → 1위 군집 중앙값으로 동결(frozen).

    state(관측 전용 out-param)에 status/n1/n2/n_outlier/remaining/rays를 실어 호출부 trace로
    넘깁니다(발행은 호출부 몫). cooldown은 memory.sign_refreeze_block을 전달받아 순수성을 유지합니다.
    """
    if state is not None:
        state.clear()
    frozen = sign_goals.get(letter)
    if frozen is not None:
        if state is not None:
            state["status"] = "frozen_sticky"
        return frozen
    if cooldown is not None and cooldown.get(letter, 0) > 0:
        cooldown[letter] -= 1  # 감소 단위 = freeze 시도(유효 ray 추가), VLM look 총수 아님.
        if state is not None:
            state["status"] = "cooldown"
            state["remaining"] = cooldown[letter]
        return None
    rays = sign_rays.get(letter) or []
    if len(rays) < 3:
        # frozen 승격 절대 금지 — 2-ray phantom (-1.39,-4.72)류를 굳힐 수 있습니다(계획서 R2 MAJOR-c).
        if state is not None:
            state["status"] = "pending_rays"
            state["rays"] = len(rays)
        return None
    points: list[dict[str, float]] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            pt = _triangulate_rays(rays[i], rays[j])  # 무수정 재사용(기하 유효 쌍만 통과).
            if pt is not None:
                points.append(pt)
    cluster = _cluster_sign_intersections(points)
    if cluster is None:
        if state is not None:
            state["status"] = "no_cluster"
        return None
    n1, n2 = cluster["n1"], cluster["n2"]
    if n2 >= n1 * SIGN_CLUSTER_TIE_RATIO:
        # 두 실물 군집이 팽팽 → 최대 군집이 데코이여도 잘못 동결하지 않습니다(라운드 2 MAJOR-b).
        if state is not None:
            state["status"] = "hold"
            state["n1"] = n1
            state["n2"] = n2
        return None
    goal = cluster["goal"]
    sign_goals[letter] = goal
    if state is not None:
        state["status"] = "frozen"
        state["n1"] = n1
        state["n2"] = n2
        state["n_outlier"] = len(cluster["outliers"])
    return goal


def _record_cube_sighting(
    sightings: list[dict[str, float]],
    detection: Any,
    pose: dict[str, float],
    *,
    keep: int = CUBE_SIGHTINGS_KEEP,
) -> None:
    """clean cube blob의 world 방위(pose.yaw − detection.angle_deg)·면적·관측 pose를 축적합니다.

    _add_sign_ray와 동형: 오래된 것부터 keep 초과분을 폐기합니다. world 방위 부호 규약은 표지 ray와
    동일(yaw − image_angle, head=0 전제의 서베이 스윕에서 호출) — 원거리 큐브라도 방위만 잡으면
    source-seek 우선순위 ①의 타깃이 됩니다. ctx 없음(perceive 결과+pose를 인자로 받음)이라 pytest로 잠급니다.
    """
    sightings.append({
        "bearing_deg": float(pose.get("yaw_deg", 0.0)) - float(detection.angle_deg),
        "area": float(detection.blob_area),
        "x": float(pose["x"]),
        "y": float(pose["y"]),
    })
    del sightings[:-keep]


def _source_target_priority(memory: AgentMemory) -> tuple[str, Any | None]:
    """source-seek 접근 우선순위 결정(순수, memory 읽기 전용): (kind, payload) 반환.

    우선순위(§5.2): ① cube — cube_sightings 중 최대 면적(동률이면 최신) blob → ("cube", sighting dict).
    ② goal — sign_goals['A'] 동결점 → ("goal", 좌표 dict). ③ ray — sign_rays['A']가 있으면 최신 방위
    → ("ray", bearing_deg)(이동이 기선을 만들어 2차 목격 시 자동 동결). ④ fallback — 셋 다 없으면
    ("fallback", None). place·navpad는 절대 이 함수를 쓰지 않습니다(획득 전용).
    """
    sightings = memory.cube_sightings
    if sightings:
        best_i = max(range(len(sightings)), key=lambda i: (float(sightings[i].get("area", 0.0)), i))
        return ("cube", sightings[best_i])
    goal = _source_goal(memory)  # 'A' 동결은 이 단일 소비 통로로만 읽습니다(§7.8 계약).
    if goal is not None:
        return ("goal", goal)
    rays = memory.sign_rays.get("A") or []
    if rays:
        return ("ray", float(rays[-1]["bearing_deg"]))
    return ("fallback", None)


def _source_goal(memory: AgentMemory) -> dict[str, float] | None:
    """동결된 소스 'A' 목표 좌표를 반환합니다(순수). 'A'는 DESTINATION_SIGN_RULES에 없어(color 매핑
    부재) pad_memory에 등재되지 않으므로, destination 게이트 밖에서 'A' 동결을 소비하는 유일한 통로입니다.
    SIGNAGE_NOTE/DESTINATION_SIGN_RULES는 절대 수정하지 않습니다.
    """
    return memory.sign_goals.get("A")


def _drop_sign_map(memory: AgentMemory | None, letter: str) -> None:
    """anchor 폐기(근접 미검출 반복) 시 letter의 동결 목표·ray를 함께 버려 재구축을 허용합니다.

    동결 좌표가 틀렸다는 증거(그 자리에서 VLM 연속 미검출)가 쌓이면 지도를 그 letter만
    초기화합니다 — 나쁜 삼각측량(한쪽 ray가 오독이었던 쌍)이 영구히 로봇을 홀리지 않게 하는
    기존 anchor 자가치유와 같은 원리입니다. M1: 폐기 직후 같은 데코이가 즉시 재응집하는 루프를
    막기 위해 재동결 쿨다운도 함께 마크합니다 — M4의 residual degrade 콜사이트가 이 함수를
    재사용하므로 그 경로에서도 쿨다운이 자동 적용됩니다.
    """
    if memory is None:
        return
    memory.sign_goals.pop(letter, None)
    memory.sign_rays.pop(letter, None)
    memory.sign_refreeze_block[letter] = SIGN_REFREEZE_COOLDOWN_LOOKS


def _sign_candidates(
    detections: list[Any],
    *,
    min_area: int = SURVEY_CAND_MIN_AREA,
    min_fill: float = SURVEY_CAND_MIN_FILL,
    max_cy_frac: float = SURVEY_CAND_MAX_CY_FRAC,
    aspect: tuple[float, float] = SURVEY_CAND_ASPECT,
) -> list[Any]:
    """OpenCV 검출 중 'pad 표지'로 보이는 후보만 고릅니다(VLM 호출 게이트, ~0원 비용).

    표지의 물리 prior 3가지: 공중 부양(중심이 프레임 상부 — 바닥 큐브·스폰 링 배제),
    단색 채움 사각(fill 높음 — 뼈대 랙·레일 배제), 대략 정사각(가로 밴드 배제). 프레임
    높이는 해상도 하드코딩 없이 관찰값에서 유도(_frame_width_from × 9/16)합니다.
    """
    out: list[Any] = []
    for d in detections:
        bbox = getattr(d, "bbox", None)
        centroid = getattr(d, "centroid", None)
        if not bbox or not centroid:
            continue
        _, _, w, h = bbox
        if w <= 0 or h <= 0 or d.blob_area < min_area:
            continue
        fill = d.blob_area / (w * h)
        if fill < min_fill:
            continue
        if not (aspect[0] <= w / h <= aspect[1]):
            continue
        fw = _frame_width_from(d)
        frame_h = fw * 9.0 / 16.0 if fw else 720.0
        if centroid[1] / frame_h > max_cy_frac:
            continue
        out.append(d)
    return out


def _side_name(side: float) -> str:
    """우회 방향 부호(+1=좌회전)를 실패 카운터 키로 변환합니다."""
    return "left" if side > 0 else "right"


def _choose_detour_side(preferred: float, fails: dict[str, int]) -> float:
    """우회 방향 선택: 실패 이력이 적은 쪽을 고르고, 같으면 preferred(타깃 쪽)를 유지합니다."""
    other = -preferred
    if fails.get(_side_name(preferred), 0) > fails.get(_side_name(other), 0):
        return other
    return preferred


def _near_known_stall(
    spots: list[dict[str, float]],
    x: float,
    y: float,
    yaw_deg: float,
    *,
    radius_m: float = STALL_SPOT_RADIUS_M,
    heading_tol_deg: float = STALL_HEADING_TOL_DEG,
) -> bool:
    """기억된 stall 지점(위치+당시 진행 방향) 근처에서 같은 방향으로 또 전진하려는지 검사합니다.

    방향까지 비교하는 이유: 장애물은 stall 당시 진행 방향 앞에 있으므로, 같은 지점이라도
    다른 방향의 전진은 막지 않아야 우회 경로 자체가 봉쇄되지 않습니다.
    """
    for s in spots:
        if math.hypot(x - float(s["x"]), y - float(s["y"])) > radius_m:
            continue
        if abs(_angle_diff_deg(yaw_deg, float(s.get("yaw_deg", yaw_deg)))) <= heading_tol_deg:
            return True
    return False


def _preferred_side_from_history(
    wins: list[dict[str, float]],
    x: float,
    y: float,
    yaw_deg: float,
    *,
    radius_m: float = STALL_SPOT_RADIUS_M,
    heading_tol_deg: float = STALL_HEADING_TOL_DEG,
) -> float | None:
    """과거에 우회가 성공했던 지점·방향 근처면 그때의 우회 side(+1/-1)를 반환합니다.

    stall 지점 기억(_near_known_stall)의 쌍둥이: 실패는 피하고 성공은 재사용합니다.
    같은 구조물을 라운드 안에서 반복 통과할 때 좌/우 탐색 없이 한 번에 뚫립니다.
    최근 기록 우선(뒤에서부터 검색) — 같은 지점의 오래된 기록보다 새 경험을 신뢰합니다.
    """
    for win in reversed(wins):
        if math.hypot(x - float(win["x"]), y - float(win["y"])) > radius_m:
            continue
        if abs(_angle_diff_deg(yaw_deg, float(win.get("yaw_deg", yaw_deg)))) <= heading_tol_deg:
            return float(win["side"])
    return None


def _sign_bbox_area_frac(target: dict[str, Any]) -> float | None:
    """VLM sign 검출의 bbox 면적을 프레임 대비 비율(0~1)로 환산합니다(면적 ∝ 1/d² 대용).

    qwen bbox_2d는 0~1000 정규화 좌표(_sign_offset_deg와 동일 규약). 범위를 벗어나거나
    형식이 깨지면 None — 소비자(수렴 판정 2차 신호)는 결측을 무시합니다.
    """
    bbox = target.get("bbox_2d") or target.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        if 0.0 <= x1 < x2 <= VLM_BBOX_SCALE and 0.0 <= y1 < y2 <= VLM_BBOX_SCALE:
            return ((x2 - x1) / VLM_BBOX_SCALE) * ((y2 - y1) / VLM_BBOX_SCALE)
    return None


def _approach_converging(
    history: list[dict[str, Any]],
    *,
    min_samples: int = APPROACH_MIN_SAMPLES,
    growth_min: float = APPROACH_AREA_GROWTH_MIN,
) -> bool | None:
    """접근 수렴 판정(사용자 스펙 distance_error_rate의 단안 proxy) — 현재 관측 모드 전용.

    history는 접근 반복마다의 {"area": 색블롭 면적, "face_turn": 조준각(도)} 표본입니다.
    1차 신호: 면적 ∝ 1/d²이므로 후반 표본의 면적 중앙값이 전반 대비 growth_min배 이상이면
    거리가 줄고 있는 것(수렴). 중앙값 비교라 green flicker 단발 노이즈에 강합니다.
    2차 신호: 면적이 안 늘어도 |face_turn|이 첫 표본보다 줄어 정면 허용각 안이면 조준 수렴.
    반환: None=표본 부족(판정 불가), True=수렴 중, False=비수렴(전진해도 진전 없음).
    """
    samples = [h for h in history if h.get("area") is not None]
    if len(samples) < min_samples:
        return None
    areas = [float(h["area"]) for h in samples]
    half = len(areas) // 2
    early, late = sorted(areas[:half]), sorted(areas[half:])
    early_med = early[len(early) // 2]
    late_med = late[len(late) // 2]
    if late_med > 0 and (early_med == 0 or late_med >= early_med * growth_min):
        return True
    turns = [abs(float(h["face_turn"])) for h in history if h.get("face_turn") is not None]
    if len(turns) >= 2 and turns[-1] <= PAD_FACE_TOL_DEG and turns[-1] < turns[0]:
        return True
    return False


def _should_commit_route(route_stats: dict[str, Any], began_new_trace: bool) -> bool:
    """R2 단순 가드: 이번 cycle의 경로 커밋 허용 여부.

    - t0가 없으면 추적된 배송이 아님(시작 전이거나 이미 커밋됨) → 금지.
    - 같은 cycle에 새 배송 trace가 방금 시작됐으면(began_new_trace) delivered 증가는
      '지연 도착한 직전 배송' 신호이고 현재 stats는 새 배송 것 → 커밋하면 score≈0
      쓰레기 경로가 best_route(min-score)로 영구 고정되므로 금지(학습 1건 생략을 감수).
    """
    if began_new_trace:
        return False
    return route_stats.get("t0") is not None


def _pad_memory_entry(pad_memory: dict[str, dict[str, Any]], color: str) -> dict[str, Any]:
    """색상별 pad 기억 슬롯을 얻거나 만듭니다."""
    return pad_memory.setdefault(
        color,
        {
            "last_seen": None,
            "anchor": None,  # sign의 world '점' 추정치 {x, y, w_sum, n} — ray와 달리 회전에 불변.
            "successful_routes": [],
            "failed_routes": [],
            "best_route": None,
        },
    )


def _commit_successful_route(
    entry: dict[str, Any],
    waypoints: list[dict[str, float]],
    stats: dict[str, Any],
    drop_pose: dict[str, float] | None,
) -> dict[str, Any]:
    """배송 성공 경로를 저장하고 best_route(score 최소)를 갱신합니다."""
    route = {
        "waypoints": waypoints,
        "score": _route_score(stats),
        "stats": dict(stats),
        "drop_pose": drop_pose,
    }
    entry.setdefault("successful_routes", []).append(route)
    entry["best_route"] = _select_best_route(entry["successful_routes"])
    return route


def _record_failed_route(entry: dict[str, Any], stats: dict[str, Any], reason: str) -> None:
    """실패한 pad 접근의 통계를 진단용으로 남깁니다(최근 FAILED_ROUTES_KEEP개만 보관)."""
    failed = entry.setdefault("failed_routes", [])
    failed.append({"stats": dict(stats), "reason": reason})
    del failed[:-FAILED_ROUTES_KEEP]


def _bump_stat(memory: AgentMemory | None, key: str, amount: float = 1) -> None:
    """현재 배송 route_stats의 카운터를 증가시킵니다(memory 없으면 no-op)."""
    if memory is None:
        return
    memory.route_stats[key] = memory.route_stats.get(key, 0) + amount


# --- M0: route_trace 파일 싱크(세션-잔존 로그) ---
# 인메모리 route_trace는 update_memory가 배송 경계에서 리셋하고 Ctrl-C·세션 사망에 통째로
# 소실됩니다. 후속 라이브 분석(M3)이 세션 사망에도 데이터를 잃지 않도록, 매 trace step을 JSONL
# 1줄로 파일에도 append합니다. 경로는 프로세스당 1회 결정(MENLO_TRACE_FILE env 우선, 없으면
# outputs/route_trace_<UTC>.jsonl; outputs/는 gitignored). 관측 전용 — 배포 에이전트의 행동
# 경로는 전혀 바뀌지 않으므로 평가 당일 켜져 있어도 무해합니다(Level 2 합법). 쓰기 실패는
# no-op으로 삼켜 로깅이 런을 죽이지 않게 하고(크래시 안전), 첫 실패에서 경고 1줄 + 전역
# disabled로 이후 스팸을 막습니다.
_TRACE_FILE_PATH: str | None = None
_TRACE_FILE_RESOLVED = False
_TRACE_FILE_DISABLED = False


def _trace_file_path() -> str | None:
    """프로세스당 1회 trace 파일 경로를 결정해 캐시합니다(env 우선, 없으면 outputs/ 타임스탬프)."""
    global _TRACE_FILE_PATH, _TRACE_FILE_RESOLVED
    if _TRACE_FILE_RESOLVED:
        return _TRACE_FILE_PATH
    _TRACE_FILE_RESOLVED = True
    env_path = os.environ.get("MENLO_TRACE_FILE")
    if env_path:
        _TRACE_FILE_PATH = env_path
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        _TRACE_FILE_PATH = os.path.join("outputs", f"route_trace_{stamp}.jsonl")
    return _TRACE_FILE_PATH


def _trace_step(memory: AgentMemory | None, **fields: Any) -> None:
    """현재 배송 route_trace에 step 하나를 기록하고, 같은 레코드를 JSONL로 파일에 append합니다.

    한 배송 trace는 보통 수십 step이지만, 비정상 장기 배회에 대비해 인메모리 상한을 둡니다
    (초과 시 앞부분을 버리므로 그 배송의 waypoint 초반부가 소실될 수 있음 — 허용). 파일에는
    상한이 없고 공통 필드 ts(epoch)가 전순서를 보장하므로, 리셋·세션 사망 뒤에도 전체 이력이
    파일에 남습니다(step은 리셋 후 0 재시작 가능 — 파일에서는 ts가 정렬 키).
    """
    global _TRACE_FILE_DISABLED
    if memory is None:
        return
    fields.setdefault("step", len(memory.route_trace))
    fields.setdefault("ts", time.time())  # 파일 내 전순서 키(step 리셋과 무관).
    memory.route_trace.append(fields)  # 인메모리 append는 항상 선행(파일 실패와 독립).
    if len(memory.route_trace) > 1000:
        del memory.route_trace[:200]
    if _TRACE_FILE_DISABLED:
        return
    try:
        path = _trace_file_path()
        if not path:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)  # outputs/가 없어도 자가 생성(자가 치유).
        line = json.dumps(fields, ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8") as fh:  # append-close per step: 크래시 안전.
            fh.write(line + "\n")
    except Exception as exc:  # 로깅이 런을 죽이면 안 됨 — 첫 실패에서 끄고 경고 1줄.
        _TRACE_FILE_DISABLED = True
        print(f"  [trace] 파일 싱크 비활성화(쓰기 실패): {type(exc).__name__}: {exc}")


def _trace_freeze(memory: AgentMemory | None, letter: str, state: dict[str, Any]) -> None:
    """M1: _maybe_freeze_sign_goal의 state(status)를 §6 freeze_* trace로 발행합니다(관측 전용).

    두 콜사이트(_look_for_sign·survey_pads)가 공유하는 status→trace 매핑입니다. 이미 동결된
    frozen_sticky는 재발행하지 않습니다 — freeze_commit으로 1회만 기록해 스팸을 막습니다.
    """
    status = state.get("status")
    if status == "frozen":
        _trace_step(
            memory, action="freeze_commit", letter=letter,
            goal=memory.sign_goals.get(letter) if memory is not None else None,
            n1=state.get("n1"), n2=state.get("n2"), n_outlier=state.get("n_outlier"),
            rays=len(memory.sign_rays.get(letter) or []) if memory is not None else None,
        )
    elif status == "hold":
        _trace_step(memory, action="freeze_hold", letter=letter, n1=state.get("n1"), n2=state.get("n2"))
    elif status == "pending_rays":
        _trace_step(memory, action="freeze_pending", letter=letter, rays=state.get("rays"))
    elif status == "no_cluster":
        _trace_step(memory, action="freeze_reject", letter=letter, reason="no_cluster")
    elif status == "cooldown":
        _trace_step(memory, action="freeze_cooldown", letter=letter, remaining=state.get("remaining"))


def _record_stall(memory: AgentMemory | None, pose: dict[str, float]) -> None:
    """stall 발생 지점(위치+진행 방향)을 기록합니다 — 이후 같은 방향 재돌진을 선제 우회."""
    if memory is None:
        return
    _bump_stat(memory, "stalls", 1)
    memory.stall_spots.append(
        {"x": float(pose["x"]), "y": float(pose["y"]), "yaw_deg": float(pose["yaw_deg"])}
    )
    del memory.stall_spots[:-40]


async def _get_pose(ctx: Any) -> dict[str, float]:
    """odometry pose {x, y, yaw_deg}를 한 번의 상태 읽기로 얻습니다(고유수용성; scene 아님)."""
    return _pose_dict(await get_robot_status(ctx))


async def _probe_free_space(ctx: Any) -> dict[str, Any] | None:
    """현재 POV 프레임으로 발 앞 바닥 free-space 프로파일을 얻습니다(선제 장애물 인지).

    get_vision 1장(캡처 실측 ~30ms)만 쓰며 VLM(6~32s)이 아닙니다 — 전진 직전 매 청크에 넣어도
    지배 비용이 아닙니다. cv2/numpy 디코드나 프레임 획득이 실패하면 None을 돌려주어(호출부는
    무시) 선제 판정이 없던 것처럼 반응형(odometry stall) 경로로 안전하게 폴백합니다.
    """
    try:
        import cv2
        import numpy as np

        jpeg = await ctx.get_vision("pov")
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        value = img.max(axis=2)  # V(HSV)=채널별 최대 밝기. 바닥은 어둡고 구조물은 밝다.
        return free_space_profile(value)
    except Exception:
        return None


async def _belt_blocks_forward(ctx: Any, memory: AgentMemory | None) -> bool:
    """전방이 벨트(초대형 belt_color blob)에 막혔는지 카메라로 판정합니다(순수 인지, VLM 0).

    벨트는 프레임을 가로지르는 초대형 blob(area > CUBE_ARRIVAL_AREA×MAX_AREA_ARRIVAL_MULT)으로
    나타나며 _detect_belt_color가 기억한 색과 일치하고, 정면 접근 중이면 화면 중앙 근처에 옵니다.
    이게 전방에 크게 있으면 '들이받아 wedge된 뒤 반응형 우회가 양측막힘으로 실패'(run3 확정)하기
    전에, 호출부가 능동적으로 free-space 쪽 벨트-따라가기로 전환하도록 True를 돌려줍니다. belt_color
    미확정이면 False(선제 판정 없음 → 기존 반응형 stall 경로로 안전 폴백). cube 크기 vs 벨트 너비를
    area로 구분하는 기존 규약(_is_clean_cube/_detect_belt_color)을 그대로 재사용 — 좌표 0(§0/§7).
    """
    if memory is None or memory.belt_color is None:
        return False
    thresh = CUBE_ARRIVAL_AREA * MAX_AREA_ARRIVAL_MULT
    for d in await perceive(ctx):
        if (
            d.color == memory.belt_color
            and d.blob_area > thresh
            and abs(d.angle_deg) <= CENTER_TOLERANCE_DEG * 2.0
        ):
            return True
    return False


async def _get_yaw_deg(ctx: Any) -> float:
    """로봇 자신의 body yaw(도)를 읽습니다.

    이는 고유수용성(gyro/IMU 상당)이며 scene_state가 아닙니다. 폐루프 회전에서 '상대 변화량'만
    쓰므로 Level 2에서 합법입니다(큐브/pad 위치 같은 scene 정보는 전혀 사용하지 않음).
    """
    status = await get_robot_status(ctx)
    pose = getattr(getattr(status, "robot", None), "pose", None)
    return float(getattr(pose, "yaw_deg", 0.0) or 0.0)


def _angle_diff_deg(a: float, b: float) -> float:
    """a-b를 (-180, 180] 범위로 정규화한 차이(도)."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


async def _turn_by_deg(ctx: Any, delta_deg: float) -> None:
    """아크(vx>0 + wz)로 body를 상대 delta_deg만큼 회전합니다(양수=좌회전).

    이 학습 정책은 제자리 회전이 불가하고 짧은 명령은 ramp-up으로 거의 안 돌기 때문에, 개루프
    회전은 부정확합니다. 그래서 로봇 자신의 yaw 피드백으로 목표에 닿을 때까지 아크를 반복하는
    폐루프로 돌립니다(상대 변화량만 사용 = gyro 적분과 동일, scene 정보 아님).
    """
    if abs(delta_deg) < 1.0:
        return
    target = (await _get_yaw_deg(ctx)) + delta_deg
    for _ in range(TURN_MAX_ARCS):
        remaining = _angle_diff_deg(target, await _get_yaw_deg(ctx))
        if abs(remaining) <= PAD_TURN_TOL_DEG:
            return
        wz = ARC_WZ if remaining > 0 else -ARC_WZ
        # 남은 각에 비례하되(과회전 방지) 램프업을 고려해 충분히 길게(짧은 아크는 거의 안 돎).
        dur = min(max(abs(remaining) / 25.0, 0.9), 1.6)
        await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=dur)


async def _pose_str(ctx: Any) -> str:
    """디버그용 위치/방향 문자열(x, y, yaw). 상대 이동 추적에만 씁니다(scene 정보 아님)."""
    status = await get_robot_status(ctx)
    pose = getattr(getattr(status, "robot", None), "pose", None)
    pos = getattr(pose, "position", None) or (0.0, 0.0, 0.0)
    yaw = float(getattr(pose, "yaw_deg", 0.0) or 0.0)
    return f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) yaw={yaw:+.0f}°"


async def _reverse_by(ctx: Any, target_m: float) -> float:
    """방금 장애물로 밀고 들어간 변위(target_m)만큼 폐루프로 후진해 충돌 직전 자세로 되돌립니다.

    선제 free-space 인지가 놓친 장애물에 부딪혀 stall(제자리 걸음)이 나면, 개루프 고정 시간 후진은
    실제 밀고 들어간 양과 무관해 과/부족 후퇴가 됩니다. 여기서는 매 짧은 후진 청크마다 odometry
    변위를 재어 누적 후퇴가 target_m에 닿을 때까지(또는 STALL_REVERSE_MAX_CHUNKS 안전 상한까지)
    반복합니다 — '전진한 만큼 되돌린다'를 속도 신뢰가 아니라 실측 이동량으로 보장합니다(청크
    granularity로 마지막 청크가 목표를 살짝 넘겨 완전히 빠져나옵니다). 반환은 실제 후퇴 총 거리(m).
    """
    if target_m <= 0.0:
        return 0.0
    backed = 0.0
    for _ in range(STALL_REVERSE_MAX_CHUNKS):
        p0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=-STALL_REVERSE_VX, duration_s=STALL_REVERSE_CHUNK_S)
        p1 = await _get_pose(ctx)
        backed += math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        if backed >= target_m:
            break
    return backed


async def _close_to_goal(
    ctx: Any,
    goal: dict[str, float],
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> float:
    """도착 선언 후 동결 목표까지 남은 거리를 소회전+짧은 전진으로 좁힙니다(place 반경 진입).

    표지 blob 도착 판정은 표지가 크고 높이 떠 있어 palette place 반경(1.2m) 밖에서도
    성립합니다(라이브 확정: blob 49k·conf 0.98 도착 지점이 동결 목표에서 1.47m → place 실패).
    협소 구역의 큰 아크는 전도 위험이 실측됐으므로 회전은 ±PAD_CLOSE_MAX_TURN_DEG 캡,
    전진은 PAD_CLOSE_CHUNK_S 짧은 청크만 씁니다. 전진이 stall하면 팔레트/랙에 닿은 것이므로
    더 밀지 않고 즉시 종료해 place에 맡깁니다. 반환: 종료 시점의 목표 잔여 거리(m).
    """
    for _ in range(PAD_CLOSE_MAX_CHUNKS):
        pose = await _get_pose(ctx)
        dist, turn = _face_turn_to(pose, goal)
        if dist <= PAD_CLOSE_ENOUGH_M:
            break
        if abs(turn) > PAD_FACE_TOL_DEG:
            await _turn_by_deg(
                ctx, max(-PAD_CLOSE_MAX_TURN_DEG, min(PAD_CLOSE_MAX_TURN_DEG, turn))
            )
        p0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=PAD_CLOSE_CHUNK_S)
        p1 = await _get_pose(ctx)
        moved = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        stalled = _is_stalled(FORWARD_EFF_SPEED_MPS * PAD_CLOSE_CHUNK_S, moved)
        _trace_step(
            memory, action="close_advance", pose=p1,
            actual_m=round(moved, 3), stall=stalled,
        )
        if stalled:
            break  # 팔레트/랙에 닿음 — 더 밀면 전도 위험, place가 판정하게 둡니다.
    pose = await _get_pose(ctx)
    dist, _ = _face_turn_to(pose, goal)
    if verbose:
        print(f"           마무리 접근 종료: 동결 목표 잔여 d={dist:.2f}m")
    return dist


def _push_flooded(area: float, ceiling: float = PAD_PUSH_FLOOD_AREA) -> bool:
    """서보 blob 면적이 flooding(구조물 밀착) 수준이면 True(순수 — pytest로 잠금).

    ★J(run15 전도)★ push-through가 벽/랙에 밀착하면 blob이 카메라를 가득 채워 area가 폭증한다
    (run15 213k). 실제 pad blob은 근접에서도 ~55k라 130k 상한은 정상 근접(Fix H 오버사이즈 추격
    40~55k)과 flooding을 명확히 가른다. flooding에서 전진하면 stall grind로 전도하므로 즉시 멈춘다.
    """
    return area >= ceiling


async def _push_through_to_target(
    ctx: Any,
    target_color: str,
    *,
    arrival_area: int = PAD_ARRIVAL_AREA,
    memory: AgentMemory | None = None,
    verbose: bool = False,
    max_advance_m: float | None = None,
    max_chunks: int | None = None,
    belt_guard: bool = True,
    oversize_ok: bool = False,
) -> float:
    """area_ok 도착 후 pick/place 반경(1.2m) 진입을 위한 push-through(§1 핸드오프 — 좌표·goal 불요).

    blob 면적 도착 게이트(CUBE_ARRIVAL_AREA/PAD_ARRIVAL_AREA·PICK_READY_AREA)는 반경 밖에서도
    참입니다 — 라이브 확정: pad 표지 blob은 1.34~1.47m, cube blob(27978)은 1.57m에서 이미 도착
    면적 충족인데 pick_entity/place는 1.2m 반경을 요구해 '거리 초과'로 실패(무동결이면 _close_to_goal/
    place-probe도 전부 우회 → run1/run2 0배달). 여기서는 target 색블롭만 서보(정면 정렬)하며 vx를
    단계 상향(정찰 vx0.5 돌파)해 빈 팔레트/구조물 stall을 뚫고 반경 안으로 들어갑니다. arrival_area는
    _best_color_blob의 유효 blob 범위만 결정(cube=CUBE_ARRIVAL_AREA, pad=PAD_ARRIVAL_AREA). 종료:
    (a) 누적 전진 PAD_PUSH_MAX_ADVANCE_M 도달(오버슈트·큐브 통과 차단), (b) vx 최대로도 stall = 벽
    (→ place/orbit·다음 cycle에 맡김), (c) 청크 상한, (d) blob 상실(상위 재관찰). 좌표 0 — setup
    1-50 무관(§0). 전도 방지로 회전·청크 캡을 _close_to_goal와 공유합니다.
    반환: 누적 전진 거리(m) — 호출부(E2)가 '전진 0 = 이 자리 차단' 판정에 씁니다.
    """
    vi = 0
    advanced = 0.0
    # 기본 예산은 도착 직후의 짧은 반경 진입용. D1b 거리-피드백 재전진은 실거리 기반의 더 긴
    # 예산을 명시적으로 넘깁니다(max_advance_m/max_chunks) — 기본 캡 0.7m로는 2.6m 미달을
    # 못 메꿔 run6처럼 '도착→실패→재도착→실패'만 반복하기 때문입니다.
    limit_m = PAD_PUSH_MAX_ADVANCE_M if max_advance_m is None else max_advance_m
    chunks = PAD_PUSH_MAX_CHUNKS if max_chunks is None else max_chunks
    for _ in range(chunks):
        if advanced >= limit_m:
            break
        # ★D2(run6 전도)★ pick push엔 벨트 가드가 없어 belt-scale blob을 향해 밀다 벨트 구조물에
        # 올라 전도했습니다(navpad 선제 가드 :4329는 navpad 전용이었음). 같은 판정을 매 청크 앞에
        # 넣어 전방이 벨트면 더 밀지 않고 중단합니다 — 반경 미달 실패는 다음 cycle이 수습하지만
        # 전도는 라운드를 통째로 태웁니다(run6: 80.4s에 종료, 잔여 519.6s 소실).
        # ★F(run8)★ 단, place 최종 진입 push는 호출부가 belt_guard=False로 끕니다: pad C 정면에선
        # 벨트 배경이 항상 중앙 초대형 blob으로 잡혀 가드가 마지막 0.5m 진입을 전부 차단했고
        # (0.00m ×11), place가 'not near pad'로만 돌았습니다. pick push(phantom 추격 위험)만 가드.
        if belt_guard and memory is not None and await _belt_blocks_forward(ctx, memory):
            _trace_step(memory, action="push_belt_abort", advanced_m=round(advanced, 2))
            if verbose:
                print("           push-through 중단: 전방 벨트 감지(전도 방지)")
            break
        blob = await _best_color_blob(ctx, target_color, arrival_area, allow_oversize=oversize_ok)
        if blob is None:
            break  # blob 상실 → 상위 루프 재관찰에 맡김(블라인드 전진 금지).
        # ★J(run15 전도)★ 구조물 밀착(flooding) 판정: blob이 flood 수준이면 벽/랙에 붙은 것이라
        # 어떤 vx로 밀어도 stall grind로 전도한다(run15: 213k에서 vx0.5→0.6 grind 2청크 후 전도).
        # 밀지 않고 즉시 중단 — place/다음 cycle이 수습하지만 전도는 라운드를 통째로 태운다.
        if _push_flooded(blob.blob_area):
            _trace_step(
                memory, action="push_flood_abort", area=blob.blob_area, advanced_m=round(advanced, 2),
            )
            if verbose:
                print("           push-through 중단: blob flooding(구조물 밀착) 감지(전도 방지)")
            break
        if abs(blob.angle_deg) > PAD_FACE_TOL_DEG:
            # angle_deg>0 = target 우측(cx>W/2)이고 _turn_by_deg(+)=좌회전이라, 정면 정렬은 -angle_deg
            # 만큼 회전해야 합니다(부호 정정 — 미정정 시 off-center에서 반대로 틀어 멀어짐).
            await _turn_by_deg(
                ctx, max(-PAD_CLOSE_MAX_TURN_DEG, min(PAD_CLOSE_MAX_TURN_DEG, -blob.angle_deg))
            )
        vx = PAD_PUSH_VX_LADDER[vi]
        p0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=vx, duration_s=PAD_CLOSE_CHUNK_S)
        p1 = await _get_pose(ctx)
        moved = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        advanced += moved
        # 기대 이동량은 vx에 비례(FORWARD_EFF_SPEED_MPS는 FORWARD_VX 기준 실효속도).
        expected = FORWARD_EFF_SPEED_MPS * (vx / FORWARD_VX) * PAD_CLOSE_CHUNK_S
        stalled = _is_stalled(expected, moved)
        _trace_step(
            memory, action="push_through", pose=p1, vx=vx,
            actual_m=round(moved, 3), stall=stalled, area=blob.blob_area,
        )
        if stalled:
            if vi < len(PAD_PUSH_VX_LADDER) - 1:
                vi += 1  # 단계 상향 재추진(정찰: vx0.5가 팔레트 stall 돌파).
            else:
                break  # vx 최대로도 stall = 벽 → 더 밀지 않고 place/orbit에 맡김(전도 방지).
    if verbose:
        print(f"           push-through 종료: 누적 전진 {advanced:.2f}m")
    return advanced


async def _advance_or_detour(
    ctx: Any,
    side: float,
    *,
    duration_s: float = PAD_ADVANCE_DUR,
    memory: AgentMemory | None = None,
    action: str = "advance",
    verbose: bool = False,
) -> bool:
    """전진 한 청크를 실행하되, 병진이 죽으면(구조물 stall) 후진+아크 우회로 전환합니다.

    학습 정책은 장애물에 막혀도 전진 명령을 수용해 병진 0·미세 회전만 남고, 호출부는 이를 몰라
    같은 자리를 배회합니다(라이브 확정: navpad가 x≈1.1 source ledge에 고착). 전진 전후
    odometry 거리(실제 이동량)를 속도×시간 기대 이동량과 비교(_is_stalled)해 stall을 감지하고,
    감지 시 전진한 만큼 폐루프 후퇴(_reverse_by)→side로 꺾음→우회 전진→역회전(재조준)으로
    장애물을 옆으로 비껴갑니다(bug-avoidance: back-off-turn-advance-turn-back). 후퇴를 '전진한
    만큼'으로 잡는 이유: 선제 free-space 인지가 놓친 충돌은 밀고 들어간 깊이가 매번 달라, 고정
    시간 후진은 과/부족 후퇴가 되기 때문입니다(선제 우회로 안 들이받은 경우는 짧은 고정 후진 유지).
    추가로 두 가지를 기록/활용합니다:
    - 모든 청크의 기대/실제 이동량·효율을 route_trace에 기록(사후 분석·경로 승격용).
    - stall 지점(위치+당시 방향)을 기억하고, 같은 지점·같은 방향 전진이면 직진을 생략하고
      선제 우회합니다(_near_known_stall) — 같은 구조물에 반복 돌진하는 낭비 제거.

    역회전이 필수인 이유: 우회각(PAD_STALL_DETOUR_DEG=50°)이 카메라 half-FOV(30°)를 넘어,
    꺾은 채로 두면 직전까지 마주보던 표지가 반드시 프레임 밖으로 나가고, not-found 복구의
    고정 +55° 회전이 같은 방향으로 오차를 누적시켜 재획득이 사실상 불가합니다. 원래 방위로
    되돌려 표지를 시야에 복귀시키고 다음 look의 bbox 비례 조향이 미세 보정하게 합니다.

    반환: 병진을 확보했으면(직진 성공 또는 우회 전진 성공) True, 우회 전진마저 stall이면
    False — 호출부는 False일 때만 side를 토글합니다(성공한 우회의 방향을 뒤집으면 진동함).
    """
    pose0 = await _get_pose(ctx)
    expected = FORWARD_EFF_SPEED_MPS * duration_s
    # 같은 지점·같은 방향에서 과거에 성공한 우회 방향이 있으면 호출부의 side보다 우선합니다
    # (실패 이력 회피와 대칭인 '성공 이력 재사용' — 좌/우 재탐색 없이 한 번에 통과).
    had_win_side = False
    if memory is not None:
        win_side = _preferred_side_from_history(
            memory.detour_wins, pose0["x"], pose0["y"], pose0["yaw_deg"]
        )
        if win_side is not None:
            had_win_side = True
            if win_side != side:
                if verbose:
                    print(f"           과거 성공 우회 방향({_side_name(win_side)}) 우선 적용")
                side = win_side
    preempt_note: str | None = None
    if memory is not None and _near_known_stall(
        memory.stall_spots, pose0["x"], pose0["y"], pose0["yaw_deg"]
    ):
        preempt_note = "preempt_known_stall"
        if verbose:
            print(
                f"           기억된 stall 지점 근접(±{STALL_SPOT_RADIUS_M}m·동일 방향)"
                f" -> 직진 생략, {side * PAD_STALL_DETOUR_DEG:+.0f}° 선제 우회"
            )
    else:
        # 선제 장애물 인지: 전진 직전 발 앞 바닥을 검사해, 정면이 막혔으면(가까이 구조물) 헛돌격
        # 대신 곧장 우회로 전환합니다(get_vision 1프레임 ~30ms, VLM 아님). 좌/우 중 열린 쪽을
        # 우회 side로 채택하되, 검증된 성공 이력(detour_wins)이 있으면 그것을 우선합니다
        # (우선순위: 성공 이력 > 눈으로 본 열린 쪽 > 호출부 표지 부호). 밝기 오검출이 멀쩡한
        # 지점을 영구 회피하게 만들지 않도록 stall_spots에는 남기지 않습니다(detour_wins 학습 유지).
        profile = await _probe_free_space(ctx)
        if profile is not None and not profile["clear"]:
            preempt_note = "preempt_free_space"
            if not had_win_side and profile["freer_side"] != 0.0:
                side = profile["freer_side"]
            if verbose:
                print(
                    f"           선제 인지: 정면 막힘(center={profile['center']:.2f})"
                    f" -> 직진 생략, {side * PAD_STALL_DETOUR_DEG:+.0f}° 우회"
                    f"({_side_name(side)}, 열린 쪽)"
                )
    # 선제 인지가 놓쳐 실제로 장애물에 밀고 들어간 변위(m). 반응형 stall에서만 채워지며,
    # 이 값이 있으면 고정 후진 대신 '전진한 만큼' 폐루프 후퇴합니다.
    rammed_m: float | None = None
    if preempt_note is not None:
        _trace_step(
            memory, action=action, pose=pose0, expected_m=round(expected, 3),
            actual_m=0.0, efficiency=0.0, stall=True, note=preempt_note,
        )
    else:
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=duration_s)
        pose1 = await _get_pose(ctx)
        moved = math.hypot(pose1["x"] - pose0["x"], pose1["y"] - pose0["y"])
        efficiency = _movement_efficiency(expected, moved)
        stalled = _is_stalled(expected, moved)
        _trace_step(
            memory, action=action, pose=pose1, expected_m=round(expected, 3),
            actual_m=round(moved, 3), efficiency=round(efficiency, 2), stall=stalled,
        )
        if not stalled:
            _bump_stat(memory, "path_len_m", moved)
            return True
        _record_stall(memory, pose0)
        rammed_m = moved  # 선제 인지가 놓쳐 실제로 밀고 들어간 변위 → 그만큼 되돌림.
        if verbose:
            print(
                f"           전진 stall(moved={moved:.2f}m, 기대 {expected:.2f}m,"
                f" 효율 {efficiency:.0%}) -> 전진분 {moved:.2f}m 후퇴"
                f" + {side * PAD_STALL_DETOUR_DEG:+.0f}° 우회 전진 후 재조준"
            )
    # 실제 충돌(rammed_m)은 전진한 만큼 폐루프 후퇴해 충돌 직전 자세로 복귀하고, 선제 우회로
    # 애초에 안 들이받은 경우는 회전 공간만 확보하는 짧은 고정 후진을 유지합니다.
    if rammed_m is not None:
        backed = await _reverse_by(ctx, rammed_m)
        _trace_step(
            memory, action="backoff", pose=await _get_pose(ctx),
            expected_m=round(rammed_m, 3), actual_m=round(backed, 3),
            note="collision_reverse",
        )
    else:
        await move_velocity(ctx, vx=-STALL_REVERSE_VX, duration_s=PAD_STALL_BACKUP_S)
    await _turn_by_deg(ctx, side * PAD_STALL_DETOUR_DEG)
    d0 = await _get_pose(ctx)
    await move_velocity(ctx, vx=FORWARD_VX, duration_s=duration_s)
    d1 = await _get_pose(ctx)
    await _turn_by_deg(ctx, -side * PAD_STALL_DETOUR_DEG)
    detour_moved = math.hypot(d1["x"] - d0["x"], d1["y"] - d0["y"])
    detour_stalled = _is_stalled(expected, detour_moved)
    _trace_step(
        memory, action="detour_advance", pose=d1, expected_m=round(expected, 3),
        actual_m=round(detour_moved, 3),
        efficiency=round(_movement_efficiency(expected, detour_moved), 2),
        stall=detour_stalled, note=_side_name(side),
    )
    if detour_stalled:
        _record_stall(memory, d0)
    else:
        _bump_stat(memory, "path_len_m", detour_moved)
        if memory is not None:
            # 성공한 우회의 (진입 지점·진입 방향·side)를 기억 — 재방문 시 이 방향을 우선.
            memory.detour_wins.append(
                {"x": pose0["x"], "y": pose0["y"], "yaw_deg": pose0["yaw_deg"], "side": side}
            )
            del memory.detour_wins[:-DETOUR_WIN_KEEP]
    if verbose:
        print(
            f"           우회 전진 moved={detour_moved:.2f}m ->"
            f" {'OK' if not detour_stalled else '우회도 stall'}"
        )
    return not detour_stalled


def _bypass_chunks(
    streak: int,
    *,
    trigger: int = PAD_BYPASS_STALL_TRIGGER,
    cap: int = PAD_BYPASS_MAX_CHUNKS,
) -> int:
    """연속 hard-stall streak에서 측면 우회로 '따라 이동'할 청크 수를 정합니다.

    trigger 미만이면 0(아직 측면 우회 안 함, 짧은 detour로 계속 시도). trigger 이상이면
    2청크에서 시작해 streak가 커질수록 1씩 늘려 cap까지 escalate합니다 — 같은 구조물에 오래
    막힐수록 더 멀리 따라 이동해 끝/틈을 지나갈 확률을 높입니다.
    """
    if streak < trigger:
        return 0
    return min(streak - trigger + 2, cap)


async def _lateral_bypass(
    ctx: Any,
    side: float,
    chunks: int,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> bool:
    """선형 구조물(벨트 등)에 반복해 막히면 목표 쪽으로 ~90° 꺾어 여러 청크를 따라 이동합니다.

    짧은 detour(1청크)로도 못 뚫는 긴 장애물을 지나기 위한 표준 bug-following입니다. 표지
    재조준을 잠시 멈추고 구조물과 대략 평행하게 이동한 뒤 원래 방위로 복귀 -> 다음 look이
    새 위치에서 재조준합니다. side는 목표(표지)가 있던 방향(+1=좌). 측면도 막히면(전진 stall)
    조기 종료하고 False를 반환해 호출부가 반대쪽을 시도하게 합니다. 카메라·odometry만 사용.
    """
    if verbose:
        print(
            f"           반복 stall -> 측면 우회: {side * PAD_BYPASS_TURN_DEG:+.0f}° 후"
            f" 최대 {chunks}청크 따라 이동"
        )
    entry_pose = await _get_pose(ctx)  # 진입 지점·방위 — 성공 시 detour_wins 기록용.
    await _turn_by_deg(ctx, side * PAD_BYPASS_TURN_DEG)
    expected = FORWARD_EFF_SPEED_MPS * PAD_ADVANCE_DUR
    moved_any = False
    for _ in range(chunks):
        p0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=PAD_ADVANCE_DUR)
        p1 = await _get_pose(ctx)
        moved = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        _trace_step(
            memory, action="bypass_advance", pose=p1, expected_m=round(expected, 3),
            actual_m=round(moved, 3), stall=_is_stalled(expected, moved),
        )
        if _is_stalled(expected, moved):
            break  # 측면도 막힘 -> 중단(호출부가 반대쪽 시도).
        _bump_stat(memory, "path_len_m", moved)
        moved_any = True
    await _turn_by_deg(ctx, -side * PAD_BYPASS_TURN_DEG)  # 원래 방위로 복귀 후 재조준.
    if moved_any and memory is not None:
        # 성공한 측면 우회의 (진입 지점·진입 방위·side)도 detour_wins에 기록합니다 — 지난
        # 런에서 검증된 북쪽 통과(+80° 2회 진전)가 기록되지 않아, 엉뚱한 방향의 옆걸음 detour
        # 성공이 방향 선택을 오염시켜 포켓에 갇혔습니다(라이브 확정). 최근 기록 우선 검색이라
        # 진짜 통과 경험이 낡은/우연한 기록을 자연히 이깁니다.
        memory.detour_wins.append(
            {
                "x": entry_pose["x"],
                "y": entry_pose["y"],
                "yaw_deg": entry_pose["yaw_deg"],
                "side": side,
            }
        )
        del memory.detour_wins[:-DETOUR_WIN_KEEP]
    if verbose:
        print(f"           측면 우회 {'진전' if moved_any else '실패(측면도 막힘)'}")
    return moved_any


# ---------------------------------------------------------------------------
# place-probe 순수 헬퍼 (첫 배달 갭 해소 — ctx 없음, pytest로 잠급니다)
# ---------------------------------------------------------------------------
def _first_float_in(text: str) -> float | None:
    """문자열에서 첫 숫자 토큰(정수/소수)을 float으로 뽑습니다(re 미사용). 없으면 None."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j, seen_dot = i, False
            while j < n and (text[j].isdigit() or (text[j] == "." and not seen_dot)):
                if text[j] == ".":
                    seen_dot = True
                j += 1
            try:
                return float(text[i:j])
            except ValueError:
                return None
        i += 1
    return None


def _pad_letter_after(text: str, start: int) -> str | None:
    """'pad' 바로 뒤(start)부터 pad 문자(A~E)를 추출합니다. 오검출 시 None.

    'pad' 직후가 알파벳이면 더 긴 단어('padding' 등)이므로 기각합니다. 구분자(공백/언더스코어/
    구두점)가 하나라도 있으면 건너뛴 뒤 첫 알파벳을 취하고, 그것이 A~E일 때만 채택합니다.
    """
    n = len(text)
    if start >= n or text[start].isalpha():
        return None
    i = start
    while i < n and not text[i].isalpha():
        i += 1
    if i >= n:
        return None
    ch = text[i].upper()
    return ch if ch in "ABCDE" else None


def _parse_place_error(text: str | None) -> dict[str, Any] | None:
    """place 실패 error.message에서 {"radius_m": float|None, "pad": "A".."E"|None}를 추출합니다(순수).

    ⚠ 형식은 정적 미검증 런타임 계약(잠정 구현 — M0 라이브 캡처 후 교체). 'must be within 1.20m of
    pad_X' 류 문자열은 레포 소스에 0건이라, 라이브에서 원문을 캡처해 이 스캔을 실측 고정해야 합니다.
    파서는 place-probe의 '부가 신호'(무료 pad 국소화 + 반경 로그)이지 필수 의존이 아닙니다 — 형식이
    어긋나거나 입력이 None이면 예외 없이 None을 돌려주어(degrade) 호출부가 동결 방위 재조준 + lateral
    만으로 진행하게 합니다. re 미사용(코드베이스 관례)으로 'within' 뒤 첫 숫자를 반경으로, 'pad' 뒤 첫
    알파벳을 pad 문자로 스캔합니다. 둘 다 못 찾으면 None.
    """
    if not isinstance(text, str):
        return None
    low = text.lower()
    radius: float | None = None
    kw = low.find("within")
    if kw >= 0:
        radius = _first_float_in(low[kw + len("within"):])
    # 'pad' 출현마다 뒤 문자를 검사해 첫 유효 A~E를 채택('padding' 오검출은 건너뛰고 계속).
    pad: str | None = None
    search_from = 0
    while True:
        pk = low.find("pad", search_from)
        if pk < 0:
            break
        cand = _pad_letter_after(text, pk + len("pad"))
        if cand is not None:
            pad = cand
            break
        search_from = pk + len("pad")
    if radius is None and pad is None:
        return None
    return {"radius_m": radius, "pad": pad}


def _radius_from(pose: dict[str, float], goal: dict[str, float]) -> float:
    """pose와 goal(동결 좌표) 사이 유클리드 거리(m) — place 성공 시점 실측 반경 기록용(순수).

    방향 벡터(dx,dy)를 저장하지 않고 '방향 불변 스칼라'만 반환합니다: 표지-팔레트 오프셋은 랙 face
    방향에 종속이라 world dx/dy를 다른 pad에 이식하면 오배송이 됩니다(§8-5). place_radius는 기록·
    로그 전용이므로 스칼라 반경으로 충분합니다.
    """
    return math.hypot(float(pose["x"]) - float(goal["x"]), float(pose["y"]) - float(goal["y"]))


def _arrive_goal_dist(
    memory: AgentMemory | None, letter: str | None, held_color: str, pose: dict[str, float]
) -> float | None:
    """arrive 계측용 목표 잔여 거리(m). 기준: sign_goals[letter] → pad_memory[held_color].anchor → None.

    **None 가드 내장(순수)**: 기준 좌표가 둘 다 없으면 None을 반환하므로 호출부는 결과를 round로 다시
    감싸지 않습니다 — round(None, 2)는 TypeError로 라이브 크래시하고 pytest가 못 잡습니다(§5.8-1).
    도착 순간의 실측 잔여를 place_probe_enter의 d와 같은 route_trace에 순차로 남겨 1.47m 갭이 어느
    구간에서 좁혀졌는지 분해합니다.
    """
    goal = memory.sign_goals.get(letter) if (memory is not None and letter is not None) else None
    if goal is None and memory is not None:
        goal = (memory.pad_memory.get(held_color) or {}).get("anchor")
    if not goal or "x" not in goal:
        return None
    return round(math.hypot(float(pose["x"]) - float(goal["x"]), float(pose["y"]) - float(goal["y"])), 2)


def _arrive_risk(
    area: float, area_thresh: float, d_to_goal_m: float | None, near_m: float
) -> str | None:
    """면적 단독 도착 판정의 위험 케이스를 분류합니다(M2 관측 전용 — 행동 불변). 순수 함수.

    같은 blob이라도 접근 면에 따라 면적이 8배 요동합니다(북면 d=1.5m 약 6k vs 서면 d=1.47m 약 49k,
    라이브 실측). 그래서 면적 임계(area_thresh)만으로는 place 반경(near_m) 안팎을 못 가릅니다:
      - area_far: 면적은 찼는데 동결 목표 잔여 d가 반경 밖(서면 49k@1.47m류 — 도착 오선언 위험).
      - area_close_miss: 반경 안인데 면적 미달(북면 6k류 — 도착 미선언 위험).
    d_to_goal_m가 None(동결 전)이면 잔여를 모르므로 판정 불능(None). 잔여거리 AND '행동' 배선은
    M4이며, M2는 이 위험을 trace로 계량만 합니다(in_range 식 불변 → grind 리스크 0).
    """
    if d_to_goal_m is None:
        return None
    if area >= area_thresh and d_to_goal_m > near_m:
        return "area_far"
    if area < area_thresh and d_to_goal_m <= near_m:
        return "area_close_miss"
    return None


def _south_bias_side(yaw_deg: float) -> float:
    """로봇 좌(+1)/우(−1) 중 world −y(남쪽, bearing −90°)에 가까운 쪽을 반환합니다(순수, M4).

    orbit 순회 방향의 tie-break 편향입니다 — 좌표 하드코딩이 아니라 '동률이면 남쪽 선호'라는 상대
    편향이라 setup 랜덤(1~50)에도 유효합니다(pad_B 남/동 통로 개방 가설). 좌측은 world 방위 yaw+90°,
    우측은 yaw−90°를 향하며, 남(−90°)에 각도상 더 가까운 쪽을 고릅니다(동률이면 좌 +1). free-space
    실측(sector map·free_space_profile)이 있으면 호출부가 그걸 우선하고 이 함수는 tie에서만 씁니다.
    """
    south = -90.0
    left_dist = abs(_angle_diff_deg(yaw_deg + 90.0, south))
    right_dist = abs(_angle_diff_deg(yaw_deg - 90.0, south))
    return 1.0 if left_dist <= right_dist else -1.0


async def _nudge_toward(ctx: Any, goal: dict[str, float], *, dur: float = PLACE_PROBE_STEP_S) -> None:
    """동결 goal로 1스텝 미세 접근: _face_turn_to 재조준(±PAD_CLOSE_MAX_TURN_DEG 캡) 후 dur초 전진.

    이동 명령만 냅니다 — 실이동량 측정·stall 판정은 호출부(_place_probe)가 전후 _get_pose로
    수행합니다(관심사 분리: nudge는 '명령', 판정은 '측정'). dur은 PLACE_PROBE_STEP_S(0.8s)가
    기본 — ADVANCE_MIN_S(0.7) 위라야 램프업을 넘겨 실제로 걷습니다(0.5s는 병진 0).
    """
    pose = await _get_pose(ctx)
    _, turn = _face_turn_to(pose, goal)
    if abs(turn) > PAD_FACE_TOL_DEG:
        await _turn_by_deg(ctx, max(-PAD_CLOSE_MAX_TURN_DEG, min(PAD_CLOSE_MAX_TURN_DEG, turn)))
    await move_velocity(ctx, vx=FORWARD_VX, duration_s=dur)


async def _place_probe(
    ctx: Any,
    letter: str,
    held_color: str,
    goal: dict[str, float],
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """place 반복 + delivered 게이트 + lateral 탈출 1회로 첫 배달을 성사시키고 성공 반경을 실측합니다.

    성공 판정은 오직 get_delivered_count() 증가입니다(status 불신 — pad 부재 시 status=done인데
    큐브가 바닥에 떨어져 delivered 불변인 라이브 2회). 종료 우선순위(순서대로 평가):
      1) delivered 증가 → 즉시 성공 반환(어느 시점이든 최우선, 유일한 성공 판정).
      2) place 루프는 PLACE_PROBE_MAX_TRIES(4)회 또는 PLACE_PROBE_MAX_WALL_S(8s)로 끝나도 종료가
         아니라 lateral 전환점(캡=탈출구 진입 신호).
      3) lateral 탈출은 place 루프 캡 밖에서 별도 1회 보장(LATERAL_MAX_WALL_S 분리 계상).
      4) lateral까지 실패 → positioned=False 반환 → 다음 cycle LLM 재결정(자연 복구).
    api_key 인자를 두지 않습니다 — 전 경로 VLM 0회 계약을 서명 수준에서 강제합니다(내부 호출 대상인
    _close_to_goal/place_nearest_zone/_lateral_bypass/get_delivered_count 모두 api_key 미수신).
    memory.held_color/stage를 수동 변이하지 않습니다 — cycle 말미 update_memory가 held ground truth로
    자동 복귀시킵니다(수동 변이는 중복·충돌, §8-9).
    """
    enter_pose = await _get_pose(ctx)
    d_enter = _radius_from(enter_pose, goal)
    _trace_step(memory, action="place_probe_enter", d_to_goal_m=round(d_enter, 2), pose=enter_pose)
    if verbose:
        print(f"    [place-probe] 진입: 동결 목표 잔여 d={d_enter:.2f}m")

    nearest_pad: str | None = None
    t_loop = time.monotonic()
    for i in range(PLACE_PROBE_MAX_TRIES):
        d0 = await get_delivered_count(ctx)
        result = await place_nearest_zone(ctx)          # VLM 0회. 라이브 즉시(~1ms) 전제.
        d1 = await get_delivered_count(ctx)
        summary = result_summary(result)
        if d1 > d0:
            p = await _get_pose(ctx)
            radius = _radius_from(p, goal)
            if memory is not None:
                memory.place_radius[letter] = round(radius, 3)  # 기록 전용(§8-5).
            if verbose:
                print(
                    f"    [place-probe] 성공: delivered {d0}->{d1}, 실측 반경={radius:.2f}m"
                    f" -> place_radius[{letter}] 기록"
                )
            return {"delivered": True, "positioned": True, "radius_m": radius,
                    "dropped": False, "nearest_pad": nearest_pad}
        # 에러 파싱은 부가 신호(무료 국소화): None이면 국소화 없이 진행(파서는 필수 의존 아님).
        parsed = _parse_place_error(summary.get("error"))
        if parsed and parsed.get("pad"):
            nearest_pad = parsed["pad"]
        status = (summary.get("status") or "").lower()
        if "done" in status:
            # done-but-dropped: status는 완료인데 delivered 불변 = 큐브가 바닥에 떨어짐. lateral로
            # 가지 않고 즉시 반환 — 큐브가 발밑에 있어 다음 cycle pick_cube 재시도가 회수합니다.
            _trace_step(memory, action="place_dropped", pose=await _get_pose(ctx), status=status)
            if verbose:
                print(f"    [place-probe] done-but-dropped(status={status}, delivered 불변) -> 회수 필요")
            return {"delivered": False, "positioned": False, "radius_m": None,
                    "dropped": True, "nearest_pad": nearest_pad}
        # 재조준+짧은 전진(nudge) 후 실이동량 측정(병진≈0 = 팔레트 접촉 stall).
        p_before = await _get_pose(ctx)
        await _nudge_toward(ctx, goal, dur=PLACE_PROBE_STEP_S)
        p_after = await _get_pose(ctx)
        moved = math.hypot(p_after["x"] - p_before["x"], p_after["y"] - p_before["y"])
        _trace_step(memory, action="place_nudge", pose=p_after, actual_m=round(moved, 3), delivered=d1)
        if verbose:
            pad_txt = f" pad={nearest_pad}" if nearest_pad else ""
            print(f"    [place-probe {i}] delivered={d1} moved={moved:.3f}m{pad_txt}")
        if _is_stalled(FORWARD_EFF_SPEED_MPS * PLACE_PROBE_STEP_S, moved):
            if verbose:
                print("    [place-probe] nudge 병진≈0(팔레트 접촉) -> place 루프 조기 종료, lateral로")
            break
        if time.monotonic() - t_loop > PLACE_PROBE_MAX_WALL_S:
            if verbose:
                print(f"    [place-probe] place 루프 {PLACE_PROBE_MAX_WALL_S}s 캡 -> lateral로")
            break

    # --- lateral 탈출 1회 보장(place 루프 캡 밖) — 같은 접근각으로는 결정적 실패라 위치를 옆으로 옮김 ---
    if verbose:
        print(f"    [place-probe] place 루프 종료(캡/{PLACE_PROBE_MAX_TRIES}회) -> lateral 탈출 발동")
    # side 선택(고정 규칙): free_space_profile.freer_side 우선 → 0이면 목표 방위 부호.
    profile = await _probe_free_space(ctx)
    rack_map = await _rack_map_from_frame(ctx)   # M2(활성 시): lateral side에 sector map freer_side 1순위 승격.
    pose = await _get_pose(ctx)
    _, goal_turn = _face_turn_to(pose, goal)
    if rack_map is not None and rack_map.get("freer_side", 0.0) != 0.0:
        side = rack_map["freer_side"]
    elif profile is not None and profile.get("freer_side", 0.0) != 0.0:
        side = profile["freer_side"]
    else:
        side = 1.0 if goal_turn > 0 else -1.0
    # §7.9 계약: side=목표(표지)가 있던 방향 부호(+1=좌). 수직화(~80°)는 _lateral_bypass가 내부에서
    # PAD_BYPASS_TURN_DEG로 자동 수행하므로 perp를 미리 계산해 넣지 않습니다(이중 회전, §8-6). chunks=2는
    # 위치 필수 인자. 함수는 평행 이동 후 원방위 복귀 — 접근각 변경은 이후 _close_to_goal 재조준이 만듭니다.
    # LATERAL_MAX_WALL_S는 lateral 국면의 설계 예산: 하위 호출(_lateral_bypass 2청크·_close_to_goal 3청크·
    # place 1회)이 전부 자기 유계라 구조적으로 이 안이며, place 루프 8s와 '분리 계상'됨을 로그로 확인합니다.
    t_lat = time.monotonic()
    ok = await _lateral_bypass(ctx, side, 2, memory=memory, verbose=verbose)
    lat_elapsed = time.monotonic() - t_lat
    # LATERAL_MAX_WALL_S는 lateral 국면의 '분리 계상 예산'(설계 목표)일 뿐, 재조준을 막는 런타임 게이트가
    # 아닙니다. place 루프 8s와 따로 계상해(§8-10) place가 8s를 다 써도 lateral이 반드시 1회 실행되게 하는
    # 개념적 분리이며, 하위 단계(_lateral_bypass·_close_to_goal·place 1회)가 전부 자기유계라 구조적으로
    # 유한합니다. bypass 성공 후 재조준(_close_to_goal)은 '명시 필수 단계'라 예산 초과로 건너뛰면 안 됩니다
    # — bypass 자체가 80° 왕복 회전 2회+2청크로 ~6-8s라 5s를 넘습니다(Codex 재검토 New Finding: 시간
    # 게이트로 걸면 정상 bypass에서 재place가 영영 실행 안 됨). 실측 lat_elapsed는 관측 로그로만 씁니다.
    if ok:
        # 재조준(명시 단계 — 생략 금지): 새 위치에서 동결 goal을 다시 조준해 접근각을 바꿉니다(§5.6 단계 2).
        await _close_to_goal(ctx, goal, memory=memory, verbose=verbose)
        d0 = await get_delivered_count(ctx)
        await place_nearest_zone(ctx)
        d1 = await get_delivered_count(ctx)
        if verbose:
            over = "" if lat_elapsed <= LATERAL_MAX_WALL_S else f" (예산 {LATERAL_MAX_WALL_S}s 초과, 관측용)"
            print(f"    [place-probe] lateral 탈출 {lat_elapsed:.1f}s{over} -> 재조준 후 재place")
        if d1 > d0:
            p = await _get_pose(ctx)
            radius = _radius_from(p, goal)
            if memory is not None:
                memory.place_radius[letter] = round(radius, 3)
            if verbose:
                print(f"    [place-probe] lateral 후 성공: delivered {d0}->{d1}, 실측 반경={radius:.2f}m")
            return {"delivered": True, "positioned": True, "radius_m": radius,
                    "dropped": False, "nearest_pad": nearest_pad}
    elif verbose:
        print("    [place-probe] lateral 측면도 막힘 -> 재place 생략")
    return {"delivered": False, "positioned": False, "radius_m": None,
            "dropped": False, "nearest_pad": nearest_pad}


async def _read_signs_vlm(
    ctx: Any,
    held_color: str | None,
    api_key: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
    label: str = "",
) -> str:
    """현재 프레임의 signage VLM 판독 1회(+provider fallback 재시도). 원문 텍스트 반환.

    Tokamak fallback 문장(상위 모델 미가용)·빈 응답은 '표지 없음'이 아니라 provider 플랩이므로
    같은 프레임을 짧은 backoff 후 재시도합니다(라이브 실측: 2회 중 1회 fallback → '미검출'로
    오인돼 acquisition 붕괴). 끝까지 fallback이면 ""를 반환합니다. _scan_pad_bearing(항법)과
    survey_pads(서베이)가 공용합니다.
    """
    raw = ""
    for attempt in range(1, VLM_MAX_RETRIES + 1):
        _bump_stat(memory, "vlm_calls", 1)  # 재시도도 실제 VLM 비용이라 매 시도를 셉니다.
        _t_vlm = time.perf_counter()  # M0-3: look 실호출 누적 초(관측 전용; VLM은 pad-nav 지배 비용).
        try:
            raw = await ask_vlm_about_frame(
                ctx,
                build_signage_vlm_prompt(held_color),
                api_key=api_key,
                max_width=SIGNAGE_VLM_MAX_WIDTH,
                quality=SIGNAGE_VLM_QUALITY,
            )
        except Exception:
            raw = ""
        finally:
            # 예외(타임아웃·플랩)도 실제 소비한 벽시계라 finally에서 누적 — 최악 look(191s급)을 놓치지 않음.
            _bump_stat(memory, "vlm_wall_s", time.perf_counter() - _t_vlm)
        if raw and not _looks_like_vlm_fallback(raw):
            return raw
        if verbose:
            print(f"    {label}VLM fallback/빈응답, 재시도 {attempt}/{VLM_MAX_RETRIES}")
        if attempt < VLM_MAX_RETRIES:
            await asyncio.sleep(0.6)
    return ""


async def _scan_pad_bearing(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
    near: bool = False,
) -> dict[str, Any] | None:
    """head를 여러 각도로 팬하며 VLM으로 목표 sign을 찾습니다. 못 찾으면 None.

    반환: {"face_turn": 도(양수=좌회전), "confidence": float, "position": str} —
    face_turn 외 필드는 last_seen 기록용입니다. VLM 실호출마다 route_stats.vlm_calls를
    셉니다(경로 점수의 지배 비용).

    head 팬(set_head)은 locomotion이 없어 로봇이 움직이지 않으므로 스캔 중 위치를 잃지 않습니다
    (아크로 돌며 스캔하면 전진 병진으로 배회함). VLM이 느리므로(7~20s) center부터 보고 처음
    확신 검출에서 조기 종료합니다(1~3회). body-bearing = image_offset + head_yaw(코드의
    full_bearing 규약: +=우측), face_turn = -body_bearing(우측이면 우회전=음수 delta).
    """
    for hy in HEAD_SCAN_YAWS_RAD:
        await set_head(ctx, yaw=hy, pitch=HEAD_PITCH_TRACK)
        raw = await _read_signs_vlm(
            ctx, held_color, api_key, memory=memory, verbose=verbose,
            label=f"scan head={hy:+.1f}rad -> ",
        )
        target = _find_target_sign(_parse_signs(raw), letter)
        conf = _as_confidence(target.get("confidence", 0)) if target else 0.0
        if verbose:
            pos_txt = target.get("position") if target else "?"
            print(f"    scan head={hy:+.1f}rad -> '{letter}':{pos_txt} conf={conf:.2f}")
        if target and conf >= VLM_MIN_CONFIDENCE:
            # bbox 비례 환산(가능하면) 또는 left/right 양자화(fallback)로 오프셋을 얻습니다.
            # set_head(+yaw)는 카메라를 '왼쪽'으로 팬합니다(라이브 실측 2026-07-04: head 0→+0.7에서
            # 중앙 기둥의 image angle -11.5°→-2.5°로 오른쪽 이동 = 시선이 왼쪽으로 감). image angle은
            # +=오른쪽 규약이라, 왼쪽 팬으로 중앙에 온 표지는 몸통 기준 '왼쪽'(음수) 방위입니다. 따라서
            # head_yaw 항은 빼야 합니다. 옛 +부호는 off-center에서 찾은 표지의 좌/우를 뒤집어 로봇을
            # 반대쪽으로 보냈고(head=0 단일각일 땐 항이 0이라 잠복), multi-yaw 스캔 도입으로 드러났습니다
            # — 라이브 확정: 앵커가 pad_C(y=+1.6) 반대편(y=-2.5)에 찍혀 ~3.9m 빗나가 배송 실패.
            body_bearing = _sign_offset_deg(target) - math.degrees(hy)
            await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)
            return {
                "face_turn": -body_bearing,
                "confidence": conf,
                "position": str(target.get("position", "")),
                # bbox 면적 비율(∝ 1/d²): R6 수렴 판정의 2차(희소) 신호. 결측이면 None.
                "bbox_area_frac": _sign_bbox_area_frac(target),
            }
    # M2였던 근접 재확인: 3-yaw 스캔 전패 + 근접(near, d<1m)이면 위쪽(HEAD_PITCH_SIGN_RECHECK)으로 1회
    # 더 판독하려던 로직. 세션5 정찰이 프레임아웃을 반증(핸드오프 §3): d≈0.85m 사인은 완전 프레임 내이고
    # 근접 상실은 플랩이라, 위로 피치하면 눈높이 이하 사인이 오히려 하단 이탈 → net-loss. 근접 상실은
    # VLM abandon + 동결 anchor goal-seek가 방어합니다. PAD_SIGN_NEAR_RECHECK=False로 비활성(가역).
    if near and PAD_SIGN_NEAR_RECHECK:
        await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_SIGN_RECHECK)
        raw = await _read_signs_vlm(
            ctx, held_color, api_key, memory=memory, verbose=verbose,
            label="near recheck(pitch up) -> ",
        )
        target = _find_target_sign(_parse_signs(raw), letter)
        conf = _as_confidence(target.get("confidence", 0)) if target else 0.0
        found = bool(target and conf >= VLM_MIN_CONFIDENCE)
        _trace_step(memory, action="sign_recheck", found=found, pitch=HEAD_PITCH_SIGN_RECHECK)
        if found:
            body_bearing = _sign_offset_deg(target)  # head yaw=0 → head_yaw 항 0.
            await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)
            return {
                "face_turn": -body_bearing,
                "confidence": conf,
                "position": str(target.get("position", "")),
                "bbox_area_frac": _sign_bbox_area_frac(target),
            }
    await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)
    return None


async def _look_for_sign(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    memory: AgentMemory | None,
    entry: dict[str, Any] | None,
    *,
    reason: str,
    verbose: bool = False,
    near: bool = False,
) -> float | None:
    """VLM look 1회(호출 사유 로그 필수). 성공 시 last_seen을 갱신하고 face_turn을 반환합니다.

    VLM은 pad-nav의 지배 비용(6~32s/회)이라, 호출부는 cached route/last_seen이 모두 소진된
    경우에만 이 함수에 도달해야 합니다. 왜 호출했는지가 로그에 남아야 이후 절감 튜닝이 가능합니다.
    """
    if verbose:
        print(f"    VLM 호출 이유: {reason}")
    sighting = await _scan_pad_bearing(
        ctx, letter, held_color, api_key, memory=memory, verbose=verbose, near=near,
    )
    pose = await _get_pose(ctx)
    if sighting is None:
        _trace_step(memory, action="look", source="vlm", found=False, note=reason, pose=pose)
        return None
    est_d: float | None = None
    if entry is not None:
        # head 팬만 했으므로 scan 동안 body pose는 그대로 — 지금 pose가 목격 pose입니다.
        entry["last_seen"] = _make_last_seen(
            pose,
            sighting["face_turn"],
            confidence=sighting.get("confidence", 0.0),
            position=sighting.get("position", ""),
        )
        conf = float(sighting.get("confidence", 0.0) or 0.0)
        # 관측 ray(목격 지점 + world 방위)를 letter 지도에 축적하고, 기선·교각이 유효한 쌍이
        # 생기면 삼각측량 교점으로 목표를 '동결'합니다. 동결된 목표는 bbox 거리 노이즈가 끌고
        # 다닐 수 없는 고정 좌표라(라이브 확정: 원거리 오추정 1회가 anchor를 반대 y-half로
        # 끌고 가 전도), anchor 슬롯에 frozen 마크와 함께 설치해 기존 goal-seek가 그대로 씁니다.
        if memory is not None and conf >= VLM_MIN_CONFIDENCE:
            _add_sign_ray(memory.sign_rays, letter, {
                "x": pose["x"], "y": pose["y"],
                "bearing_deg": _angle_diff_deg(
                    float(pose.get("yaw_deg", 0.0)) + float(sighting["face_turn"]), 0.0
                ),
                "conf": conf,
            })
            # Option A 봉인: 동결 커밋은 8런 동안 0회 성사됐고, 데모 중 '첫 발화'가 곧 미검증 경로
            # 진입이므로 스위치로 잠급니다. ray 축적(위)은 진단용으로 유지합니다.
            goal = None
            if SIGN_FREEZE_ENABLED:
                st_fz: dict[str, Any] = {}
                goal = _maybe_freeze_sign_goal(
                    memory.sign_rays, memory.sign_goals, letter,
                    cooldown=memory.sign_refreeze_block, state=st_fz,
                )
                _trace_freeze(memory, letter, st_fz)  # M1: freeze_commit/hold/pending/reject/cooldown 관측.
            if goal is not None and not (entry.get("anchor") or {}).get("frozen"):
                entry["anchor"] = {
                    "x": goal["x"], "y": goal["y"],
                    "w_sum": PAD_ANCHOR_W_CAP, "n": 2, "frozen": True,
                }
                if verbose:
                    print(
                        f"    목표 동결(삼각측량): '{letter}' -> ({goal['x']:+.2f},{goal['y']:+.2f})"
                        f" (ray {len(memory.sign_rays.get(letter) or [])}개)"
                    )
        # bbox 면적이 있으면 거리까지 추정해 sign의 world '점'(anchor)을 융합 갱신합니다.
        # 융합 게이트는 nav 게이트보다 엄격(0.6): 낮은 확신의 오독이 점 추정을 오염시키면
        # ray보다 수명이 길어(회전 불변) 피해가 크기 때문입니다. 동결(frozen) 후에는 융합을
        # 건너뜁니다 — 삼각측량 좌표가 단안 거리 추정보다 항상 우월하기 때문입니다.
        est_d = _estimate_sign_distance(sighting.get("bbox_area_frac"))
        if (
            est_d is not None
            and conf >= PAD_ANCHOR_MIN_CONF
            and not (entry.get("anchor") or {}).get("frozen")
        ):
            point = _project_point(pose, sighting["face_turn"], est_d)
            entry["anchor"] = _fuse_anchor(
                entry.get("anchor"), point, _anchor_weight(conf, est_d)
            )
            if verbose:
                a = entry["anchor"]
                print(
                    f"    anchor 갱신: d≈{est_d:.1f}m -> ({a['x']:+.2f},{a['y']:+.2f})"
                    f" (목격 {a['n']}회, w={a['w_sum']:.2f})"
                )
    _trace_step(
        memory, action="look", source="vlm", found=True,
        face_turn_deg=round(sighting["face_turn"], 1),
        conf=sighting.get("confidence"),  # 관측 전용(게이트 승격 금지, §8-3); sign_rays의 conf와 동일 명명.
        bbox_area_frac=sighting.get("bbox_area_frac"),
        est_dist_m=None if est_d is None else round(est_d, 2),
        note=reason, pose=pose,
    )
    return sighting["face_turn"]


async def survey_pads(
    ctx: Any,
    memory: AgentMemory,
    api_key: str,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """스폰 개활지에서 아크 회전 서베이로 pad 표지 지도를 부트스트랩합니다(런당 1회 권장).

    설계 원칙(사용자 합의): "pad 위치를 결정적으로 알아야 경로가 비결정적이어도 임무가
    수렴한다". 45°×8 아크 회전으로 한 바퀴를 훑되, 매 방위는 공짜인 OpenCV 표지후보 검사
    (_sign_candidates, ~30ms)로 먼저 거르고 후보가 있는 방위에서만 VLM을 호출합니다(상한
    SURVEY_VLM_MAX_CALLS). 표지가 하나도 안 보이는 랜덤 스폰(setup 1~50)이면 VLM 0회·약 20초
    no-op으로 끝나 비용이 자동 조절됩니다. 판독된 각 표지는 (1) letter별 관측 ray로 축적되고
    (이후 이동 중 2차 목격과 삼각측량 → 목표 동결), (2) bbox 거리 추정으로 색상별 anchor를
    시딩해 pick 직후 navpad가 곧장 goal-seek로 출발하게 합니다. 같은 서베이에서 이미 판독한
    world 방위 ±SURVEY_DEDUPE_BEARING_DEG 안의 후보는 재판독을 생략합니다(인접 스텝은 FOV가
    겹치므로). 전부 카메라+자기 odometry 유도 — 좌표/색 하드코딩 없음(Level 2 합법).
    """
    color_of_letter = {v: k for k, v in DESTINATION_SIGN_RULES.items()}
    read_bearings: list[float] = []   # 이번 서베이에서 VLM 판독을 마친 world 방위들.
    vlm_used = 0
    found: dict[str, int] = {}
    t_survey = time.monotonic()  # 서베이 실시간 캡(SURVEY_MAX_WALL_S) 기준 — VLM 플랩 누적 시 조기 종료.
    for step in range(SURVEY_STEPS):
        if time.monotonic() - t_survey > SURVEY_MAX_WALL_S:
            if verbose:
                print(
                    f"  [survey] wall-clock {SURVEY_MAX_WALL_S:.0f}s 캡 -> 조기 종료"
                    f"(step {step}/{SURVEY_STEPS}) — 첫 pick 착수 지연 방지"
                )
            break
        pose = await _get_pose(ctx)
        yaw = float(pose.get("yaw_deg", 0.0))
        frame_dets = await perceive(ctx)
        cands = _sign_candidates(frame_dets)
        # source-seek 우선순위 ①: 같은 프레임에서 clean cube blob의 world 방위를 별도 축적합니다
        # (VLM 0회, _sign_candidates와 별개). perceive는 위에서 1회만 호출 — 서베이 비용 불변.
        for _cd in frame_dets:
            if _is_clean_cube(_cd, CUBE_ARRIVAL_AREA):
                _record_cube_sighting(memory.cube_sightings, _cd, pose)
        # 아직 판독 안 된 방위의 후보만 남깁니다(blob body-bearing ≈ image angle, head=0).
        fresh = [
            c for c in cands
            if all(
                abs(_angle_diff_deg(yaw - float(c.angle_deg), rb)) > SURVEY_DEDUPE_BEARING_DEG
                for rb in read_bearings
            )
        ]
        if verbose:
            print(
                f"  [survey {step}] yaw={yaw:+.0f}° 후보 {len(cands)}개"
                f"(신규 {len(fresh)}개), VLM {vlm_used}/{SURVEY_VLM_MAX_CALLS}"
            )
        if fresh and vlm_used < SURVEY_VLM_MAX_CALLS:
            vlm_used += 1
            raw = await _read_signs_vlm(
                ctx, None, api_key, memory=memory, verbose=verbose,
                label=f"survey {step} -> ",
            )
            for s in _parse_signs(raw):
                conf = _as_confidence(s.get("confidence", 0))
                if conf < VLM_MIN_CONFIDENCE:
                    continue
                letter = str(s["letter"]).upper()
                body = _sign_offset_deg(s)          # += 우측(이미지 규약)
                wb = _angle_diff_deg(yaw - body, 0.0)  # world 방위 = yaw - body_bearing
                read_bearings.append(wb)
                _add_sign_ray(memory.sign_rays, letter, {
                    "x": pose["x"], "y": pose["y"], "bearing_deg": wb, "conf": conf,
                })
                if SIGN_FREEZE_ENABLED:  # Option A 봉인(위 navpad 콜사이트와 동일 근거).
                    st_fz: dict[str, Any] = {}
                    _maybe_freeze_sign_goal(
                        memory.sign_rays, memory.sign_goals, letter,
                        cooldown=memory.sign_refreeze_block, state=st_fz,
                    )
                    _trace_freeze(memory, letter, st_fz)  # M1: 서베이 동결 상태 관측.
                found[letter] = found.get(letter, 0) + 1
                est_d = _estimate_sign_distance(_sign_bbox_area_frac(s))
                color = color_of_letter.get(letter)
                if color is not None:
                    entry = _pad_memory_entry(memory.pad_memory, color)
                    entry["last_seen"] = _make_last_seen(
                        pose, -body, confidence=conf, position=str(s.get("position", "")),
                    )
                    if (
                        est_d is not None
                        and conf >= PAD_ANCHOR_MIN_CONF
                        and not (entry.get("anchor") or {}).get("frozen")
                    ):
                        entry["anchor"] = _fuse_anchor(
                            entry.get("anchor"),
                            _project_point(pose, -body, est_d),
                            _anchor_weight(conf, est_d),
                        )
                if verbose:
                    d_txt = "?" if est_d is None else f"{est_d:.1f}m"
                    print(f"    survey 표지 '{letter}' wb={wb:+.0f}° conf={conf:.2f} d≈{d_txt}")
        if step < SURVEY_STEPS - 1:
            await _turn_by_deg(ctx, SURVEY_STEP_DEG)
    _trace_step(
        memory, action="survey", letters={k: v for k, v in sorted(found.items())},
        vlm_calls=vlm_used, wall_s=round(time.monotonic() - t_survey, 2),  # M0-2: 국면 wall-clock.
        pose=await _get_pose(ctx),
    )
    if verbose:
        goals = {
            L: f"({g['x']:+.2f},{g['y']:+.2f})" for L, g in sorted(memory.sign_goals.items())
        }
        print(
            f"  [survey] 완료: 표지 {sorted(found)} / 동결 {goals or '없음'}"
            f" / VLM 판독 {vlm_used}회(프레임 단위; 제공자 재시도는 판독 내부)"
        )
    return {"letters": found, "vlm_calls": vlm_used, "goals": dict(memory.sign_goals)}


async def _replay_route(
    ctx: Any,
    waypoints: list[dict[str, float]],
    memory: AgentMemory | None,
    *,
    verbose: bool = False,
) -> bool:
    """기억된 성공 경로의 waypoint들을 odometry 폐루프로 따라갑니다(카메라·VLM 불필요).

    현재 위치에서 최근접 waypoint로 합류해 경로 후반만 재사용합니다. waypoint마다 거리/방위
    벡터(_face_turn_to)를 다시 계산해 회전 후 거리 기반 시간(_advance_duration_s)만큼
    전진하는 것을 반복하고, stall 우회까지 실패하거나 waypoint를 청크 예산 안에 못 따라가면
    False를 반환해 상위(VLM 탐색)로 넘깁니다. odometry 오차는 waypoint마다 재계산하는
    폐루프가 흡수하고, 최종 도착 판정은 어차피 상위의 VLM+색블롭 게이트가 책임집니다.
    """
    pose = await _get_pose(ctx)
    start_i = _nearest_waypoint_index(waypoints, pose["x"], pose["y"])
    if verbose:
        print(f"  [replay] waypoint {start_i + 1}/{len(waypoints)}부터 합류  {await _pose_str(ctx)}")
    for wi, wp in enumerate(waypoints[start_i:], start=start_i):
        for _chunk in range(ROUTE_REPLAY_CHUNKS_PER_WP):
            pose = await _get_pose(ctx)
            dist, face_turn = _face_turn_to(pose, wp)
            if dist <= WAYPOINT_TOL_M:
                break
            if verbose:
                print(f"  [replay wp{wi}] dist={dist:.2f}m turn={face_turn:+.0f}°")
            if abs(face_turn) > PAD_FACE_TOL_DEG:
                await _turn_by_deg(ctx, face_turn)
            side = 1.0 if face_turn > 0 else -1.0
            if not await _advance_or_detour(
                ctx, side, duration_s=_advance_duration_s(dist),
                memory=memory, action="replay_advance", verbose=verbose,
            ):
                if verbose:
                    print("  [replay] 우회까지 stall -> replay 중단, VLM 탐색으로 전환")
                return False
        else:
            pose = await _get_pose(ctx)
            dist, _ = _face_turn_to(pose, wp)
            if dist > WAYPOINT_TOL_M * 2:
                if verbose:
                    print(f"  [replay] wp{wi} 미도달(dist={dist:.2f}m) -> replay 중단")
                return False
            # 살짝 어긋난 정도면 다음 waypoint로 계속(폐루프가 흡수).
    return True


async def _orbit_for_face(
    ctx: Any,
    goal: dict[str, float],
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> bool:
    """동결 goal 주위를 구조물과 평행하게 순회해 반경 진입 가능한 '열린 면'을 찾습니다(M4, P3).

    접근 면 문제: pad place 원이 특정 통로(남/동)에서만 열려 북측 정면 접근으로는 반경에 못 듭니다.
    측면 우회(_lateral_bypass)가 cap까지 소진돼도 hard_stall이 재발하면, 구조물을 따라 한 방향으로
    순회하며 매 청크 후 goal 방위가 열렸는지 검사합니다. 열리면 _close_to_goal로 재조준·접근하고 True를
    반환합니다(도착 판정은 navpad 본 루프가 계속). 전부 카메라 free-space + odometry만 — waypoint·좌표 0.

    전도 방지: 회전은 폐루프 _turn_by_deg, 전진은 1.4s 청크(_advance_or_detour)만 — 대회전 아크 신설 금지.
    순회 side는 최초 1회 결정 후 고정(방향 반전 금지 — 왕복 진동 방지). 연속 stall ORBIT_STALL_GIVEUP회
    또는 ORBIT_MAX_CHUNKS 소진 시 포기(False). nav당 1회만 호출됩니다(호출부 게이트).
    ★개방 검사식은 sector map 비활성 상태의 근사(재조준+비stall)이며 M3 실측으로 확정될 잠정 설계입니다
    (§OQ-B). E2E 중 보정.★
    """
    pose0 = await _get_pose(ctx)
    # 순회 side(최초 1회 고정): sector map freer_side → free_space_profile.freer_side → tie면 남측 편향.
    side = 0.0
    rack_map = await _rack_map_from_frame(ctx)
    if rack_map is not None and rack_map.get("freer_side", 0.0) != 0.0:
        side = rack_map["freer_side"]
    else:
        profile = await _probe_free_space(ctx)
        if profile is not None and profile.get("freer_side", 0.0) != 0.0:
            side = profile["freer_side"]
    if side == 0.0:
        side = _south_bias_side(float(pose0.get("yaw_deg", 0.0)))
    _trace_step(
        memory, action="orbit_start", side=_side_name(side), pose=pose0,
        goal_bearing_deg=round(
            math.degrees(math.atan2(goal["y"] - pose0["y"], goal["x"] - pose0["x"])), 1
        ),
    )
    if verbose:
        print(f"  [orbit] 발동: side={_side_name(side)}로 구조물 순회(≤{ORBIT_MAX_CHUNKS}청크)")
    stall_streak = 0
    for _chunk in range(ORBIT_MAX_CHUNKS):
        moved = await _advance_or_detour(ctx, side, memory=memory, action="orbit_step", verbose=verbose)
        p = await _get_pose(ctx)
        _trace_step(memory, action="orbit_step", pose=p, moved=moved, stall=not moved)
        if not moved:
            stall_streak += 1
            if stall_streak >= ORBIT_STALL_GIVEUP:
                _trace_step(memory, action="orbit_giveup", reason="stall")
                if verbose:
                    print(f"  [orbit] 순회 연속 stall {stall_streak}회 -> 포기")
                return False
            continue
        stall_streak = 0
        # 개방 검사(sector map 비활성 근사): goal로 재조준(캡) 후 1청크가 실제로 goal에 가까워지면(단순
        # 이동이 아니라 잔여 d 감소 — _advance_or_detour가 detour로 옆으로 새는 false-open 방지) 그 면이
        # 열렸다고 판정합니다(§OQ-C 근사 강화).
        _, turn = _face_turn_to(p, goal)
        if abs(turn) > PAD_FACE_TOL_DEG:
            await _turn_by_deg(ctx, max(-PAD_CLOSE_MAX_TURN_DEG, min(PAD_CLOSE_MAX_TURN_DEG, turn)))
        d_before = _radius_from(await _get_pose(ctx), goal)
        moved_probe = await _advance_or_detour(ctx, side, memory=memory, action="orbit_probe", verbose=verbose)
        d_after = _radius_from(await _get_pose(ctx), goal)
        if moved_probe and d_after < d_before - ORBIT_OPEN_MIN_GAIN_M:
            _trace_step(memory, action="orbit_open", d_to_goal_m=round(d_after, 2), basis="reaim_closer")
            if verbose:
                print(f"  [orbit] 개방 판정(재조준+접근 {d_before:.2f}→{d_after:.2f}m) -> _close_to_goal")
            await _close_to_goal(ctx, goal, memory=memory, verbose=verbose)
            return True
    _trace_step(memory, action="orbit_giveup", reason="chunks")
    if verbose:
        print(f"  [orbit] {ORBIT_MAX_CHUNKS}청크 소진 -> 포기")
    return False


async def visual_navigate_to_pad(
    ctx: Any,
    held_color: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> bool:
    """navpad 얇은 래퍼: 1콜 누적 wall-clock + VLM 델타를 try/finally로 '모든' 반환 경로(정상 반환·
    조기 guard·예외)에서 1줄 로그로 남깁니다(§5.8-5, Codex MINOR 1). 실제 항법은 _navpad_impl가
    수행하며 (reached, attempts)를 돌려줍니다 — 예외가 나도 finally가 로그를 흘리지 않습니다.
    VLM 델타 출처는 memory.route_stats["vlm_calls"](_bump_stat이 갱신).
    """
    t_nav = time.perf_counter()
    stats0 = memory.route_stats if memory is not None else {}
    vlm_before = stats0.get("vlm_calls", 0)
    vlm_wall_before = stats0.get("vlm_wall_s", 0.0)
    stalls_before = stats0.get("stalls", 0)
    reached, attempts = False, 0
    try:
        reached, attempts = await _navpad_impl(ctx, held_color, memory=memory, verbose=verbose)
        return reached
    finally:
        # M0-2: pad_exit — M3가 orbit·잔여 상수를 도출할 국면 종료 스냅샷. 모든 반환 경로(정상·조기
        # guard·예외)에서 1건 남깁니다(관측 전용, 행동 불변). 종료 pose 읽기가 실패해도(예외 전파 중)
        # None으로 삼켜 로깅이 원 예외를 가리지 않게 합니다. 별도 nav_seq 카운터는 신설하지 않고
        # (선행 스펙 §8-8) 파일 레코드 간 연결은 공통 ts + pad_exit·place_probe_enter 인접으로 합니다.
        exit_pose: dict[str, float] | None = None
        if memory is not None:
            try:
                exit_pose = await _get_pose(ctx)
            except Exception:
                exit_pose = None
        letter = DESTINATION_SIGN_RULES.get(held_color)
        goal = memory.sign_goals.get(letter) if (memory is not None and letter is not None) else None
        goal_bearing = None
        if goal is not None and exit_pose is not None:
            # ② 접근 면 방위(로봇→goal 월드 방위각) — 인라인 계산(신규 헬퍼 금지, M0은 순수 함수 0개).
            goal_bearing = round(
                math.degrees(math.atan2(goal["y"] - exit_pose["y"], goal["x"] - exit_pose["x"])), 1
            )
        stats1 = memory.route_stats if memory is not None else {}
        stall_recent = (
            [{"x": round(s["x"], 2), "y": round(s["y"], 2)} for s in memory.stall_spots[-5:]]
            if memory is not None else []
        )
        _trace_step(
            memory, action="pad_exit", reached=reached, attempts=attempts, pose=exit_pose,
            goal_bearing_deg=goal_bearing,                              # ②
            wall_s=round(time.perf_counter() - t_nav, 2),              # ③ 국면 wall-clock(navpad)
            vlm_calls=stats1.get("vlm_calls", 0) - vlm_before,
            vlm_wall_s=round(stats1.get("vlm_wall_s", 0.0) - vlm_wall_before, 2),
            stalls=stats1.get("stalls", 0) - stalls_before,
            stall_recent=stall_recent,                                 # ④ stall 좌표(최근 ≤5)
        )
        if verbose:
            vlm_now = stats1.get("vlm_calls", 0)
            print(
                f"  [pad] 종료: reached={reached} attempts={attempts}"
                f" 누적 {time.perf_counter() - t_nav:.1f}s VLM {vlm_now - vlm_before}회"
            )


async def _navpad_impl(
    ctx: Any,
    held_color: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> tuple[bool, int]:
    """VLM signage + 경로 기억으로 destination pad를 국소화·접근합니다(Level 2 합법). 반환 (reached,
    attempts) — 종료 로그는 래퍼(visual_navigate_to_pad)가 담당합니다.

    행동 우선순위 — 위가 싸고(0 VLM) 아래가 비쌉니다(VLM 6~32s/회는 지배 비용):
    1) best_route replay: 같은 색을 이미 배송했으면 그 성공 경로의 waypoint를 odometry
       폐루프로 재주행합니다(합류는 최근접 waypoint부터, greedy 재사용). 종점에서 VLM 1회로
       진짜 목표 sign인지 확인만 합니다 — 수 회의 look 루프가 1회로 줄어듭니다.
    2) anchor goal-seek: sign을 bbox와 함께 목격한 적 있으면 그 world '점' 추정치로 매 반복
       '현재 pose'에서 (거리, 회전각)을 새로 계산해 접근합니다(VLM 0회). ray와 달리 회전·
       이동으로 낡지 않아, VLM이 일시적으로 안 보여도(가림·네트워크 플랩) 길을 잃지 않습니다
       — 사람이 한 번 본 목적지의 대략적 위치를 기억해 두고 장애물을 우회하며 접근하는 방식.
    3) last_seen 재조준: anchor가 없을 때(=bbox 없는 목격뿐), 기억한 world 방위(ray)로 회전
       (VLM 0회). 연속 LAST_SEEN_MAX_REUSE회까지만 — ray 오차가 누적되면 VLM으로 재확인.
    4) 목격 지점 복귀: last_seen이 신뢰반경 밖이고 anchor도 없으면 목격 pose 자체로 되돌아가
       재조준합니다(pad가 한 번 보였다 사라져도 무작정 전진하지 않음).
    5) VLM look: 위가 모두 불가할 때만. 호출 사유를 반드시 로그에 남깁니다.
    찾은 뒤에는 폐루프 회전(_turn_by_deg)으로 정면 정렬 후 색블롭 게이트(_target_in_range)로
    근접을 확인하고, 아니면 stall-감지 전진 한 청크 후 재관찰합니다. 기억(anchor/last_seen)만으로
    조향해 도착한 경우에는 place 전에 VLM 1회로 최종 확인합니다(색블롭은 같은 색 source에서도 참).
    """
    letter = DESTINATION_SIGN_RULES.get(held_color)
    if letter is None:
        return False, 0
    try:
        config = load_config(require_tokamak=True)
    except Exception:
        return False, 0
    api_key = config.tokamak_api_key
    entry = _pad_memory_entry(memory.pad_memory, held_color) if memory is not None else None

    # --- 1) 성공 경로 greedy 재사용 ---
    best = entry.get("best_route") if entry is not None else None
    if best and best.get("waypoints"):
        if verbose:
            print(
                f"  [pad] best_route replay: score={best['score']:.1f},"
                f" wp {len(best['waypoints'])}개  {await _pose_str(ctx)}"
            )
        if await _replay_route(ctx, best["waypoints"], memory, verbose=verbose):
            face_turn = await _look_for_sign(
                ctx, letter, held_color, api_key, memory, entry,
                reason="route replay 종점 확인(오배송 방지)", verbose=verbose,
            )
            if face_turn is not None:
                if abs(face_turn) > PAD_FACE_TOL_DEG:
                    await _turn_by_deg(ctx, face_turn)
                if await _target_in_range(ctx, held_color, PAD_ARRIVAL_AREA):
                    close_goal = memory.sign_goals.get(letter) if memory is not None else None
                    if close_goal is not None:
                        await _close_to_goal(ctx, close_goal, memory=memory, verbose=verbose)
                    if verbose:
                        print(f"  [pad] replay 도착 확인  {await _pose_str(ctx)}")
                    ap = await _get_pose(ctx)
                    _trace_step(
                        memory, action="arrive", source="route", pose=ap,
                        d_to_goal_m=_arrive_goal_dist(memory, letter, held_color, ap),
                        goal_bearing_deg=(  # M0-2: 접근 면 방위(로봇→goal 월드 방위각), 동결 전이면 None.
                            round(math.degrees(math.atan2(_g["y"] - ap["y"], _g["x"] - ap["x"])), 1)
                            if (_g := memory.sign_goals.get(letter) if memory is not None else None) is not None
                            else None
                        ),
                        arrive_area=None, area_thresh=PAD_ARRIVAL_AREA,  # replay는 blob 미실행.
                        arrive_risk=None,  # M2: replay는 area 미측정 → 판정 불능.
                    )
                    return True, 0
            if verbose:
                print("  [pad] replay 종점에서 미확인 -> 일반 탐색 루프로 전환")
        elif verbose:
            print("  [pad] replay 실패 -> 일반 탐색 루프로 전환")

    # --- 2~5) anchor goal-seek → last_seen 재조준 → 목격 지점 복귀 → VLM look 계층 루프 ---
    fails = 0
    ls_streak = 0          # last_seen 연속 재사용 횟수(상한 초과 시 VLM 재확인 강제).
    anchor_streak = 0      # anchor 연속 goal-seek 횟수(상한 초과 시 VLM 재확인 강제).
    anchor_near_miss = 0   # anchor 근접 VLM 미검출 횟수(상한 도달 시 anchor 폐기).
    vlm_empty_streak = 0   # M1-4(P5): 탐색 look 연속 빈응답 횟수(PAD_VLM_ABANDON_N 도달 시 look 소각 중단).
    residual_fail_streak = 0  # M4-1: area_ok인데 동결 잔여 초과가 연속된 횟수(FAIL_N 도달 시 goal 폐기 degrade).
    orbit_used = False     # M4-2: free-space orbit는 nav당 1회만(무한 순회 방지).
    ls_return_used = False # 목격 지점 복귀는 nav당 1회만(무한 왕복 방지).
    detour_side = 1.0      # stall 우회 방향(+1=좌회전). target이 보이면 그쪽으로.
    detour_fails: dict[str, int] = {"left": 0, "right": 0}
    approach_history: list[dict[str, Any]] = []  # R6 관측: 반복별 {area, face_turn} 표본.
    hard_stall_streak = 0  # 접근 전진이 detour까지 연속 실패한 횟수(측면 우회 escalate 트리거).
    bypass_rounds = 0      # 이번 nav의 측면 우회 발동 누계 — streak 리셋과 무관하게 escalate.
    staged_once = False    # M2 staging 경유는 nav당 1회만(동시 1개 규약).
    for attempt in range(PAD_OUTER_MAX):
        t0 = time.perf_counter()
        pose = await _get_pose(ctx)
        # M2 sector map(활성 시): 이번 attempt의 랙 지도. 비활성/미보정이면 None → 아래 배선 전부 무영향.
        rack_map = await _rack_map_from_frame(ctx)
        face_turn: float | None = None
        source = "vlm"
        last_seen = entry.get("last_seen") if entry is not None else None
        ls_turn = _last_seen_face_turn(last_seen, pose)
        # 리뷰 반영(Codex Major): 동결 goal이 있는데 frozen anchor가 미설치면(survey_pads 동결은 bbox
        # 거리/conf 조건부로만 anchor를 심음) 여기서 승격합니다. 이게 없으면 VLM 플랩 + vlm_abandon
        # 조합에서 조향 근거가 sign_goals뿐이라 어떤 티어(anchor/last_seen)도 못 쓰고 attempt를
        # 소각합니다 — _look_for_sign의 동결 anchor 설치(:3465대)와 동형으로 goal-seek를 보장합니다.
        if entry is not None and memory is not None:
            _frozen_goal = memory.sign_goals.get(letter)
            if _frozen_goal is not None and not (entry.get("anchor") or {}).get("frozen"):
                entry["anchor"] = {
                    "x": _frozen_goal["x"], "y": _frozen_goal["y"],
                    "w_sum": PAD_ANCHOR_W_CAP, "n": 2, "frozen": True,
                }
        anchor = entry.get("anchor") if entry is not None else None
        anchor_dist: float | None = None
        anchor_turn: float | None = None
        if anchor is not None:
            anchor_dist, anchor_turn = _face_turn_to(pose, anchor)

        if (
            anchor_turn is not None
            and anchor_dist is not None
            and anchor_dist > PAD_ANCHOR_NEAR_M
            and anchor_streak < PAD_ANCHOR_MAX_REUSE
        ):
            # 2) anchor goal-seek: 기억한 world 점으로의 (거리, 회전각)을 '현재 pose'에서
            #    재계산합니다(폐루프) — ray 재조준과 달리 회전이 누적돼도 낡지 않습니다.
            #    재조준각은 ±PAD_ANCHOR_MAX_REAIM_DEG 캡: 큰 후방 회전은 두 반복에 나눠
            #    (회전→전진→재평가) 지난 런의 스핀-전도 패턴을 차단합니다.
            face_turn = max(-PAD_ANCHOR_MAX_REAIM_DEG, min(PAD_ANCHOR_MAX_REAIM_DEG, anchor_turn))
            source = "anchor"
            anchor_streak += 1
            _trace_step(
                memory, action="look", source="anchor", found=True,
                face_turn_deg=round(face_turn, 1), anchor_dist_m=round(anchor_dist, 2), pose=pose,
            )
            if verbose:
                print(
                    f"  [pad {attempt}] anchor 조준 d={anchor_dist:.2f}m turn={face_turn:+.0f}°"
                    f" (VLM 생략 {anchor_streak}/{PAD_ANCHOR_MAX_REUSE})  {await _pose_str(ctx)}"
                )
        elif anchor is None and ls_turn is not None and ls_streak < LAST_SEEN_MAX_REUSE:
            # 3) last_seen 재조준(anchor 없을 때만): 기억한 world 방위와 현재 yaw의 차만큼 회전.
            #    anchor가 있으면 ray는 같은 목격의 열화판이라 항상 생략합니다 — 지난 런에서
            #    '우회 직후 VLM 강제'(anchor 연속 상한 소진)를 이 티어가 가로채 ray 재조준이
            #    stall 지점으로 돌진했습니다(라이브 확정).
            face_turn, source = ls_turn, "last_seen"
            ls_streak += 1
            _trace_step(
                memory, action="look", source="last_seen", found=True,
                face_turn_deg=round(face_turn, 1), pose=pose,
            )
            if verbose:
                print(
                    f"  [pad {attempt}] last_seen 재조준 face_turn={face_turn:+.0f}°"
                    f" (VLM 생략 {ls_streak}/{LAST_SEEN_MAX_REUSE})  {await _pose_str(ctx)}"
                )
        else:
            if anchor is None and last_seen is not None and ls_turn is None and not ls_return_used:
                # 4) 목격 지점 복귀: ray 신뢰반경 밖 -> 마지막으로 보인 pose로 되돌아가 재조준.
                #    anchor가 있으면 생략 — 점 재조준은 어디서든 성립해 복귀 왕복이 낭비입니다.
                ls_return_used = True
                if verbose:
                    print(
                        f"  [pad {attempt}] last_seen 신뢰반경({LAST_SEEN_MAX_DRIFT_M}m) 밖"
                        " -> 목격 지점으로 복귀"
                    )
                if await _replay_route(ctx, [last_seen["pose"]], memory, verbose=verbose):
                    ls_streak = 0
                    continue  # 복귀 성공: 다음 반복에서 last_seen ray를 재사용.
            # 5) VLM look — 왜 여기까지 왔는지 사유를 남깁니다.
            near_look = (
                anchor is not None and anchor_dist is not None and anchor_dist <= PAD_ANCHOR_NEAR_M
            )
            if near_look:
                reason = f"anchor 근접(d={anchor_dist:.2f}m) -> VLM 확인"
            elif anchor is not None:
                reason = "anchor 재확인(연속 상한 소진 또는 우회 직후)"
            elif last_seen is None:
                reason = "last_seen 없음(첫 탐색)"
            elif ls_turn is None:
                reason = "last_seen 신뢰반경 밖(복귀 시도 후)"
            else:
                reason = f"last_seen 연속 {LAST_SEEN_MAX_REUSE}회 사용 -> VLM 재확인"
            # M1-4(P5): 탐색 look이 연속 빈응답이고 조향 근거(anchor 또는 동결 goal)가 있으면, anchor
            # 원거리 탐색 look만 소각 대신 건너뛰고 anchor goal-seek로 계속합니다(191s 플랩 방어).
            # 예외 — near_look(anchor 근접 look)은 근접 미검출 자가치유의 유일한 신호원이라 항상 실행하고,
            # in_range 후의 place 전 confirm look은 이 분기에 오지 않아 역시 항상 실행됩니다(오배송 방지).
            # 동결 goal은 루프 헤드에서 frozen anchor로 승격되므로 anchor 존재 == 조향 근거 존재입니다
            # (survey 동결 후 anchor 미설치 시에도 abandon이 goal-seek를 굶기지 않음, Codex Major 반영).
            if vlm_empty_streak >= PAD_VLM_ABANDON_N and anchor is not None and not near_look:
                _trace_step(
                    memory, action="vlm_abandon", streak=vlm_empty_streak, letter=letter,
                    anchor_dist_m=round(anchor_dist, 2) if anchor_dist is not None else None,
                )
                if verbose:
                    print(
                        f"           VLM 조기 포기(탐색 빈응답 {vlm_empty_streak}회 ≥ {PAD_VLM_ABANDON_N})"
                        " -> look 생략, anchor goal-seek 계속"
                    )
                anchor_streak = 0  # anchor goal-seek 재개(원거리 anchor면 다음 attempt 티어2).
                ls_streak = 0
                continue
            face_turn = await _look_for_sign(
                ctx, letter, held_color, api_key, memory, entry,
                reason=reason, verbose=verbose,
                # M2: 동결/anchor에 근접(≤PAD_SIGN_LOST_NEAR_M)한데 표지를 잃으면 위쪽 재확인 1회 허용.
                near=(anchor_dist is not None and anchor_dist <= PAD_SIGN_LOST_NEAR_M),
            )
            ls_streak = 0
            anchor_streak = 0
            vlm_empty_streak = vlm_empty_streak + 1 if face_turn is None else 0  # P5 streak 갱신.
            if verbose:
                shown = "None" if face_turn is None else f"{face_turn:+.0f}°"
                print(
                    f"  [pad {attempt}] look face_turn={shown}"
                    f"  (vlm {time.perf_counter() - t0:.1f}s)  {await _pose_str(ctx)}"
                )
            if face_turn is not None:
                anchor_near_miss = 0
            elif anchor is not None:
                # VLM 미검출이어도 anchor가 있으면 길을 잃지 않습니다: 블라인드 배회(전진/회전)
                # 로 빠지지 않고 다음 반복에서 anchor 재조준을 계속합니다. 단 anchor 근접에서
                # 미검출이 반복되면 점 추정이 오염/무효라는 뜻이므로 폐기(자가치유)합니다.
                if anchor_dist is not None and anchor_dist <= PAD_ANCHOR_NEAR_M:
                    anchor_near_miss += 1
                    if anchor_near_miss >= PAD_ANCHOR_NEAR_MISS_LIMIT:
                        entry["anchor"] = None
                        _drop_sign_map(memory, letter)  # 동결 목표도 오염 → 함께 재구축.
                        anchor_near_miss = 0
                        vlm_empty_streak = 0  # M1-4: 조향 근거 폐기 → 재구축될 새 anchor에 look 예산 재부여.
                        _trace_step(
                            memory, action="anchor_drop", pose=pose,
                            note=f"근접 미검출 {PAD_ANCHOR_NEAR_MISS_LIMIT}회",
                        )
                        if verbose:
                            print("           anchor 근접인데 sign 미검출 반복 -> anchor 폐기")
                        continue
                if verbose:
                    print("           VLM 미검출이지만 anchor 유지 -> 재조준 계속")
                continue

        if face_turn is None:
            # 안 보이면 회전만 반복하지 말고 전진해 시야(vantage)를 바꿉니다(occlusion 회피).
            # PAD_FWD_BEFORE_TURN 회마다 한 번은 크게 회전해 다른 방향도 봅니다.
            fails += 1
            if fails % PAD_FWD_BEFORE_TURN == 0:
                if verbose:
                    print(f"           못 찾음 {fails}회 -> {PAD_SEARCH_TURN_DEG:.0f}° 회전")
                await _turn_by_deg(ctx, PAD_SEARCH_TURN_DEG)
            else:
                if verbose:
                    print("           못 찾음 -> 전진(새 시야)")
                side = _choose_detour_side(detour_side, detour_fails)
                if not await _advance_or_detour(ctx, side, memory=memory, verbose=verbose):
                    detour_fails[_side_name(side)] += 1
                    detour_side = -side  # 우회로도 막히면 다음엔 반대쪽으로.
            continue

        fails = 0
        if abs(face_turn) > 1.0:
            # 이후 전진이 stall하면 target이 있던 쪽으로 우회해 시야를 유지합니다.
            detour_side = 1.0 if face_turn > 0 else -1.0
        if abs(face_turn) > PAD_FACE_TOL_DEG:
            await _turn_by_deg(ctx, face_turn)  # 목표 sign을 정면에 맞춤(폐루프)
        # 마주봄: placard 근접이면 도착. 아니면 odometry 전진 한 청크만 하고 다음 look에서 재보정합니다
        # (색블롭 servo는 green flicker로 target을 잃으므로 접근에 쓰지 않습니다).
        # R6(관측 모드): 도착 게이트와 동일한 인식으로 면적 표본을 얻어 수렴 추세를 기록만
        # 합니다 — 행동은 바꾸지 않고, R4 라이브에서 임계 보정 후 발동으로 승격합니다.
        blob = await _best_color_blob(ctx, held_color, PAD_ARRIVAL_AREA)
        blob_area = blob.blob_area if blob is not None else 0
        approach_history.append({"area": blob_area, "face_turn": face_turn})
        converging = _approach_converging(approach_history)
        # M2: 도착 위험 계측(관측 전용). close_goal(동결) 있으면 잔여 d와 arrive_risk를 남깁니다.
        _as_goal = memory.sign_goals.get(letter) if memory is not None else None
        _as_d = _radius_from(pose, _as_goal) if _as_goal is not None else None
        _trace_step(
            memory, action="approach_sample", source=source, area=blob_area,
            face_turn_deg=round(face_turn, 1), converging=converging,
            d_to_goal_m=None if _as_d is None else round(_as_d, 2),
            arrive_risk=_arrive_risk(blob_area, PAD_ARRIVAL_AREA, _as_d, PAD_ANCHOR_NEAR_M),
        )
        if verbose and converging is not None:
            print(
                f"           수렴 판정(관측): {'수렴 중' if converging else '비수렴'}"
                f" (표본 {len(approach_history)}개, area={blob_area})"
            )
        # M4-1: 면적 단독 도착 → 면적 AND 동결 목표 잔여거리. area_ok는 기존 in_range 그대로이고, 잔여는
        # 위 approach_sample에서 계산한 _as_goal/_as_d를 재사용합니다(pose는 회전 불변이라 attempt 헤드 pose로
        # 충분 — 도착 attempt엔 회전만 개입). 접근 면 문제: 표지 blob은 1.47m 밖에서도 도착 면적을 채우지만
        # place 반경(1.2m) 밖일 수 있어, 잔여 AND로 '면적만 찬 오도착'을 차단합니다.
        area_ok = (
            blob is not None
            and blob.blob_area >= PAD_ARRIVAL_AREA
            and abs(blob.angle_deg) <= CENTER_TOLERANCE_DEG * 1.5
        )
        residual_ok = _as_d is None or _as_d <= PAD_ARRIVAL_RESIDUAL_M
        in_range = area_ok and residual_ok
        # 폐기 경로(G4 무한 grind 탈출구): area_ok인데 잔여 초과가 FAIL_N회 반복되면 도달 불가 오염 goal로
        # 보고 동결·anchor를 폐기해 면적 단독 판정으로 degrade합니다(다음 attempt부터 _as_d=None → residual_ok
        # =True). 잔여 AND와 이 폐기는 한 몸 — 분리하면 PAD_OUTER_MAX 소진 무한 grind가 됩니다(같은 diff 반입).
        if area_ok and not residual_ok:
            residual_fail_streak += 1
            _trace_step(
                memory, action="arrive_blocked", pose=pose,
                d_to_goal_m=None if _as_d is None else round(_as_d, 2),
                area=blob_area, streak=residual_fail_streak,
            )
            if verbose:
                print(
                    f"  [pad {attempt}] 면적 충족이나 동결 잔여 d={_as_d}m > {PAD_ARRIVAL_RESIDUAL_M}m"
                    f" (잔여 초과 {residual_fail_streak}/{PAD_ARRIVAL_RESIDUAL_FAIL_N})"
                )
            if residual_fail_streak >= PAD_ARRIVAL_RESIDUAL_FAIL_N:
                if entry is not None:
                    entry["anchor"] = None  # frozen anchor 동반 폐기(오염 좌표 goal-seek 차단).
                _drop_sign_map(memory, letter)  # 동결·ray 폐기 + M1 쿨다운 자동 마크(재응집 차단).
                vlm_empty_streak = 0
                _trace_step(memory, action="residual_drop", letter=letter, streak=residual_fail_streak)
                residual_fail_streak = 0
                if verbose:
                    print("           -> 도달 불가 오염 goal 폐기: 면적 단독 판정으로 degrade")
        elif in_range:
            residual_fail_streak = 0  # 도착(사건 해소). area_ok 미성립 attempt는 streak 불변(접근 중).
        if in_range:
            if source in ("last_seen", "anchor"):
                # 기억(ray/anchor)만으로 조향해 왔으므로 place 전 VLM 1회로 진짜 목표 pad인지
                # 확정합니다(색블롭은 같은 색 source에서도 참 -> 오배송 방지).
                confirm = await _look_for_sign(
                    ctx, letter, held_color, api_key, memory, entry,
                    reason=f"place 전 최종 확인({source} 조향으로 도착)", verbose=verbose,
                )
                if confirm is None:
                    if entry is not None:
                        entry["last_seen"] = None  # ray 신뢰 소진: 다음 반복은 VLM부터.
                    if (
                        source == "anchor"
                        and anchor_dist is not None
                        and anchor_dist <= PAD_ANCHOR_NEAR_M
                        and entry is not None
                    ):
                        # anchor 부근의 큰 중앙 블롭인데 sign 확인 실패 -> 근접 미검출로 셈.
                        anchor_near_miss += 1
                        if anchor_near_miss >= PAD_ANCHOR_NEAR_MISS_LIMIT:
                            entry["anchor"] = None
                            _drop_sign_map(memory, letter)  # 동결 목표도 오염 → 함께 재구축.
                            anchor_near_miss = 0
                            vlm_empty_streak = 0  # M1-4: 조향 근거 폐기 → 재구축될 새 anchor에 look 예산 재부여.
                            _trace_step(
                                memory, action="anchor_drop", pose=pose,
                                note="confirm 미검출 반복",
                            )
                            if verbose:
                                print("           anchor 부근 confirm 실패 반복 -> anchor 폐기")
                    fails += 1
                    continue
                if abs(confirm) > PAD_FACE_TOL_DEG:
                    await _turn_by_deg(ctx, confirm)
                    # R1 재게이트: 큰 confirm 각은 "게이트를 통과시킨 중앙 블롭이 목표 pad가
                    # 아니었다"는 증거이고(같은 색 source 등), 방금 회전으로 게이트 결과 자체가
                    # 낡았습니다. 새 heading에서 근접 게이트를 다시 통과해야만 도착을 선언합니다
                    # — replay/vlm 분기의 "회전 먼저 → 게이트 나중" 불변식과 일치(리뷰 확정 결함 ①).
                    if not await _target_in_range(ctx, held_color, PAD_ARRIVAL_AREA):
                        if verbose:
                            print(
                                f"           confirm {confirm:+.0f}° 회전 후 재게이트 실패"
                                " -> 도착 취소, 재관찰"
                            )
                        _trace_step(
                            memory, action="regate_fail", source=source,
                            face_turn_deg=round(confirm, 1), pose=await _get_pose(ctx),
                        )
                        continue
            # 마무리 접근: blob 도착 지점은 place 반경(팔레트 1.2m) 밖일 수 있음(라이브 확정
            # 1.47m) → 동결 목표가 있으면 그 좌표까지 남은 거리를 소회전+짧은 전진으로 좁힌 뒤
            # 도착을 선언합니다. 동결 전이면(목표 미확정) 기존처럼 blob 판정만으로 진행합니다.
            close_goal = memory.sign_goals.get(letter) if memory is not None else None
            need_push = True
            if close_goal is not None:
                _cd = await _close_to_goal(ctx, close_goal, memory=memory, verbose=verbose)
                need_push = _cd > PAD_CLOSE_ENOUGH_M  # 마무리 접근이 stall로 반경 밖에서 멈춘 경우만.
            # ★I(run12 전도)★ anchor 잔여 sanity: area 게이트는 같은 색 거대 blob(blue 배달의 벨트)에
            # 속으므로, anchor가 '아직 멀다'고 증언하면(4.4m vs 도착) area가 커도 도착을 거부하고
            # 재관찰합니다. 이 게이트가 진입 push(F 무가드 + H 무상한)도 함께 막아 벨트 grind 전도를
            # 차단합니다. anchor 미형성이면 게이트 생략(초기 run1식 blob-only 도착 유지).
            _pre_pose = await _get_pose(ctx)
            _pre_d = _arrive_goal_dist(memory, letter, held_color, _pre_pose)
            if _pre_d is not None and _pre_d > PAD_ARRIVAL_ANCHOR_MAX_M:
                if verbose:
                    print(
                        f"           도착 거부: anchor 잔여 {_pre_d:.1f}m >"
                        f" {PAD_ARRIVAL_ANCHOR_MAX_M}m (area 신기루 의심) -> 재관찰"
                    )
                _trace_step(
                    memory, action="arrive_reject_anchor_far",
                    d_to_goal_m=round(_pre_d, 2), pose=_pre_pose,
                )
                continue
            # §1 push-through: 동결이면 _close_to_goal이 stall로 못 좁힌 잔여를, 무동결이면(run1) 전
            # 구간을 blob 서보 + vx 단계 상향으로 place 반경까지 밀어넣습니다(1.34m place→0배달 수정).
            # ★F(run8)★ belt_guard=False: VLM으로 pad를 확인하고 도착한 뒤의 ≤0.7m 진입이라 pad가
            # 로봇과 벨트 사이에 있고(run1~5 place push 전도 0회), 가드를 켜면 pad C 정면의 벨트
            # 배경 오탐으로 마지막 진입이 전부 차단됩니다(run8: 0.00m ×11 → place 반경 영원 미달).
            if need_push and held_color:
                # ★H★ oversize_ok: 근접 pad blob은 상한 초과로 사라져 push가 1청크에 끝나던 결함
                # (run11 0.09m) — 팔레트 접촉(stall)까지 서보 유지해 place 착지점을 존 중심으로.
                await _push_through_to_target(
                    ctx, held_color, memory=memory, verbose=verbose,
                    belt_guard=False, oversize_ok=True,
                )
            if verbose:
                print(f"  [pad {attempt}] 정면·근접 -> 도착  {await _pose_str(ctx)}")
            ap = await _get_pose(ctx)
            _ad = _arrive_goal_dist(memory, letter, held_color, ap)
            _trace_step(
                memory, action="arrive", source=source, pose=ap,
                d_to_goal_m=_ad,
                goal_bearing_deg=(  # M0-2: 접근 면 방위(로봇→goal 월드 방위각), 동결 전이면 None.
                    round(math.degrees(math.atan2(_g["y"] - ap["y"], _g["x"] - ap["x"])), 1)
                    if (_g := memory.sign_goals.get(letter) if memory is not None else None) is not None
                    else None
                ),
                arrive_area=blob_area, area_thresh=PAD_ARRIVAL_AREA,
                arrive_risk=_arrive_risk(blob_area, PAD_ARRIVAL_AREA, _ad, PAD_ANCHOR_NEAR_M),  # M2 관측.
            )
            return True, attempt + 1
        if verbose:
            print(f"  [pad {attempt}] 정면(src={source}) -> 접근 전진")
        side = _choose_detour_side(detour_side, detour_fails)
        # ★사용자 제안(능동 벨트 우회): 들이받기 전에 카메라로 전방 벨트(초대형 belt_color blob)를
        # 식별하면, free-space 열린 쪽(freer_side)으로 꺾어 벨트를 커밋해 따라갑니다. 반응형 우회는
        # 램→wedge 후 발동해 양측막힘으로 실패했습니다(run3 4/6). 선제 전환으로 wedge 자체를 피하고,
        # 방향은 카메라 free-space(맹목 90° 아님 — freer_side, 없으면 pad 방위 side)라 결정적입니다.
        if await _belt_blocks_forward(ctx, memory):
            _prof = await _probe_free_space(ctx)
            belt_side = (_prof["freer_side"] if (_prof and _prof.get("freer_side")) else side) or 1.0
            _trace_step(
                memory, action="belt_bypass_proactive", side=belt_side, pose=await _get_pose(ctx),
            )
            if verbose:
                print(
                    f"           능동 벨트 우회: 전방 벨트 식별 -> free-space {_side_name(belt_side)}쪽"
                    f" {BELT_FOLLOW_CHUNKS}청크 따라가기"
                )
            await _lateral_bypass(ctx, belt_side, BELT_FOLLOW_CHUNKS, memory=memory, verbose=verbose)
            continue
        # M2 sector map(활성 시): 목표 방위 섹터에 랙이 1.2~1.8m로 다가오면(원거리 진입 억제·용도2) 인접
        # 열린 섹터로 staging 1회(용도3, 동시 1개) 또는 열린 쪽을 detour side로 선제 채택(용도1: freer_side
        # 1순위). 발밑 free_space는 _advance_or_detour 내부가 최우선 처리하므로 용도 우선순위(발밑>sector
        # map)가 유지됩니다. rack_map None(미보정·비활성)이면 이 블록 전체가 무영향입니다.
        if rack_map is not None and _rack_front_blocked(rack_map):
            stage_side = _rack_staging_side(rack_map)
            if not staged_once and stage_side != 0.0:
                staged_once = True
                stage_chunks = _stage_chunk_count(RACK_STAGE_DIST_M)
                if verbose:
                    print(
                        f"           sector map staging({_side_name(stage_side)}) ->"
                        f" {RACK_STAGE_DIST_M}m({stage_chunks}청크) 경유 우회"
                    )
                await _turn_by_deg(ctx, stage_side * PAD_STALL_DETOUR_DEG)
                for _ in range(stage_chunks):  # 실제 ~RACK_STAGE_DIST_M 이동(stall이면 조기 종료).
                    if not await _advance_or_detour(
                        ctx, stage_side, memory=memory, action="rack_stage", verbose=verbose
                    ):
                        break
                await _turn_by_deg(ctx, -stage_side * PAD_STALL_DETOUR_DEG)
                continue
            side = _rack_preferred_side(rack_map, side)
            if verbose:
                print(f"           sector map: 목표 섹터 랙 근접(1.2~1.8m) -> 선제 우회 side={_side_name(side)}")
        if await _advance_or_detour(ctx, side, memory=memory, verbose=verbose):
            hard_stall_streak = 0
        else:
            # 초근접·정면 확인 도착(색블롭 area=0 우회): 이번 반복에서 sign을 VLM으로 방금 정면
            # 확인했고(source=vlm, 오배송 위험 없음), anchor 추정 거리가 PAD_PLACE_NEAR_M 안이며
            # 전진이 막혀(직진+detour 모두 stall) 더 다가갈 수 없으면, 색블롭 도착 크기를 못 채워도
            # 도착으로 선언합니다. 초근접에서 바닥 pad가 카메라 하단 밖으로 나가 blob=0이 되는 것을
            # place 실패로 오판하던 라이브 결함 해소((2.69,1.53)에서 'C' center conf 0.99인데
            # area=0으로 18 attempt 소진 후 전도 — 확정). anchor_dist는 anchor 존재 시에만 설정되어
            # anchor 없으면 조건이 성립하지 않습니다(초근접 오도착 방지).
            if (
                source == "vlm"
                and face_turn is not None
                and anchor_dist is not None
                and anchor_dist <= PAD_PLACE_NEAR_M
            ):
                if verbose:
                    print(
                        f"  [pad {attempt}] 초근접(anchor d={anchor_dist:.2f}m)·sign 정면 확인·"
                        "전진 막힘 -> 도착 선언(블롭 미도달 무시)"
                    )
                ap = await _get_pose(ctx)
                _ad = _arrive_goal_dist(memory, letter, held_color, ap)
                _trace_step(
                    memory, action="arrive", source="close_blocked",
                    anchor_dist_m=round(anchor_dist, 2), pose=ap,
                    d_to_goal_m=_ad,
                    goal_bearing_deg=(  # M0-2: 접근 면 방위(로봇→goal 월드 방위각), 동결 전이면 None.
                        round(math.degrees(math.atan2(_g["y"] - ap["y"], _g["x"] - ap["x"])), 1)
                        if (_g := memory.sign_goals.get(letter) if memory is not None else None) is not None
                        else None
                    ),
                    arrive_area=0, area_thresh=PAD_ARRIVAL_AREA,  # blob=0 우회 경로.
                    arrive_risk=_arrive_risk(0, PAD_ARRIVAL_AREA, _ad, PAD_ANCHOR_NEAR_M),  # M2 관측.
                )
                return True, attempt + 1
            detour_fails[_side_name(side)] += 1
            detour_side = -side
            hard_stall_streak += 1
            # detour로도 못 뚫는 반복 stall(pad가 벨트 같은 선형 구조물 너머) -> 목표 쪽으로
            # 측면 우회해 구조물을 따라 이동. R6 비수렴 신호가 이 escalate를 뒷받침합니다.
            # escalation 기준에 bypass_rounds를 더합니다: 선제 우회의 '옆걸음 성공'이
            # hard_stall_streak를 매번 리셋해 라이브에서 항상 2청크에 머물렀습니다(확정).
            # 같은 nav에서 우회가 반복 발동될수록 발동이 빨라지고 2→3→4청크로 더 멀리
            # 따라가 벽의 끝/틈을 지나갈 확률을 높입니다.
            chunks = _bypass_chunks(hard_stall_streak + bypass_rounds)
            if chunks:
                p_now = await _get_pose(ctx)
                win_side = (
                    _preferred_side_from_history(
                        memory.detour_wins, p_now["x"], p_now["y"], p_now["yaw_deg"]
                    )
                    if memory is not None
                    else None
                )
                # 방향은 검증된 통과 경험(최근 성공 우회) > 표지가 보였던 쪽 순으로 —
                # 지난 런에서 face_turn 부호만 따르다 검증된 북쪽 대신 남쪽으로 꺾여 갇혔습니다.
                target_side = win_side if win_side is not None else (1.0 if face_turn > 0 else -1.0)
                bypassed = await _lateral_bypass(
                    ctx, target_side, chunks, memory=memory, verbose=verbose
                )
                hard_stall_streak = 0
                bypass_rounds += 1
                if bypassed:
                    # 우회 직후 곧장 재조준하면 방금 피한 구조물로 되꺾일 수 있어 1청크 직진해
                    # 모서리를 확실히 지난 뒤, anchor가 있으면 VLM 재확인을 강제해(연속 상한
                    # 소진; ray 티어는 anchor 존재 시 생략되므로 이번엔 가로채이지 않음)
                    # 새 위치에서 점 추정을 갱신합니다.
                    await _advance_or_detour(
                        ctx, target_side, memory=memory,
                        action="bypass_clear", verbose=verbose,
                    )
                    anchor_streak = PAD_ANCHOR_MAX_REUSE
            # M4-2: 측면 우회(선형)가 ORBIT_TRIGGER_BYPASS_ROUNDS회 발동해도 hard_stall이 안 풀리면
            # 코너 웨지로 판단 → 동결 goal 주위를 orbit해 다른(남/동) 면을 찾습니다(nav당 1회). orbit이
            # 열린 면을 확보하면 이번 nav의 우회 카운터를 리셋해 도착 판정을 새 위치에서 재개합니다.
            orbit_goal = memory.sign_goals.get(letter) if memory is not None else None
            # Option A 봉인: orbit은 동결 goal 전제 + M3 미보정 잠정 상수라 8런 트리거 0 — 스위치로 잠급니다.
            if ORBIT_ENABLED and orbit_goal is not None and bypass_rounds >= ORBIT_TRIGGER_BYPASS_ROUNDS and not orbit_used:
                orbit_used = True
                if await _orbit_for_face(ctx, orbit_goal, memory=memory, verbose=verbose):
                    hard_stall_streak = 0
                    bypass_rounds = 0

    if verbose:
        print(f"  [pad] FAIL: {PAD_OUTER_MAX} attempt 내 pad 도착 실패")
    return False, PAD_OUTER_MAX


def _recover_plan(
    front_blocked: bool,
    freer_side: float,
    fails: int,
    pick_fail_streak: int,
    recent_pick_fail: bool,
    *,
    backup_dur: float = RECOVER_BACKUP_DUR,
    backup_dur_blocked: float = RECOVER_BACKUP_DUR_BLOCKED,
) -> dict[str, Any]:
    """recover_motion의 순수 결정부: 후퇴 시간·회전 방향·wedge 여부를 계산합니다(async·ctx 무관).

    ★run14 전도 방어★ 회전 아크는 vx>0 전진을 동반하므로 구조물에 붙은 채 돌면 파고들어 전도한다.
    그래서 ① 전방 막힘이면 후퇴를 키워 이탈부터 하고 ② 양측이 다 막힌 wedge(전방막힘 + 열린 쪽
    없음)면 회전을 아예 생략(직진 후퇴만)하며 ③ 그 외에는 카메라가 지목한 freer_side로만 아크
    회전한다. freer_side 부호 미정(0)이면 기존 기본 방향(좌, +1)을 유지한다.

    반환 {"backup_dur","wedged","wz_sign","turn_dur"}. pytest로 잠근다.
    """
    dur = backup_dur_blocked if front_blocked else backup_dur
    wedged = front_blocked and freer_side == 0.0
    if wedged:
        return {"backup_dur": dur, "wedged": True, "wz_sign": 0.0, "turn_dur": 0.0}
    wz_sign = freer_side if freer_side != 0.0 else 1.0
    extra = 0.3 * pick_fail_streak if recent_pick_fail else 0.0
    turn_dur = min(0.8 + 0.4 * fails + extra, 2.5)
    return {"backup_dur": dur, "wedged": False, "wz_sign": wz_sign, "turn_dur": turn_dur}


async def recover_motion(ctx: Any, memory: AgentMemory, reason: str | None = None) -> dict[str, Any]:
    """Target loss, blocked motion, failed manipulation에서 recover합니다.

    TODO:
    - Step back, rotate, rescan, detour 선택, LLM skip 요청 등을 구현하세요.
    - 같은 failed action을 무한 반복하지 않도록 memory를 사용하세요.
    """
    # 반복 실패를 memory로 파악해 회전량을 키우고, 너무 잦으면 skip을 권고합니다.
    color = memory.active_color or memory.held_color
    fails = memory.failed_attempts.get(color, 0) if color else 0

    # ★run14 전도 방어(구조물-근접 회전)★: 회전 아크는 vx>0 전진을 동반하므로 벨트/랙에 붙은 채
    # 돌면 구조물로 파고들며 전도한다(오늘 전도 6/6이 구조물 접촉 중 회전·측면). 그래서 회전 전에
    # 카메라로 전방·측방 여유를 보고(_probe_free_space), 후퇴로 이탈시킨 뒤 '열린 쪽'으로만 아크
    # 회전한다. 양측이 다 막힌 wedge면 회전을 아예 접고 직진 후퇴만 한다(_recover_plan 순수 결정).
    profile = await _probe_free_space(ctx)
    front_blocked = bool(profile and not profile.get("clear", True))
    freer = float(profile["freer_side"]) if profile and profile.get("freer_side") else 0.0
    plan = _recover_plan(
        front_blocked, freer, fails, memory.pick_fail_streak, bool(memory.recent_pick_fail),
    )

    # 1) 후퇴로 구조물에서 이탈합니다(전방 막힘이면 더 크게 물러나 접촉을 확실히 끊음).
    await move_velocity(ctx, vx=RECOVER_BACKUP_VX, duration_s=plan["backup_dur"])
    if plan["wedged"]:
        # 2a) wedge(양측 막힘): 회전(전진 동반)은 구조물로 파고들어 전도한다 → 직진 후퇴만 한 번 더.
        await move_velocity(ctx, vx=RECOVER_BACKUP_VX, duration_s=plan["backup_dur"])
    else:
        # 2b) 열린 쪽(freer_side)으로만 아크 회전. 실패가 쌓일수록 더 오래 선회해 다른 경로를 찾음.
        await move_velocity(
            ctx, vx=SWEEP_VX, wz=SWEEP_WZ * plan["wz_sign"], duration_s=plan["turn_dur"],
        )
    _trace_step(
        memory, action="recover", reason=reason, front_blocked=front_blocked,
        freer_side=freer, wedged=plan["wedged"], wz_sign=plan["wz_sign"],
        backup_s=round(plan["backup_dur"], 2), turn_s=round(plan["turn_dur"], 2),
    )
    # 3) 회전 뒤 주변을 다시 훑어 target 재획득 기회를 만들고 head를 정면으로 둡니다.
    await scan_head(ctx)
    await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)

    # 4) 같은 target에서 반복 실패하면(LLM이 skip_target을 고르도록) skip을 권고합니다.
    suggest_skip = color is not None and fails >= 3
    return {
        "action": "recover",
        "reason": reason,
        "color": color,
        "fails": fails,
        "suggest_skip": suggest_skip,
        "status": "stepped_back_and_rotated",
    }


async def _clean_cube_visible(ctx: Any) -> bool:
    """현재 프레임에 _is_clean_cube 통과 blob이 하나라도 있으면 True(source-seek 종료 판정)."""
    return any(_is_clean_cube(d, CUBE_ARRIVAL_AREA) for d in await perceive(ctx))


async def _source_seek_fallback(ctx: Any, memory: AgentMemory, *, verbose: bool = False) -> bool:
    """소스 단서 전무 시 폴백: 가장 열린 방향으로 ~SRC_FALLBACK_ADVANCE_M 전진 후 무료 재스윕(VLM 0회).

    free_space_profile.freer_side로 열린 쪽을 고르고(없으면 좌), _advance_or_detour로 전진하며 매
    청크 후 clean cube를 확인합니다. 못 찾으면 head 스캔 1회로 재스윕 — 전부 OpenCV, VLM 0회.
    """
    profile = await _probe_free_space(ctx)
    rack_map = await _rack_map_from_frame(ctx)   # M2(활성 시): 폴백 side에 sector map freer_side 1순위 승격.
    if rack_map is not None and rack_map.get("freer_side", 0.0) != 0.0:
        side = rack_map["freer_side"]
    elif profile is not None and profile.get("freer_side", 0.0) != 0.0:
        side = profile["freer_side"]
    else:
        side = 1.0
    n = min(SRC_SEEK_MAX_CHUNKS, max(1, round(SRC_FALLBACK_ADVANCE_M / (FORWARD_EFF_SPEED_MPS * PAD_ADVANCE_DUR))))
    for _ in range(n):
        await _advance_or_detour(ctx, side, memory=memory, action="source_fallback", verbose=verbose)
        if await _clean_cube_visible(ctx):
            if verbose:
                print("  [source-seek] 폴백 전진 중 clean cube 가시화 -> 종료")
            return True
    # 무료 재스윕 1회(head 스캔, VLM 0회).
    dets = await scan_head(ctx)
    await set_head(ctx, yaw=0.0, pitch=HEAD_PITCH_TRACK)
    return any(_is_clean_cube(d, CUBE_ARRIVAL_AREA) for d in dets)


async def _source_seek_step(ctx: Any, memory: AgentMemory, *, verbose: bool = False) -> bool:
    """clean cube 미시야 시 소스(A)를 향해 한 턴 전진해 'clean cube 가시화' 상태를 만듭니다(§5.2).

    종료 = 매 청크 후 perceive에서 _is_clean_cube 통과 blob이 보이면 즉시 True — 먼 큐브 조기 종료는
    의도된 정상 동작이고 접근은 navigate_to_cube 소관입니다(별도 면적 게이트 금지, §8-11). 전진은
    _advance_or_detour 재사용(stall·free-space preempt·lateral bypass 전부 상속), 턴당 ≤SRC_SEEK_MAX_CHUNKS.
    접근 우선순위 _source_target_priority: cube→goal→ray→fallback. 전부 VLM 0회. 매 턴 LLM에 복귀합니다.
    """
    kind, payload = _source_target_priority(memory)
    if verbose:
        print(f"  [source-seek] 우선순위={kind}")
    if kind == "fallback":
        # 소스 단서 전무: 누계 캡 안에서만 폴백, 초과 시 기존 아크-스윕에 위임.
        if memory.source_fallback_rounds >= SRC_FALLBACK_MAX_ROUNDS:
            if verbose:
                print(f"  [source-seek] 폴백 누계 {SRC_FALLBACK_MAX_ROUNDS}턴 소진 -> visual_search 위임")
            return await visual_search(ctx, None)
        memory.source_fallback_rounds += 1
        return await _source_seek_fallback(ctx, memory, verbose=verbose)
    memory.source_fallback_rounds = 0  # 유효 단서 확보 → 폴백 카운터 리셋.
    for chunk in range(SRC_SEEK_MAX_CHUNKS):
        pose = await _get_pose(ctx)
        if kind == "goal":
            _, turn = _face_turn_to(pose, payload)
        else:  # cube(sighting) 또는 ray: payload에서 world 방위를 뽑아 현재 yaw와의 차로 재조준각 산출.
            bearing = float(payload["bearing_deg"]) if kind == "cube" else float(payload)
            turn = _angle_diff_deg(bearing, float(pose.get("yaw_deg", 0.0)))
        side = 1.0 if turn > 0 else -1.0
        if abs(turn) > PAD_FACE_TOL_DEG:
            await _turn_by_deg(ctx, max(-PAD_STALL_DETOUR_DEG, min(PAD_STALL_DETOUR_DEG, turn)))
        await _advance_or_detour(ctx, side, memory=memory, action="source_advance", verbose=verbose)
        if await _clean_cube_visible(ctx):
            if verbose:
                print(f"  [source-seek] clean cube 가시화 -> 소스-seek 종료(청크 {chunk + 1})")
            return True
    return await _clean_cube_visible(ctx)


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

    M0-2: 국면(next_action)별 wall-clock 소비율을 finally에서 phase trace 1건으로 남깁니다
    (관측 전용, 행동 불변) — visual_navigate_to_pad↔_navpad_impl과 같은 얇은 래퍼 패턴이라
    분기 실행부는 _execute_decision_impl에 그대로 있고 모든 반환 경로에서 phase가 계량됩니다.
    """
    t_phase = time.perf_counter()
    try:
        return await _execute_decision_impl(ctx, decision, observation, memory)
    finally:
        _trace_step(
            memory, action="phase", phase=decision.next_action,
            wall_s=round(time.perf_counter() - t_phase, 2),
        )


async def _execute_decision_impl(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """execute_decision의 분기 실행부(래퍼가 국면 wall-clock을 계량). 로직은 종전 그대로입니다."""
    action = decision.next_action
    # 아직 큐브를 안 들었으면 '획득' 단계(색 고정 없이 최근접 깨끗한 큐브), 들었으면 '배송'
    # 단계(실제 든 색 held_color로 pad를 찾음). color-blind pick과 싸우지 않는 것이 핵심입니다.
    acquiring = memory.held_color is None

    if action in {"search_cube", "search_pad"}:
        if action == "search_pad" and memory.held_color:
            # pad는 색블롭이 아니라 VLM signage로 찾습니다(같은 색 source/큐브와 구분).
            found = await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True)
            return {"action": action, "found": found}
        if action == "search_cube" and acquiring:
            # 획득 탐색: clean cube가 이미 보이면 기존 흐름(다음 cycle pick_cube)에 맡깁니다. 미시야면
            # 소스-seek로 소스(A)를 향해 전진해 'clean cube 가시화'를 만듭니다(신규 action 없이 기존
            # search_cube 실행부 강화, §5.2 — 랜덤 스폰 setup 1~50 대응).
            if await _clean_cube_visible(ctx):
                return {"action": action, "found": True}
            found = await _source_seek_step(ctx, memory, verbose=True)
            return {"action": action, "found": found}
        search_target = memory.held_color or decision.target_color
        found = await visual_search(ctx, search_target)
        return {"action": action, "found": found}

    if action in {"navigate_to_cube", "navigate_to_pad"}:
        if action == "navigate_to_pad" and memory.held_color:
            # 배송: pad는 VLM signage로 방향을 잡아 아크로 접근합니다(색블롭 불가).
            # A2(자기완결 place): 직전 도착 후 거의 안 움직였으면 navpad 재실행(도착 재증명 17~104s
            # + 재확인 VLM 플랩)을 생략하고 도착 유지로 답합니다 — run8 reached=True ×5 재증명 제거.
            nav_letter = DESTINATION_SIGN_RULES.get(memory.held_color)
            pose_now = await _get_pose(ctx)
            if _pad_arrival_fresh(memory.pad_arrived, nav_letter, pose_now, PAD_ARRIVED_STICKY_M):
                _trace_step(memory, action="pad_arrival_reuse", letter=nav_letter, pose=pose_now)
                reached = True
            else:
                reached = await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True)
                if reached and nav_letter is not None:
                    p_arr = await _get_pose(ctx)
                    memory.pad_arrived = {"letter": nav_letter, "x": p_arr["x"], "y": p_arr["y"]}
        elif action == "navigate_to_cube" and acquiring:
            # 획득: clean 큐브가 이미 pick 가능할 만큼 가까우면 정렬 nav를 생략합니다. 짧은 회전은
            # 학습 정책 ramp-up으로 거의 안 돌아 근접 큐브 주위를 맴돌 뿐이고, pick_entity는 각도
            # 무관 최근접 큐브를 스스로 파지하므로 정렬이 불필요합니다(다음 cycle에서 pick_cube).
            ready, _seen = await _clean_cube_ready(ctx)
            reached = ready or await visual_navigate_to_target(ctx, None)
        else:
            nav_target = memory.held_color or decision.target_color
            reached = await visual_navigate_to_target(ctx, nav_target)
        if not reached:
            # 도착 실패 -> recovery로 자세를 바꾼 뒤 다음 cycle에서 재시도합니다.
            recovery = await recover_motion(ctx, memory, reason=f"{action}_failed")
            return {"action": action, "reached": False, "recovery": recovery}
        return {"action": action, "reached": True}

    if action == "pick_cube":
        # ★사용자(정지 큐브 우선): color-blind pick_entity는 3D 최근접 큐브를 색 무관하게 잡지만, clean
        # 큐브가 컨베이어 위에 있으면 접근하는 사이 벨트가 실어 옮겨 1.2m 반경에 못 넣고 실패·relocate
        # 순환에 빠집니다(run2/4 확정: 이동 큐브 추격). 그래서 2프레임 차로 '정지' clean 큐브만 인정하고,
        # 없으면 이동 큐브를 쫓지 말고 재배치해 정지 큐브(벨트 끝/떨어진 더미)를 찾습니다. 실제 잡은 색은
        # 이후 get_held_cube_info로 확정해 그 색 pad로 배송합니다(채점 색 무관 30pt/개).
        stationary = await _stationary_clean_cubes(ctx, CUBE_ARRIVAL_AREA, memory=memory)
        if not stationary:
            recovery = await recover_motion(ctx, memory, reason="pick_no_stationary_cube")
            return {"action": "pick_cube", "result": None, "positioned": False, "recovery": recovery}
        # ★E1(run7 마비)★ '최대 blob' 선택은 벨트색(벨트 건너편/벨트변 blue 13~14k)만 22사이클 내내
        # 골라 non-belt 실큐브(cycle5 red:8348 — 실측 pick 성공 스케일)를 계속 밀어냈습니다. 획득
        # nav와 동일한 벨트색-후순위 선택(_select_acquire_target: color != belt 우선, 그다음 면적)을
        # 재사용합니다 — 벨트색뿐이면 기존대로 그중 최대 blob. 정면 정렬(-angle 회전) 후 push-through로
        # 반경 진입 후 pick. pick_entity가 3D 최근접을 잡으므로 정지 큐브에 붙으면 그걸 집습니다.
        target = _select_acquire_target(stationary, memory.belt_color, None) or max(
            stationary, key=lambda d: d.blob_area
        )
        seen = target.color
        if abs(target.angle_deg) > CENTER_TOLERANCE_DEG:
            await _turn_by_deg(ctx, -target.angle_deg)
        advanced = await _push_through_to_target(
            ctx, seen, arrival_area=CUBE_ARRIVAL_AREA, memory=memory, verbose=True
        )
        result = await pick_nearest_cube(ctx)
        summary = result_summary(result)
        # ★D1b(run6)★ area 도착 게이트는 병합 blob에 속아 1.2m 밖에서 pick을 쏩니다(area 17241
        # '도착' → 실거리 2.60m). too-far 실패 메시지의 실거리는 SDK가 잰 최근접 큐브까지의 정밀
        # range이므로, 부족분(실거리 − 목표 1.0m)만큼 폐루프 재전진 후 1회 재시도합니다. 상한
        # PICK_RETRY_MAX_DIST_M 초과면 쫓던 blob이 큐브가 아닐 공산이 커 전진하지 않고 실패를
        # 반환합니다(LLM이 재관찰/재배치 결정). 재전진에도 D2 벨트 가드·stall 사다리가 걸립니다.
        d_m = _too_far_m(summary.get("error"))
        if d_m is not None and d_m <= PICK_RETRY_MAX_DIST_M:
            extra = max(0.0, d_m - PICK_RETRY_TARGET_M)
            _trace_step(memory, action="pick_too_far", d_m=d_m, advance_m=round(extra, 2))
            print(f"           pick 거리 초과 {d_m:.2f}m -> {extra:.2f}m 재전진 후 1회 재시도")
            advanced += await _push_through_to_target(
                ctx, seen, arrival_area=CUBE_ARRIVAL_AREA, memory=memory, verbose=True,
                max_advance_m=extra, max_chunks=PICK_RETRY_MAX_CHUNKS,
            )
            result = await pick_nearest_cube(ctx)
            summary = result_summary(result)
        # ★E2(run7 마비)★ 여전히 too-far인데 이번 cycle 순전진이 사실상 0이면(벨트 가드가 모든
        # push를 차단) 이 자리·이 각도에선 영원히 못 집습니다 — 제자리 재시도(run7: 22사이클 ×
        # 0.00m)를 recover(후퇴+회전 재배치)로 끊어 다음 cycle이 다른 큐브/각도를 보게 합니다.
        if _too_far_m(summary.get("error")) is not None and advanced < PICK_MIN_CYCLE_ADVANCE_M:
            _trace_step(memory, action="pick_blocked_no_advance", advanced_m=round(advanced, 3))
            memory.pick_blocked_streak += 1
            # ★A3(run9)★ 국소 recover(후퇴+회전)로는 벨트를 못 건넙니다 — run9는 모든 정지 큐브가
            # 벨트 건너 ~2m에 있어 recover ×14로도 520s 내내 0픽. 연속 차단이 쌓이면 navpad의 능동
            # 벨트 우회와 동형으로 free-space 쪽 벨트-따라가기를 실행해 큐브 쪽으로 건너갑니다.
            if PICK_BELT_BYPASS_ENABLED and memory.pick_blocked_streak >= PICK_BYPASS_AFTER_N:
                memory.pick_blocked_streak = 0
                _prof = await _probe_free_space(ctx)
                belt_side = (_prof.get("freer_side") if _prof else None) or 1.0
                _trace_step(
                    memory, action="pick_belt_bypass", side=belt_side, pose=await _get_pose(ctx),
                )
                print(
                    f"           pick 반복 차단 -> 벨트 우회({_side_name(belt_side)}) "
                    f"{BELT_FOLLOW_CHUNKS}청크 따라가기"
                )
                await _lateral_bypass(ctx, belt_side, BELT_FOLLOW_CHUNKS, memory=memory, verbose=True)
                return {"action": "pick_cube", "result": summary, "positioned": False,
                        "seen_color": seen, "recovery": {"reason": "pick_belt_bypass"}}
            print("           pick 차단(전진 0) -> recover로 재배치")
            recovery = await recover_motion(ctx, memory, reason="pick_blocked_no_advance")
            return {"action": "pick_cube", "result": summary, "positioned": False,
                    "seen_color": seen, "recovery": recovery}
        memory.pick_blocked_streak = 0  # 전진이 있었거나 too-far가 아니면 차단 연속 해제.
        return {"action": "pick_cube", "result": summary, "positioned": True, "seen_color": seen}

    if action == "place_cube":
        # 실제 들고 있는 색(ground truth)을 우선해 그 색 pad로 이동/place합니다.
        pad_color = memory.held_color or decision.target_color
        if memory.held_color:
            # 배송: pad는 색블롭이 아니라 VLM signage로 접근합니다(같은 색 source 오배치 방지).
            # A2(자기완결 place): 직전 도착이 유효하면 navpad 재실행을 생략하고 진입 push만 직접
            # 실행합니다 — 진입 push는 원래 navpad 내부 '마무리 접근'(:§1)에 있어 navpad를 생략하면
            # 함께 사라지므로 여기서 동형(belt_guard=False, Fix F)으로 수행합니다. run8에서 place
            # 실패마다 navpad를 재실행해 재확인 VLM 90~144s/사이클을 태우던 핑퐁을 제거합니다.
            letter = DESTINATION_SIGN_RULES.get(memory.held_color)
            pose_now = await _get_pose(ctx)
            if _pad_arrival_fresh(memory.pad_arrived, letter, pose_now, PAD_ARRIVED_STICKY_M):
                _trace_step(memory, action="pad_arrival_reuse", letter=letter, pose=pose_now)
                await _push_through_to_target(
                    ctx, memory.held_color, memory=memory, verbose=True,
                    belt_guard=False, oversize_ok=True,
                )
            else:
                if not await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True):
                    recovery = await recover_motion(ctx, memory, reason="place_positioning_failed")
                    return {"action": "place_cube", "result": None, "positioned": False, "recovery": recovery}
                if letter is not None:
                    p_arr = await _get_pose(ctx)
                    memory.pad_arrived = {"letter": letter, "x": p_arr["x"], "y": p_arr["y"]}
            # place-probe 진입 게이트(셋 다): ⓪ PLACE_PROBE_ENABLED(Option A 봉인 — 동결 goal 전제
            # 경로라 8런 라이브 진입 0회) ① 동결 goal 존재 ② 잔여 d ≤ PLACE_PROBE_START_M(1.35m).
            goal = memory.sign_goals.get(letter) if letter is not None else None
            if PLACE_PROBE_ENABLED and goal is not None and _radius_from(await _get_pose(ctx), goal) <= PLACE_PROBE_START_M:
                probe = await _place_probe(
                    ctx, letter, memory.held_color, goal, memory=memory, verbose=True
                )
                if probe.get("delivered"):
                    return {"action": "place_cube", "result": {"status": "delivered"},
                            "positioned": True, "radius_m": probe.get("radius_m")}
                if probe.get("dropped"):
                    # done-but-dropped: recover_motion을 부르지 않습니다(relocate가 발밑 큐브 회수를 방해).
                    # update_memory가 held=None ground truth로 stage를 복귀 → 다음 cycle pick_cube 재시도.
                    return {"action": "place_cube", "result": None, "positioned": False, "dropped": True}
                # probe+lateral 전체 실패 → 기존 실패 반환과 동형(다음 cycle LLM 재결정).
                recovery = await recover_motion(ctx, memory, reason="place_probe_failed")
                return {"action": "place_cube", "result": None, "positioned": False, "recovery": recovery}
            # 게이트 미충족(동결 전 또는 잔여>1.35m): 아래 공용 단일 place로 fall-through.
        elif not await _target_in_range(ctx, pad_color, PAD_ARRIVAL_AREA):
            reached = await visual_navigate_to_target(ctx, pad_color)
            if not reached:
                recovery = await recover_motion(ctx, memory, reason="place_positioning_failed")
                return {"action": "place_cube", "result": None, "positioned": False, "recovery": recovery}
        result = await place_nearest_zone(ctx)
        place_summary = result_summary(result)
        if place_summary.get("status") == "done":
            memory.pad_arrived = None  # A2: 배치 완료 — 도착 기록 소모(다음 배송은 새로 접근).
        return {"action": "place_cube", "result": place_summary, "positioned": True}

    if action == "recover":
        return await recover_motion(ctx, memory, decision.recovery_strategy)

    if action == "skip_target":
        # 실제 스킵 기록은 update_memory가 처리합니다.
        return {"action": "skip_target", "target_color": decision.target_color, "status": "skipped"}

    return {"action": action, "status": "no_op"}


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

    if tracker is not None:
        tracker.start_first_cycle()
        tracker.print_start()

    # --- 서베이 부트스트랩(런당 1회): pick 전에 개활지에서 pad 표지 지도를 세웁니다. 타이머는 이미
    # 시작됐으므로 서베이도 라운드 시간에 포함됩니다. try/except 필수 — 서베이 실패(VLM 플랩·
    # 네트워크)가 런 전체를 죽이면 안 되고, 빈손 서베이도 navpad가 pick 후 자체 look 계층으로 재획득합니다. ---
    try:
        survey_config = load_config(require_tokamak=True)
        await survey_pads(ctx, memory, survey_config.tokamak_api_key, verbose=True)
    except Exception as exc:
        print(f"[survey] 스킵(부트스트랩 실패, navpad가 자체 재획득): {exc}")

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
    if memory.pad_memory:
        # 발표·디버그용 경로 기억 요약: 색상별 성공 경로 수와 best score(낮을수록 좋음).
        print("Route memory:")
        for pad_color, entry in memory.pad_memory.items():
            best = entry.get("best_route")
            n_ok = len(entry.get("successful_routes", []))
            n_fail = len(entry.get("failed_routes", []))
            if best:
                print(
                    f"  {pad_color}: 성공 {n_ok}회/실패 {n_fail}회,"
                    f" best score={best['score']:.1f}"
                    f" (vlm={best['stats'].get('vlm_calls', 0)},"
                    f" stall={best['stats'].get('stalls', 0)},"
                    f" wp={len(best['waypoints'])})"
                )
            else:
                print(f"  {pad_color}: 성공 0회/실패 {n_fail}회 (best_route 없음)")
    print("Logs:")
    for item in memory.logs:
        print(item)


