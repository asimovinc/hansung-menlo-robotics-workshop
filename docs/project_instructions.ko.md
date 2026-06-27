# 프로젝트 안내

## 과제

모든 팀은 같은 자연어 과제를 받습니다.

> 창고에 있는 여섯 개의 큐브를 찾아 각 색상에 맞는 목적지 패드로 분류하세요.

팀은 큐브 색상 순서와 로봇 시작 위치가 바뀌어도 소스 코드를 수정하지 않고 동작하는 하나의 일반적인 LLM 보조 로봇 에이전트를 만들어야 합니다.

LLM은 실행 중 고수준 작업 감독자 역할을 합니다. 관찰값, 메모리, 최근 행동 결과를 바탕으로 로봇이 다음에 무엇을 해야 하는지 결정합니다. 저수준 perception, navigation, manipulation, validation, safety 처리는 deterministic code가 담당할 수 있습니다.

## 환경

각 평가는 고정된 창고 환경에서 진행됩니다.

- 컨베이어/큐브 공급 구역에 제시되는 큐브 6개
- 무작위 큐브 색상
- 고정된 목적지 패드 위치
- 고정된 목적지 표지판과 배경색
- 고정된 장애물 배치
- 고정된 색상-패드 매칭 규칙
- 무작위 로봇 시작 위치

고정 표지판 정보는 다음과 같습니다.

| 표지판 | 의미 |
| --- | --- |
| A | 컨베이어/큐브 공급 구역이며 목적지 패드가 아님 |
| 빨간 배경의 B | 빨간 큐브 목적지 |
| 초록 배경의 C | 초록 큐브 목적지 |
| 파란 배경의 D | 파란 큐브 목적지 |
| 노란 배경의 E | 노란 큐브 목적지 |

각 실행 시작 시 큐브는 컨베이어/큐브 공급 구역에 있습니다. 어떤 큐브를 먼저 집을지는 자유롭게 선택할 수 있지만, 일반적으로 처음 접근 가능한 큐브를 집는 전략이 가장 단순합니다. 큐브를 성공적으로 집으면 다음 큐브가 같은 pickup 구역에 제시될 수 있습니다.

## 프로젝트 레벨

팀은 세 가지 레벨 중 하나를 선택할 수 있습니다. 선택한 레벨은 전체 평가 점수에 반영됩니다.

### Level 0: Full-State Agent

`scene_state`를 통해 환경의 전체 상태 정보를 사용할 수 있습니다.

사용 가능 정보와 기능:

- `scene_state`
- `cube_2`, `pad_C` 같은 entity ID
- entity target을 사용하는 `go_to`
- 카메라 관찰값, 선택 사항

학생들이 집중해야 할 내용:

- LLM task planning
- 고수준 의사결정
- entity target 기반 `go_to` navigation
- pick/place 실행
- 실패한 행동에서의 recovery

Perception이나 localization은 요구되지 않습니다.

주요 과제: 완전한 환경 정보를 사용하되, 고정 스크립트가 아니라 LLM이 의미 있게 다음 행동을 결정하는 task planner를 설계하는 것입니다.

관련 helper는 `menlo_runner.scene`, `menlo_runner.basics.go_to_entity`, Workshop 4의 `WorkshopAgent` 구조입니다. raw SDK call을 사용할 때 entity navigation은 다음 형태입니다.

```python
result = await ctx.invoke(
    "go_to",
    {"target": {"kind": "entity", "entity_id": "pad_C"}},
    timeout_s=300,
)
```

### Level 1: Adaptive Navigation Agent

`scene_state`는 사용할 수 없습니다.

카메라 관찰값으로 큐브와 목적지 패드를 찾아야 합니다. 학생들은 다음을 수행해야 합니다.

- 시각적으로 target detection 수행
- target 방향으로 navigation 수행
- `set_velocity`로 수동 이동하며 pick/place가 성공할 때까지 접근
- pick/place 성공을 검증한 뒤 학생 시스템이 기록한 좌표에 한해 coordinate-based `go_to` 사용
- 필요하다면 memory를 사용해 이후 navigation 개선

주요 과제: perception, memory, visual approach, 기록된 좌표 기반 navigation, LLM reasoning을 결합해 성능을 점진적으로 개선하는 것입니다.

현재 Level 1 starter는 `menlo_runner/programs/project/en/level_1_starter.py`와 한국어 버전 `menlo_runner/programs/project/ko/level_1_starter_ko.py`입니다.

### Level 2: Autonomous Vision Agent

`scene_state`와 coordinate-based `go_to`를 사용할 수 없습니다.

카메라 관찰값과 수동 로봇 제어만으로 navigation해야 합니다. 학생들은 다음을 수행해야 합니다.

- 큐브와 목적지 패드 detect/track
- `set_head`, `set_velocity`, closed-loop visual feedback으로 navigation
- 장애물 회피
- navigation 실패, target loss, manipulation 실패에서 recovery
- LLM을 고수준 planning과 decision-making에 사용

주요 과제: coordinate navigation 없이 vision-based navigation으로 과제를 완료하는 자율 에이전트를 구현하는 것입니다.

현재 Level 2 starter는 `menlo_runner/programs/project/en/level_2_starter.py`와 한국어 버전 `menlo_runner/programs/project/ko/level_2_starter_ko.py`입니다.

Closed-loop navigation은 다음 흐름을 따르는 것이 좋습니다.

```text
observe -> move briefly -> observe again -> correct or stop
```

## 허용 정보

모든 제출 에이전트는 다음 정보를 사용할 수 있습니다.

- 카메라 관찰값
- 자연어 과제
- 고정 색상-패드 및 표지판-패드 매칭 규칙
- 로봇 pose, status, neck state를 포함한 `robot_status`
- action result
- 프로젝트에서 허용된 SDK skill과 helper function
- 고수준 의사결정을 위한 LLM output

모든 project agent는 개발과 평가에서 `MENLO_API_KEY`와 `TOKAMAK_API_KEY`가 모두 필요합니다. `MENLO_API_KEY`는 robot platform 연결에 사용합니다. `TOKAMAK_API_KEY`는 text LLM decision loop에 필요하며, 선택적으로 VLM을 사용할 때도 필요합니다.

레벨별 추가 허용 범위:

| 정보 또는 기능 | Level 0 | Level 1 | Level 2 |
| --- | --- | --- | --- |
| `scene_state` | 허용 | 불가 | 불가 |
| scene에서 얻은 정확한 entity ID | 허용 | 불가 | 불가 |
| entity target 기반 `go_to` | 허용 | 불가 | 불가 |
| 검증된 성공 후 학생이 기록한 world pose 기반 `go_to` | 허용 | 허용 | 불가 |
| `set_velocity` | 허용 | 허용 | 허용 |
| `set_head` | 허용 | 허용 | 허용 |
| 카메라 관찰값 | 허용 | 필수 | 필수 |
| Text LLM decision loop | 필수 | 필수 | 필수 |
| VLM 관찰 | 선택 | 선택 | 선택 |

Level 1과 Level 2는 카메라 관찰값과 기타 허용 입력으로 target 정보를 도출해야 합니다. raw `scene_state`, 정답 object coordinate, 정확한 cube/pad entity ID, global asset map은 사용할 수 없습니다.

특정 시작 위치나 특정 큐브 색상 구성에서만 동작하는 고정 action sequence는 어떤 레벨에서도 허용되지 않습니다.

## 필수 LLM 에이전트 구조

모든 팀은 LLM-assisted decision loop를 구현해야 합니다. LLM은 저수준 로봇 명령을 직접 생성하는 것이 아니라 의미 있는 고수준 reasoning을 수행해야 합니다.

필수 실행 loop:

```text
observe -> decide -> validate -> act -> verify -> update memory -> continue
```

LLM은 다음과 같은 결정을 내려야 합니다.

- 다음 큐브 선택
- target 우선순위 결정
- 다음 high-level action 선택
- navigation, pick, place 실패 후 recovery 행동 결정
- retry, skip, stop 여부 결정
- memory를 사용한 이후 decision 개선
- final evaluation의 hidden natural-language instruction 해석

학생 코드는 LLM 응답을 실행하기 전에 반드시 검증해야 합니다.

최소 응답 schema:

```json
{
  "next_action": "search_cube",
  "target_color": "red",
  "reason": "A red cube is visible and has not been attempted recently."
}
```

허용되는 `next_action` 값:

```text
search_cube
navigate_to_cube
pick_cube
search_pad
navigate_to_pad
place_cube
recover
skip_target
stop
```

색상이 필요 없는 action에서는 `target_color`가 `null`일 수 있습니다. `retry_limit`, `memory_update`, `priority_colors`, `delivery_limit`, `recovery_strategy` 같은 추가 field를 사용할 수 있습니다.

VLM 사용은 선택 사항입니다. 목적지 표지판 읽기처럼 더 풍부한 scene understanding에 사용할 수 있습니다. 다만 필수 AI-agent 구성요소는 structured text-LLM decision loop입니다.

처음에 LLM을 한 번만 호출하는 것은 충분하지 않습니다. target selection, recovery, skip, stop, hidden instruction adaptation 등 실행 중 decision loop에 LLM이 참여해야 합니다.

## Hidden Task Adaptation

제출 에이전트는 기본 목표만 hard-code하지 말고 자연어 과제를 입력으로 받아야 합니다.

Final evaluation 직전에 공개 과제를 수정하는 추가 hidden natural-language instruction이 제공될 수 있습니다. 가능한 변형 유형은 다음과 같습니다.

- 지정된 개수의 큐브만 delivery
- 하나 이상의 큐브 색상 우선 처리

정확한 instruction은 final evaluation 전까지 공개되지 않습니다. 평가 중 소스 코드 수정은 허용되지 않으므로, LLM prompt, decision schema, validation, memory가 이런 변형을 처리할 수 있게 설계해야 합니다.

유용한 memory field 예:

- `delivered_count`
- `delivery_limit`
- `priority_colors`
- `held_color`
- `completed_colors`
- `failed_attempts`
- `recent_outcomes`

## Starter Code와 Helper

학생들은 다음 module의 helper를 사용하거나 수정할 수 있습니다.

- `menlo_runner.scene`
- `menlo_runner.basics`
- `menlo_runner.perception`
- `menlo_runner.navigation`
- `menlo_runner.llm`
- `menlo_runner.agents`

자주 쓰는 SDK call:

```python
jpeg = await ctx.get_vision("pov")
status = await ctx.state("robot_status")

await ctx.invoke("set_head", {"yaw": 0.5, "pitch": 0.2}, timeout_s=10)

await ctx.invoke(
    "set_velocity",
    {"vx": 0.25, "vy": 0.0, "wz": 0.0, "duration_s": 1.0},
    timeout_s=30,
)

await ctx.invoke("cancel", {})
```

Pick/place action:

```python
pick_result = await ctx.invoke(
    "pick_entity",
    {"target": {"kind": "entity", "entity_id": "cube"}},
    timeout_s=300,
)

place_result = await ctx.invoke("place_entity", {}, timeout_s=300)
```

Level 0 entity navigation:

```python
result = await ctx.invoke(
    "go_to",
    {"target": {"kind": "entity", "entity_id": "pad_C"}},
    timeout_s=300,
)
```

Level 1 recorded-coordinate navigation:

```python
result = await ctx.invoke(
    "go_to",
    {
        "target": {
            "kind": "pose",
            "pose": {"frame_id": "world", "position": [x, y, 0]},
        }
    },
    timeout_s=300,
)
```

`set_velocity` parameter:

- `vx`: 전진 속도, m/s
- `vy`: 왼쪽 방향 속도, m/s
- `wz`: yaw rate, rad/s
- `duration_s`: 명령 지속 시간, 초

명령은 policy range로 clip됩니다: `|vx|, |vy| <= 1.5`, `|wz| <= 0.6`. 새 `set_velocity` 명령은 실행 중인 `go_to` 또는 `set_velocity` 명령을 대체합니다.

중요 제한:

- Level 0은 `scene_state`, entity ID, entity-target `go_to`를 사용할 수 있습니다.
- Level 1은 pick/place 성공을 검증한 뒤 학생 시스템이 기록한 좌표에 대해서만 coordinate `go_to`를 사용할 수 있습니다.
- Level 2는 `go_to`를 호출하면 안 됩니다.
- `my_go_to_global`은 `scene_state`와 정확한 entity ID를 사용하므로 Level 0에서만 사용할 수 있습니다.
- 기본 `WorkshopAgent`는 학습 예제입니다. Level 0에서는 수정해 사용할 수 있지만, 기본 tool들이 `scene_state`와 정확한 entity ID를 사용하므로 Level 1 또는 Level 2 제출용으로는 그대로 사용할 수 없습니다.

## 평가 방식

### Practice

개발 중에는 무작위 cube-color order와 무작위 robot starting position으로 테스트할 수 있습니다.

### Interim Evaluation

Interim evaluation은 다음 조건으로 진행됩니다.

- hidden cube-color configuration 1개
- hidden robot starting position 1개

공개 과제는 동일합니다.

> Find and sort the six cubes into their matching destination pads.

같은 project level의 모든 팀은 같은 hidden setup으로 평가됩니다. 평가 중 소스 코드 수정은 허용되지 않습니다. 팀은 결과와 피드백을 바탕으로 final evaluation 전에 시스템을 개선할 수 있습니다.

### Final Evaluation

Final evaluation은 다음 조건으로 진행됩니다.

- interim과 다른 hidden cube-color configuration
- interim과 다른 hidden robot starting position
- 선택적으로 hidden natural-language instruction 1개

공개 과제는 동일합니다.

> Find and sort the six cubes into their matching destination pads.

평가 중 소스 코드 수정은 허용되지 않습니다. Final 결과가 judging에 사용됩니다.

## 공통 요구사항

모든 팀은 다음을 만족해야 합니다.

- 자연어 과제를 입력으로 받기
- 네 번의 workshop 개념 사용
- 필수 LLM-assisted decision loop 구현
- structured LLM output 생성 및 실행 전 validation
- robot status, action result, camera observation으로 실행 결과 검증
- observation, LLM decision, executed action, outcome log 유지
- navigation, pick, place 실패에서 적절히 recovery
- 설계 결정과 구현 접근 방식 설명
- 현재 접근 방식의 한계 논의

## 평가 기준

총점은 100점입니다.

| 범주 | 평가 지표 | 최대 점수 |
| --- | --- | --- |
| 1. Task Performance | 올바르게 분류한 큐브 수와 delivery당 평균 LLM decision cycle 수 | 40 |
| 2. Project Level | 선택한 project level의 성공적 수행 | 30 |
| 3. Hidden Task Adaptation | final hidden natural-language instruction 대응 성능 | 20 |
| 4. Engineering and Presentation | 시스템 설계, 구현 결정, 논의 | 10 |
| 합계 |  | 100 |

### 1. Task Performance: 40점

Task Performance는 두 요소로 평가됩니다.

#### Task Completion: 30점

평가 시간 안에 올바르게 분류한 큐브 수에 따라 점수를 받습니다.

| 올바르게 분류한 큐브 수 | 점수 |
| --- | --- |
| 0 | 0 |
| 1 | 5 |
| 2 | 10 |
| 3 | 15 |
| 4 | 20 |
| 5 | 25 |
| 6 | 30 |

잘못된 placement는 benchmark 규칙에 따라 감점되거나 run을 종료시킬 수 있습니다.

#### LLM Efficiency: 10점

성공적으로 delivery한 큐브 1개당 평균 LLM decision cycle 수로 평가합니다. 평균 cycle 수가 적을수록 높은 점수를 받습니다.

### 2. Project Level: 30점

팀은 하나의 project level을 선택합니다.

| Project level | 최대 점수 |
| --- | --- |
| Level 0: Full-State Agent | 10 |
| Level 1: Adaptive Navigation Agent | 20 |
| Level 2: Autonomous Vision Agent | 30 |

선택한 level의 제한 조건을 지키면서 큐브를 최소 3개 이상 성공적으로 delivery한 팀은 해당 level의 최대 점수까지 받을 수 있습니다.

### 3. Hidden Task Adaptation: 20점

소스 코드 수정 없이 final hidden natural-language modification을 올바르게 해석하고 수행하는 능력을 평가합니다.

가능한 변형 유형:

- 지정된 개수의 큐브만 delivery
- 하나 이상의 큐브 색상 우선 처리

### 4. Engineering and Presentation: 10점

팀은 다음을 명확히 발표해야 합니다.

- 전체 시스템 아키텍처
- 주요 설계 결정
- 구현 highlight
- 시스템 안에서 LLM의 역할
- validation 및 recovery 전략
- 현재 접근 방식의 한계
- simulated environment를 넘어 실제 AI-agentic robotics에 적용할 수 있는 방법
