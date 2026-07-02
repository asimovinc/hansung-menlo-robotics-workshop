# 프로젝트 안내

## 과제

모든 팀은 같은 자연어 과제를 받습니다.

> source area에서 cube를 찾아 matching destination pad로 분류하세요.

팀은 cube 색상과 robot 시작 위치가 바뀌어도 source code 변경 없이 동작하는 하나의
LLM-assisted robot agent를 만들어야 합니다.

LLM은 실행 내내 high-level task supervisor 역할을 합니다. 관찰, memory, 최근 action
결과를 바탕으로 다음에 무엇을 해야 하는지 결정합니다. Low-level perception,
navigation, manipulation, validation, safety는 deterministic code가 처리해야 합니다.

## 환경

각 평가 run은 다음을 포함하는 고정 warehouse 환경에서 진행됩니다.

- Conveyor/cube source area에서 제공되는 cube
- Randomized cube colors
- Fixed destination pad locations
- Fixed destination signage and color backgrounds
- Fixed obstacle layout
- Fixed color-to-pad matching rules
- Randomized robot starting position

고정 signage는 다음과 같습니다.

| Sign | 의미 |
| --- | --- |
| A | Conveyor/cube source area, destination pad가 아님 |
| B with red background | Red cube destination |
| C with green background | Green cube destination |
| D with blue background | Blue cube destination |
| E with yellow background | Yellow cube destination |

Run 시작 시 cube queue는 conveyor/cube source area에 있습니다. 팀은 어떤 cube를 먼저
집을지 선택할 수 있지만, 보통 첫 available cube를 집는 전략이 가장 단순합니다. Cube를
성공적으로 집으면 다음 cube가 같은 pickup area에 나타날 수 있습니다.

## 프로젝트 레벨

팀은 세 가지 project level 중 하나를 선택할 수 있습니다. 선택한 level은 delivery 점수에
영향을 줍니다.

### Level 0: Full-State Agent

`scene_state`를 통해 complete environment information을 사용할 수 있습니다.

사용 가능:

- `scene_state`
- `cube_2`, `pad_C` 같은 entity ID
- `go_to` entity-target navigation
- Camera observations, optional

기대 사항:

- LLM task planning
- High-level decision-making
- Entity target `go_to` navigation
- Pick and place execution
- Failed action recovery

Perception이나 localization은 필수 요구사항이 아닙니다. 핵심은 complete state를 사용하되
고정 script가 아니라 LLM-driven task planner를 설계하는 것입니다.

### Level 1: Adaptive Navigation Agent

`scene_state`는 사용할 수 없습니다.

학생 시스템은 camera observation으로 cube와 destination pad를 찾아야 합니다.

기대 사항:

- Visual target detection
- Target navigation
- `set_velocity`를 사용한 manual approach
- 학생 시스템이 관찰로 추정하거나 성공 후 기록한 coordinate에만 coordinate-based `go_to` 사용
- Memory를 사용해 이후 navigation 개선

핵심은 perception, memory, coordinate estimation, navigation, LLM reasoning을 결합해
성능을 개선하는 것입니다.

### Level 2: Autonomous Vision Agent

`scene_state`와 coordinate-based `go_to`는 사용할 수 없습니다.

기대 사항:

- Camera observation으로 cube와 destination pad detect/track
- `set_head`, `set_velocity`, closed-loop visual feedback으로 navigation
- Obstacle avoidance
- Failed navigation, target loss, failed manipulation recovery
- LLM high-level planning and decision-making

핵심은 coordinate navigation 없이 vision-based navigation system을 만드는 것입니다.

권장 closed-loop pattern:

```text
observe -> move briefly -> observe again -> correct or stop
```

## 허용 정보

모든 project agent가 사용할 수 있는 정보:

- Camera observations
- Natural-language task
- Fixed color-to-pad and sign-to-pad matching rules
- `robot_status`, including robot pose and neck state
- Action results
- Project-allowed SDK skills and helper functions
- High-level decision-making을 위한 LLM outputs

개발과 평가에는 `MENLO_API_KEY`와 `TOKAMAK_API_KEY`가 모두 필요합니다. `MENLO_API_KEY`는
robot platform 연결에 사용됩니다. `TOKAMAK_API_KEY`는 text LLM decision loop와 optional
VLM call에 필요합니다.

기본적으로 `menlo_runner.llm.call_llm(...)`은 `minimaxai/minimax-m3`를 사용합니다.
팀은 package source code를 직접 수정하지 않고도 승인된 다른 모델을 선택할 수 있습니다.

```python
import os

os.environ["MENLO_LLM_MODEL"] = "minimaxai/minimax-m3"
os.environ["MENLO_VLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# 두 변수에 사용할 수 있는 승인된 다른 선택지:
# os.environ["MENLO_LLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# os.environ["MENLO_VLM_MODEL"] = "minimaxai/minimax-m3"
```

Notebook 사용자는 setup cell 실행 후 agent를 시작하기 전에 이 값을 설정하세요. Local IDE
사용자는 `.env`에 `MENLO_LLM_MODEL`과 `MENLO_VLM_MODEL`을 설정하거나 `call_llm(...)` /
`ask_vlm(...)`에 `model=...`을 직접 넘길 수 있습니다.

Level별 추가 허용 정보:

| Data source or capability | Level 0 | Level 1 | Level 2 |
| --- | --- | --- | --- |
| `scene_state` | 허용 | 금지 | 금지 |
| Scene의 정확한 entity ID | 허용 | 금지 | 금지 |
| Entity target `go_to` | 허용 | 금지 | 금지 |
| 학생이 추정한 world pose 기반 `go_to` | 허용 | 허용 | 금지 |
| `set_velocity` | 허용 | 허용 | 허용 |
| `set_head` | 허용 | 허용 | 허용 |
| Camera observations | 허용 | 필수 | 필수 |
| Text LLM decision loop | 필수 | 필수 | 필수 |
| VLM observations | 선택 | 선택 | 선택 |

Level 1과 Level 2는 target 정보를 camera observations와 level별 허용 input에서 얻어야 합니다.
Raw `scene_state`, ground-truth object coordinates, exact cube/pad entity IDs, global asset map은
사용할 수 없습니다.

한 가지 start pose나 한 가지 cube-color setup에서만 동작하는 fixed action sequence는 모든
level에서 허용되지 않습니다.

## 필수 LLM Agent 구조

모든 팀은 LLM-assisted decision loop를 구현해야 합니다. LLM은 low-level robot command를
생성하는 대신 meaningful high-level reasoning을 해야 합니다.

필수 execution loop:

```text
observe -> decide -> validate -> act -> verify -> update memory -> continue
```

LLM decision 예시:

- 다음 cube 선택
- Target priority 결정
- 다음 high-level action 선택
- Failed navigation/pick/place 이후 recovery 결정
- Retry, skip, stop 결정
- Future decision 개선을 위한 memory 사용

Student code는 LLM response를 실행하기 전에 반드시 validate해야 합니다.

최소 response schema:

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

`target_color`는 color가 필요 없는 action에서는 `null`일 수 있습니다. `retry_limit`,
`memory_update`, `recovery_strategy` 같은 추가 field도 사용할 수 있습니다.

VLM 사용은 선택입니다. Destination sign을 camera frame에서 읽거나 scene understanding을
보강하는 데 사용할 수 있습니다. 하지만 필수 AI-agent component는 structured text-LLM
decision loop입니다.

처음에 LLM을 한 번만 호출하는 것은 충분하지 않습니다. LLM은 target selection, recovery,
skipping, stopping 등 실행 중 의사결정에 계속 참여해야 합니다.

## Starter Code와 Helpers

학생은 다음 helper를 사용하거나 수정할 수 있습니다.

- `menlo_runner.scene`
- `menlo_runner.basics`
- `menlo_runner.perception`
- `menlo_runner.navigation`
- `menlo_runner.llm`
- `menlo_runner.agents`

Project starter는 Python file과 notebook 양쪽에 있습니다.

- English notebooks: `notebooks/project/en/`
- Korean notebooks: `notebooks/project/ko/`
- English Python starters: `menlo_runner/programs/project/en/`
- Korean Python starters: `menlo_runner/programs/project/ko/`

Starter run path와 completion wrapper는 라운드별 제한 시간에 맞춰 scored simulation을 실행할 수 있습니다.

## 평가 기준

평가는 라운드별 큐브 이동 성과, 코드 구현, 발표 점수를 합산합니다. 각 라운드는 성공적으로 이동한 큐브를 최대 12개까지만 인정합니다. 라운드 타이머는 엄격하게 적용됩니다. 코드가 LLM/VLM/API 응답이나 robot action 결과를 기다리는 중에 시간이 끝나도 즉시 run을 중단하고, 현재까지 이동한 큐브 개수와 delivery score를 출력합니다. 로봇이 넘어져서 재시작 없이 복구가 불가능한 경우에는 run을 중단하고 남은 시간을 기록하며, 해당 라운드의 남은 시간 안에서 재시작할 수 있습니다.

| 항목 | 평가 내용 | 점수 |
| --- | --- | --- |
| 1. 라운드별 작업 수행 | 라운드별 상한 12개 내 성공 큐브 이동 개수 | 레벨별 산정 |
| 2. 코드 구현 | LLM 판단 루프 구현 및 하드코딩 확인 | 최대 10점 |
| 3. 발표 | 중간 발표와 최종 발표 | 최대 30점 |

### 라운드 시간

| 라운드 | 제한 시간 |
| --- | --- |
| 라운드 1 | 5분 |
| 라운드 2 | 10분 |
| 라운드 3 | 15분 |

같은 라운드와 같은 setup option 번호를 사용하면 모든 level에서 시작 위치와 cube color order가 동일합니다. Level별로 달라지는 것은 scoring formula뿐입니다.

### 1. 라운드별 작업 수행

각 라운드는 성공적으로 이동한 큐브 개수로 평가하며, 라운드별 최대 인정 개수는 12개입니다. 잘못된 위치에 놓은 경우에는 benchmark 규칙에 따라 감점되거나 run이 종료될 수 있습니다.

| 프로젝트 레벨 | 큐브 이동 점수 |
| --- | --- |
| Level 0: Full-State Agent | 성공적으로 이동한 큐브 1개당 5점 |
| Level 1: Adaptive Navigation Agent | 첫 큐브 이동 성공 시 60점, 이후 성공적으로 이동한 큐브 1개당 20점 |
| Level 2: Autonomous Vision Agent | 첫 큐브 이동 성공 시 60점, 이후 성공적으로 이동한 큐브 1개당 40점 |

### 2. 코드 구현: 10점

Judge는 필수 LLM 판단 루프가 구현되어 있는지, 고정 hard-coded script가 아닌지, 제출 코드와 실행 behavior를 바탕으로 최대 10점을 부여합니다.

평가 요소:

- 선택한 level의 허용 정보 규칙 준수
- 고정 action sequence가 아닌 일반화 가능한 strategy
- 의미 있는 LLM decision loop와 structured output validation
- observation, decision, action, verification, memory의 명확한 분리
- navigation, pick, place 실패 후 recovery behavior
- 읽기 쉬운 코드 구조와 유용한 실행 로그

### 3. 발표: 30점

팀은 project code를 실행하여 robot behavior를 시연해야 합니다. Presentation slide는 line-by-line 구현 설명보다 핵심 설계와 결과 요약에 집중하세요.

| 발표 | 점수 |
| --- | --- |
| 중간 발표 | 10점 |
| 최종 발표 | 20점 |

중간 발표:

- 구현한 robot action flow
- LLM의 역할
- 현재 성공 사례와 한계
- 개선 계획

최종 발표:

- 완성된 robot action flow
- LLM의 역할
- 중간 발표 이후 개선점과 남은 한계
- 실제 AI-agent robotics로 확장할 수 있는 방향

## 제출 전 확인

- 선택한 level의 금지 정보와 금지 API를 사용하지 않았는지 확인하세요.
- Text LLM decision loop가 실행 중 반복적으로 사용되는지 확인하세요.
- LLM response validation이 있는지 확인하세요.
- Action 후 verification과 memory update가 있는지 확인하세요.
- Starter 또는 completion wrapper의 라운드별 scored simulation run에서 delivery score가 출력되는지 확인하세요.

