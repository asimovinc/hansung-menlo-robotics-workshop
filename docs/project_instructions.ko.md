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

Starter run path는 기본적으로 10분 scored simulation을 실행합니다.

## Evaluation Criteria

최종 점수에는 100점 상한이 없습니다. Cube delivery 점수는 10분 simulation run 안에서 획득하며,
code quality와 presentation 점수를 더합니다.

| Category | Evaluation metric | Points |
| --- | --- | --- |
| 1. Cube Delivery | 10분 simulation run 안의 successful deliveries | Level-based, no maximum |
| 2. Code Structure and Quality | Source code와 runtime behavior에 대한 judge review | Up to 10 |
| 3. Presentation | Theory, design decisions, robot behavior, reflection | Up to 10 |

### 1. Cube Delivery

10분 simulation run 안에 완료한 successful cube delivery마다 선택한 project level에 따라 점수를
받습니다. 점수로 인정되는 cube delivery 개수에는 상한이 없습니다.

| Project level | Points per successful delivery |
| --- | --- |
| Level 0: Full-State Agent | 10 |
| Level 1: Adaptive Navigation Agent | 20 |
| Level 2: Autonomous Vision Agent | 30 |

Incorrect placement는 benchmark rules에 따라 감점되거나 run 종료 사유가 될 수 있습니다.

### 2. Code Structure and Quality: 10 Points

Judge는 제출 source code와 observed runtime behavior를 바탕으로 최대 10점을 부여합니다.

평가 요소:

- Level rules 준수
- Hard-coded fixed script가 아닌 general strategy
- Correct LLM usage and structured output validation
- Clear observation, decision, action, verification, memory separation
- Recovery behavior
- Readable code structure and logs

### 3. Presentation: 10 Points

팀은 project code를 실행해 robot behavior를 시연해야 합니다. Presentation slide는 세부 구현
line-by-line 설명보다 다음 요약 주제에 집중하세요.

Interim presentation:

- Implemented robot action flow
- Role of the LLM
- Current successes and limitations
- Improvement plan

Final presentation:

- Full robot action flow
- Role of the LLM
- Improvements and remaining limitations
- How this would apply to real AI-agent robotics

## 제출 전 확인

- 선택한 level의 금지 정보와 금지 API를 사용하지 않았는지 확인하세요.
- Text LLM decision loop가 실행 중 반복적으로 사용되는지 확인하세요.
- LLM response validation이 있는지 확인하세요.
- Action 후 verification과 memory update가 있는지 확인하세요.
- Starter의 10분 scored simulation run에서 delivery score가 출력되는지 확인하세요.
