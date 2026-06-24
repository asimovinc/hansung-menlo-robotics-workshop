# 프로젝트 안내

## 과제

모든 팀은 동일한 고정 자연어 과제를 받습니다.

> 창고 안의 큐브 6개를 찾아 각 색상에 맞는 목적지 패드로 분류하세요.

과제 문장은 모든 팀과 모든 평가 실행에서 동일합니다. 단, 큐브 색상 순서와 로봇 시작 위치는 달라질 수 있으므로, 각 팀은 코드 수정 없이 동작하는 하나의 일반적인 LLM-assisted robot agent를 만들어야 합니다.

LLM은 실행 중 high-level supervisor로 반드시 사용해야 합니다. LLM은 관찰값, memory, 최근 action outcome을 바탕으로 로봇이 다음에 무엇을 해야 하는지 결정합니다. Perception, localization, navigation, pick/place 실행, safety check는 deterministic code로 처리해도 됩니다.

## 환경

각 실행은 정적인 창고 환경에서 진행됩니다.

- 큐브 6개
- 랜덤 큐브 색상
- 고정된 목적지 패드 위치
- 고정된 목적지 표지판과 배경색
- 추가 레이아웃이 제공되지 않는 한 고정된 장애물 배치
- 고정된 색상-패드 매칭 규칙
- 학생 코드 시작 전에 생성된 (x, y) 위치로 이동된 랜덤 로봇 시작 위치

고정 표지판 정보는 다음과 같습니다.

| 표지판 | 의미 |
| --- | --- |
| A | 컨베이어/큐브 공급 구역이며 목적지 패드가 아닙니다 |
| 빨간 배경의 B | 빨간 큐브 목적지 |
| 초록 배경의 C | 초록 큐브 목적지 |
| 파란 배경의 D | 파란 큐브 목적지 |
| 노란 배경의 E | 노란 큐브 목적지 |

## 사용할 수 있는 정보

제출 에이전트는 다음 정보를 사용할 수 있습니다.

- 카메라 관찰값
- 고정 자연어 과제
- 고정 색상-패드 및 표지판-패드 매칭 규칙
- robot_status, including robot pose and neck state
- 액션 실행 결과
- 프로젝트에서 허용된 SDK 스킬과 헬퍼 함수
- high-level decision을 위한 LLM output

제출 에이전트는 다음 정보를 사용할 수 없습니다.

- raw scene_state
- 객체 또는 장애물의 정답 좌표
- 정확한 큐브 또는 패드 entity ID
- 전체 asset map
- 특정 환경에서만 동작하는 고정 액션 순서

scene_state는 워크숍 학습, 디버깅, TA 평가, 채점 용도로만 사용합니다. 제출 에이전트에서는 사용할 수 없습니다.

## Required LLM Agent Structure

모든 팀은 LLM을 의미 있는 high-level decision-making에 사용해야 합니다. LLM이 직접 low-level velocity command를 출력할 필요는 없습니다. Perception, coordinate estimation, navigation execution, safety check는 deterministic code로 구현해도 됩니다.

에이전트는 다음 loop를 따라야 합니다.

    observe -> LLM decide -> validate -> act -> verify -> update memory -> continue

LLM은 최소한 다음 high-level decision에 사용되어야 합니다.

- 다음 cube target 선택 또는 우선순위 결정
- 다음 high-level action 선택
- search, navigate, pick, place, recover, skip, stop 중 무엇을 할지 결정
- navigation, pick, place 실패 후 다음 행동 결정
- memory를 사용해 같은 실패 행동을 반복하지 않기

LLM은 structured decision object를 반환해야 하며, 학생 코드는 robot action을 실행하기 전에 이 object를 검증해야 합니다.

필수 최소 schema:

    {
       next_action: search_cube,
      target_color: red,
      reason: A red cube is visible and has not been attempted recently.
    }

허용되는 next_action 값:

    search_cube
    navigate_to_cube
    pick_cube
    search_pad
    navigate_to_pad
    place_cube
    recover
    skip_target
    stop

target_color는 색상이 필요 없는 action에서는 null일 수 있습니다. recovery_strategy, retry_limit, memory_update 같은 추가 field를 사용해도 됩니다.

VLM 사용은 선택 사항입니다. VLM은 더 풍부한 scene understanding에 사용할 수 있지만, 필수 AI-agent component는 text LLM decision loop입니다.

예를 들어, 팀은 VLM을 사용해 카메라 프레임에서 고정된 목적지 표지판을 읽거나 확인할 수 있습니다. 초록 배경의 컨베이어 표지판 A와 초록 큐브 목적지 표지판 C를 구분하는 것이 한 예입니다. VLM 출력은 high-level loop의 관찰 근거로 사용해야 하며, structured text-LLM decision이나 action validation을 대체하지 않습니다.

## Level 1: Coordinate-Guided Sorting Agent

목표: 큐브 6개를 모두 올바르게 분류합니다. 일부만 성공해도 올바르게 분류한 큐브 수에 따라 부분 점수를 받을 수 있습니다.

- Perception: 카메라 관찰값에서 큐브와 목적지 패드를 탐지합니다.
- Optional VLM perception: 색상 blob만으로 모호할 때 목적지 표지판의 글자와 배경색을 읽습니다.
- Localization: perception, depth, camera geometry, robot_status를 사용해 목표의 world coordinate를 추정합니다.
- Navigation: 직접 추정한 좌표를 coordinate-based go_to에 전달해 이동합니다.
- LLM decision-making: target 선택, high-level action 선택, recovery step 결정, memory 기반 진행 상태 추적을 수행합니다.
- Recovery: 다시 관찰하고, 부정확한 위치 추정이나 실패한 액션을 수정합니다.
- Main challenge: 시각 관찰값을 충분히 정확한 world coordinate로 변환하면서 LLM을 high-level task supervisor로 사용하는 것입니다.
- Difficulty: Standard.

## Level 2: Vision-Guided Sorting Agent

목표: 큐브 6개를 모두 올바르게 분류합니다. 일부만 성공해도 올바르게 분류한 큐브 수에 따라 부분 점수를 받을 수 있습니다.

- Perception: 카메라 관찰값에서 큐브와 목적지 패드를 탐지하고 추적합니다.
- Optional VLM perception: 색상 blob만으로 모호할 때 목적지 표지판의 글자와 배경색을 읽습니다.
- Navigation: set_head, 카메라 관찰값, set_velocity를 사용해 큐브와 패드까지 closed-loop 방식으로 이동합니다.
- Coordinate navigation: go_to를 호출하면 안 됩니다.
- LLM decision-making: high-level search/navigation/recovery action 선택, memory 관리, retry/skip/stop 판단을 수행합니다.
- Obstacle handling: 가능한 경우 장애물을 감지하고 우회하거나 목표를 다시 찾습니다.
- Recovery: target loss, overshoot, blocked movement, failed action을 처리합니다.
- Main challenge: 안정적인 vision-only navigation을 구현하면서 LLM을 high-level task supervisor로 사용하는 것입니다.
- Difficulty: Advanced.

Closed-loop navigation은 다음 흐름을 따릅니다.

    observe -> move briefly -> observe again -> correct or stop

## 스타터 코드

학생들은 다음 모듈의 프로젝트용 헬퍼 함수를 사용하거나 수정해서 사용할 수 있습니다.

- menlo_runner.perception
- menlo_runner.navigation
- menlo_runner.llm

정확히 어떤 도구와 함수가 허용되는지는 docs/project_allowed_tools.ko.md를 확인하세요.

중요 제한 사항:

- 좌표를 받는 헬퍼 함수는 학생 시스템이 직접 추정한 좌표에만 사용할 수 있습니다.
- my_go_to_global은 scene_state와 정확한 entity ID를 사용하므로 제출 에이전트에서 사용할 수 없습니다.
- 기본 WorkshopAgent는 학습 예제이며, 기본 도구가 scene_state와 정확한 entity ID를 사용하므로 그대로 제출용 에이전트로 사용할 수 없습니다.
- 시작 시 한 번만 LLM을 호출하는 것은 충분하지 않습니다. LLM은 task execution 중 decision loop에 참여해야 합니다.

## 평가 방식

### Practice

개발 중에는 랜덤 큐브 색상 순서와 랜덤 로봇 시작 위치로 여러 번 테스트할 수 있습니다.

### Interim Evaluation

중간 평가는 TA가 선택한 하나의 hidden cube-color order와 하나의 hidden robot starting position으로 진행합니다.

- 고정 과제 문장은 모든 팀에게 동일합니다.
- 같은 레벨의 모든 팀은 동일한 중간 평가 조건에서 실행합니다.
- 평가 실행 중에는 소스코드를 수정할 수 없습니다.
- 팀은 평가 결과와 피드백을 바탕으로 이후 시스템을 개선할 수 있습니다.

### Final Evaluation

최종 평가는 중간 평가와 다른 hidden cube-color order와 hidden robot starting position으로 진행합니다.

- 고정 과제 문장은 모든 팀에게 동일합니다.
- 같은 레벨의 모든 팀은 동일한 최종 평가 조건에서 실행합니다.
- 최종 평가 조건은 중간 평가 조건과 다릅니다.
- 평가 실행 중에는 소스코드를 수정할 수 없습니다.
- 최종 결과는 심사에 사용됩니다.

## 공통 요구사항

모든 팀은 다음을 만족해야 합니다.

- 고정 자연어 과제를 입력으로 받습니다.
- 네 개의 워크숍 개념을 활용합니다.
- LLM-assisted observe-decide-act loop를 구현합니다.
- structured LLM decision을 사용하고 action 실행 전 검증합니다.
- 현재 관찰값에서 목표 정보를 도출합니다.
- 액션 결과, 로봇 상태, 카메라 관찰값으로 결과를 검증합니다.
- 실패 상황에서 적절히 복구합니다.
- observation, LLM decision, executed action, outcome을 기록합니다.
- 접근 방식, 결과, 한계점을 설명합니다.

## 평가 기준

### 1. Task Performance

- 올바르게 분류한 큐브 수
- 잘못 배치한 큐브 수

### 2. LLM Agent Behavior

- valid structured LLM decision 사용
- observation, memory, action outcome, recovery reasoning의 의미 있는 활용
- observation -> LLM decision -> action -> result 로그
- LLM decision이 실제 high-level action sequence에 영향을 주었다는 증거

### 3. Reliability

- TA가 선택한 평가 조건에서의 성능
- 실패한 액션에서 복구하는 능력
- 평가 실행 사이에 코드 수정 없이 동작하는 능력

### 4. Engineering and Presentation

- 워크숍 개념의 효과적인 활용
- 코드 품질과 시스템 설계
- 결과와 한계점에 대한 명확한 시연 및 설명
