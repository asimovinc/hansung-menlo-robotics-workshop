# 프로젝트 허용 도구 및 헬퍼 함수

프로젝트 에이전트를 만들 때 참고하는 문서입니다. 제출 코드에서 사용할 수 있는 SDK 호출과 헬퍼 함수를 정리합니다.

제출 에이전트는 카메라 관찰값, 고정 자연어 과제, 고정 색상/표지판 매칭 규칙, robot_status, structured LLM decision을 바탕으로 목표를 찾아야 합니다. scene_state, 정답 좌표, 정확한 entity ID, global asset map은 사용할 수 없습니다.

고정 목적지 표지판 정보는 사용할 수 있습니다.

| 표지판 | 의미 |
| --- | --- |
| A | 컨베이어/큐브 공급 구역이며 목적지 패드가 아닙니다 |
| 빨간 배경의 B | 빨간 큐브 목적지 |
| 초록 배경의 C | 초록 큐브 목적지 |
| 파란 배경의 D | 파란 큐브 목적지 |
| 노란 배경의 E | 노란 큐브 목적지 |

## 레벨별 허용 도구

| 도구 또는 데이터 | Level 1 | Level 2 | 용도 |
| --- | --- | --- | --- |
| world pose target을 사용하는 go_to | 허용 | 불가 | 좌표 기반 이동 |
| set_velocity | 허용 | 허용 | 짧은 이동 명령 |
| cancel | 허용 | 허용 | 실행 중인 액션 중지 |
| nearest-cube pick | 허용 | 허용 | 시각적으로 위치를 맞춘 뒤 큐브 집기 |
| nearest-zone place | 허용 | 허용 | 시각적으로 위치를 맞춘 뒤 패드에 놓기 |
| set_head | 허용 | 허용 | 카메라 스캔 및 조준 |
| ctx.get_vision("pov") | 허용 | 허용 | 카메라 관찰 |
| ctx.state("robot_status") | 허용 | 허용 | 로봇 위치, 상태, 목 상태 |
| ctx.state("scene_state") | 불가 | 불가 | 디버깅/워크숍 전용 |
| Text LLM decision call | 필수 | 필수 | High-level agent decision |
| VLM call | 선택 | 선택 | Extra scene understanding, 표지판 글자 읽기 |

## SDK 스킬

### Coordinate go_to

Level 1은 학생 시스템이 직접 추정한 world-coordinate pose로만 go_to를 사용할 수 있습니다.

    result = await session.invoke("go_to", {
        "target": {
            "kind": "pose",
            "pose": {
                "frame_id": "world",
                "position": [x, y, 0]
            }
        }
    }, timeout_s=300)

Level 2는 go_to를 호출하면 안 됩니다.

### set_velocity

짧은 속도 명령을 실행한 뒤 다시 관찰하고 다음 행동을 결정하세요.

    result = await session.invoke("set_velocity", {
        "vx": 0.25,
        "vy": 0.0,
        "wz": 0.0,
        "duration_s": 1.0
    })

파라미터:

- vx: 전진 속도, m/s
- vy: 왼쪽 방향 속도, m/s
- wz: yaw 회전 속도, rad/s
- duration_s: 명령 실행 시간, 초

명령은 policy range로 clip됩니다: |vx|, |vy| <= 1.5, |wz| <= 0.6. 새로운 set_velocity 명령은 실행 중인 go_to 또는 set_velocity를 중단하고 대체합니다.

권장 루프:

    observe -> move briefly -> observe again -> correct or stop

### cancel

    result = await session.invoke("cancel", {})

실행 중인 액션을 중지할 때 사용합니다.

### Nearest-Cube Pick

로봇이 의도한 큐브 가까이 시각적으로 이동한 뒤에만 pick을 호출하세요. 로컬 Workshop 1 헬퍼는 기본적으로 nearest-cube pick을 사용합니다.

    from menlo_runner.basics import pick_entity

    result = await pick_entity(ctx)

로봇은 큐브에 충분히 가까워야 하며, 손이 비어 있어야 합니다. 여러 큐브가 가까이 있으면 의도하지 않은 큐브를 집을 수 있습니다.

### Nearest-Zone Place

로봇이 의도한 matching pad 가까이 시각적으로 이동한 뒤에만 place를 호출하세요.

    result = await session.invoke("place_entity", {}, timeout_s=300)

로봇은 큐브를 들고 있어야 합니다. 잘못된 색상의 목적지에 놓으면 sorting benchmark가 종료될 수 있으므로, 놓기 전후로 검증하세요.

### set_head

로봇의 보행 방향과 독립적으로 head/neck을 조준합니다. POV 카메라 방향을 바꾸지만 walking policy는 바꾸지 않습니다.

    result = await session.invoke("set_head", {
        "yaw": 0.5,
        "pitch": 0.2
    })

- yaw: 좌우 pan, rad 단위, 양수는 왼쪽
- pitch: 상하 tilt, rad 단위, 양수는 아래쪽

둘 중 하나만 전달할 수도 있습니다. 실제 측정된 목 상태는 robot_status에서 확인하세요.

## Robot Status와 Camera Access

카메라 프레임:

    jpeg = await ctx.get_vision("pov")

로봇 위치, 상태, 목 상태:

    state = await ctx.state("robot_status")

    position = state.robot.pose.position
    heading_deg = state.robot.pose.yaw_deg
    status = state.robot.status

    head_target = state.robot.extra["head"]["target"]
    head_measured = state.robot.extra["head"]["measured"]
    yaw_range = state.robot.extra["head"]["yaw_range"]
    pitch_range = state.robot.extra["head"]["pitch_range"]

카메라 geometry로 목표 방향이나 world coordinate를 추정할 때 neck state를 함께 사용하세요.

## 재사용 가능한 헬퍼 함수

### Perception

    from menlo_runner.perception import (
        perceive,
        perceive_jpeg,
        detect_color_blobs,
        annotate_detections,
        estimate_depth_map,
    )

- perceive(ctx): POV 카메라를 캡처하고 {color: {angle_deg, blob_area}} 형식으로 반환합니다.
- detect_color_blobs(jpeg_bytes): centroid, bounding box, angle, blob area가 포함된 detection을 반환합니다.
- perceive_jpeg(jpeg_bytes): 기존 JPEG bytes에 Workshop 2 perception 형식을 적용합니다.
- annotate_detections(jpeg_bytes): detection이 그려진 디버그 이미지를 생성합니다.
- estimate_depth_map(jpeg_bytes, depth_pipe): 카메라 프레임에 depth-estimation pipeline을 실행합니다.

### Navigation

    from menlo_runner.navigation import (
        angle_error_deg,
        turn_to_face,
        drive_to_distance,
        center_on_color,
        drive_toward_color,
        my_go_to_visual,
    )

- angle_error_deg(robot_xy, yaw_deg, target_xy): 특정 좌표까지의 heading error를 계산합니다.
- turn_to_face(ctx, target_pos): robot_status와 set_velocity로 좌표 방향을 향해 회전합니다.
- drive_to_distance(ctx, target_pos): robot_status와 set_velocity로 좌표를 향해 이동합니다.
- center_on_color(ctx, target_color): 특정 색상이 카메라 중앙에 오도록 회전합니다.
- drive_toward_color(ctx, target_color): 보이는 색상 목표를 향해 접근합니다.
- my_go_to_visual(ctx, target_color): 특정 색상 큐브로 이동하는 기본 vision navigation 예제입니다.

좌표 기반 헬퍼는 학생 시스템이 직접 추정한 좌표에만 사용할 수 있습니다. my_go_to_global은 scene_state와 정확한 entity ID를 사용하므로 프로젝트 제출 코드에서는 사용할 수 없습니다.

Vision navigation 헬퍼는 baseline입니다. 장애물이 경로를 막거나 target이 카메라에서 사라지면 실패할 수 있으며, 이를 개선하는 것이 프로젝트의 일부입니다.

### Required Text LLM Decision

모든 제출물은 high-level agent decision을 위해 text LLM을 사용해야 합니다. LLM은 raw velocity command가 아니라 task-level action을 선택해야 합니다.

권장 decision schema:

    {
      "next_action": "search_cube",
      "target_color": "red",
      "reason": "Red is highest priority and has not been completed."
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

학생 코드는 robot action을 실행하기 전에 LLM decision을 검증해야 합니다. target_color는 필요하지 않을 때 null일 수 있습니다. recovery_strategy, retry_limit, memory_update 같은 추가 field를 사용할 수 있습니다.

시작 시 한 번만 LLM을 호출하는 것은 충분하지 않습니다. LLM은 observe-decide-act loop 중 target choice, recovery, skip, stop decision에 참여해야 합니다.

### LLM and VLM

    from menlo_runner.llm import (
        call_llm,
        ask_vlm,
        parse_tool_call,
        build_system_prompt,
    )

- call_llm(messages, api_key=...): text LLM을 호출합니다.
- ask_vlm(jpeg_bytes, prompt, api_key=...): 카메라 프레임에 대해 VLM에 질문합니다. 카메라 관찰에서 고정 목적지 표지판을 읽거나 확인하는 데 사용할 수 있습니다.
- parse_tool_call(text): LLM 응답에서 JSON tool call을 파싱합니다.
- build_system_prompt(tools): tool-use system prompt를 만듭니다.

표지판 읽기 VLM prompt 예시:

    jpeg = await ctx.get_vision("pov")
    reply = ask_vlm(
        jpeg,
        (
            "Read the floating warehouse signs visible in this robot camera frame. "
            "A is the conveyor/cube source area and is not a destination. "
            "Destination signs are B red, C green, D blue, E yellow. "
            "Return JSON with visible sign letters, colors, rough positions, and confidence."
        ),
        api_key=tokamak_api_key,
    )

VLM 출력은 text LLM decision loop와 validation check에 사용할 관찰 근거로 다루세요. VLM으로 required structured decision object를 대체하거나, scene_state, 정확한 entity ID, 정답 좌표와 결합해 사용하면 안 됩니다.

기본 WorkshopAgent는 학습 예제이며 제출용으로 그대로 사용할 수 없습니다. 기본 도구가 scene_state와 정확한 entity ID를 사용하기 때문입니다. 구조만 참고하여 프로젝트에서 허용된 도구로 수정해 사용하세요.

## 결과 확인

워크북 예제는 SDK 호출 후 result.status를 사용합니다. Workshop 4는 실패 시 result.error.message도 확인합니다. 안전하게 다음 패턴을 사용하세요.

    result = await session.invoke("set_velocity", {"vx": 0.25, "duration_s": 1.0})
    print(result.status)

    if getattr(result, "error", None):
        print(result.error.message)

중요한 액션 후에는 result.status만 믿지 말고, robot status와 camera observation으로 다시 확인하세요.
