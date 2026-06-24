# Project Allowed Tools and Helper Functions

Use this page while building your project agent. It lists SDK calls and helper functions that are safe to use in submitted project code.

Submitted agents must derive targets from camera observations, the fixed natural-language task, the fixed color/sign matching rules, `robot_status`, and structured LLM decisions. Do not use `scene_state`, ground-truth coordinates, exact entity IDs, or the global asset map.

Fixed destination signage is allowed information:

| Sign | Meaning |
| --- | --- |
| A | Conveyor/cube source area, not a destination pad |
| B with red background | Red cube destination |
| C with green background | Green cube destination |
| D with blue background | Blue cube destination |
| E with yellow background | Yellow cube destination |

## Allowed Tools by Level

| Tool or data source | Level 1 | Level 2 | Use for |
| --- | --- | --- | --- |
| `go_to` with world pose target | Allowed | Not allowed | Coordinate-guided navigation |
| `set_velocity` | Allowed | Allowed | Short movement commands |
| `cancel` | Allowed | Allowed | Stop an active action |
| nearest-cube pick | Allowed | Allowed | Pick after visual positioning |
| nearest-zone place | Allowed | Allowed | Place after visual positioning |
| `set_head` | Allowed | Allowed | Camera scanning and aiming |
| `ctx.get_vision("pov")` | Allowed | Allowed | Camera observations |
| `ctx.state("robot_status")` | Allowed | Allowed | Robot pose, status, and neck state |
| `ctx.state("scene_state")` | Not allowed | Not allowed | Debug/workshop only |
| Text LLM decision call | Required | Required | High-level agent decisions |
| VLM call | Optional | Optional | Extra scene understanding, including sign-letter reading |

## SDK Skills

### Coordinate `go_to`

Level 1 may use `go_to` only with a world-coordinate pose estimated by the student system.

```python
result = await session.invoke("go_to", {
    "target": {
        "kind": "pose",
        "pose": {
            "frame_id": "world",
            "position": [x, y, 0]
        }
    }
}, timeout_s=300)
```

Level 2 must not call `go_to`.

### `set_velocity`

Use short velocity commands, then re-observe before deciding the next move.

```python
result = await session.invoke("set_velocity", {
    "vx": 0.25,
    "vy": 0.0,
    "wz": 0.0,
    "duration_s": 1.0
})
```

Parameters:

- `vx`: forward velocity in m/s
- `vy`: left velocity in m/s
- `wz`: yaw rate in rad/s
- `duration_s`: command duration in seconds

Commands are clipped to the trained policy ranges: `|vx|, |vy| <= 1.5`, `|wz| <= 0.6`. A new `set_velocity` command preempts any active `go_to` or `set_velocity` command.

Recommended loop:

```text
observe -> move briefly -> observe again -> correct or stop
```

### `cancel`

```python
result = await session.invoke("cancel", {})
```

Use this to stop an active runtime action, such as an in-flight `go_to`.

### Nearest-Cube Pick

Pick only after the robot has visually navigated close to the intended cube. The local Workshop 1 helper defaults to nearest-cube picking:

```python
from menlo_runner.basics import pick_entity

result = await pick_entity(ctx)
```

The robot must be within reach, roughly 1 m, and its hands must be empty. If multiple cubes are close together, nearest-cube picking may grab the wrong cube.

### Nearest-Zone Place

Place only after the robot has visually navigated close to the intended matching pad.

```python
result = await session.invoke("place_entity", {}, timeout_s=300)
```

The robot must be holding a cube. A wrong-color destination can terminate the sorting benchmark, so verify before and after placing.

### `set_head`

Aim the head/neck independently of locomotion. This changes the POV camera direction without changing the walking policy.

```python
result = await session.invoke("set_head", {
    "yaw": 0.5,
    "pitch": 0.2
})
```

- `yaw`: pan left/right in radians, positive is left
- `pitch`: tilt up/down in radians, positive looks down

Either field may be omitted. Use `robot_status` to read the measured neck state.

## Robot Status and Camera Access

Camera frame:

```python
jpeg = await ctx.get_vision("pov")
```

Robot pose, status, and neck state:

```python
state = await ctx.state("robot_status")

position = state.robot.pose.position
heading_deg = state.robot.pose.yaw_deg
status = state.robot.status

head_target = state.robot.extra["head"]["target"]
head_measured = state.robot.extra["head"]["measured"]
yaw_range = state.robot.extra["head"]["yaw_range"]
pitch_range = state.robot.extra["head"]["pitch_range"]
```

Use neck state with camera geometry when estimating target direction or world coordinates.

## Reusable Helpers

### Perception

```python
from menlo_runner.perception import (
    perceive,
    perceive_jpeg,
    detect_color_blobs,
    annotate_detections,
    estimate_depth_map,
)
```

- `perceive(ctx)`: captures the POV camera and returns `{color: {angle_deg, blob_area}}`.
- `detect_color_blobs(jpeg_bytes)`: returns detections with centroid, bounding box, angle, and blob area.
- `perceive_jpeg(jpeg_bytes)`: runs the Workshop 2 perception format on existing JPEG bytes.
- `annotate_detections(jpeg_bytes)`: creates a debug image with detections drawn on it.
- `estimate_depth_map(jpeg_bytes, depth_pipe)`: runs a depth-estimation pipeline on a camera frame.

### Navigation

```python
from menlo_runner.navigation import (
    angle_error_deg,
    turn_to_face,
    drive_to_distance,
    center_on_color,
    drive_toward_color,
    my_go_to_visual,
)
```

- `angle_error_deg(robot_xy, yaw_deg, target_xy)`: heading error to a coordinate.
- `turn_to_face(ctx, target_pos)`: turns toward a coordinate using `robot_status` and `set_velocity`.
- `drive_to_distance(ctx, target_pos)`: drives toward a coordinate using `robot_status` and `set_velocity`.
- `center_on_color(ctx, target_color)`: turns until a color is centered in the camera.
- `drive_toward_color(ctx, target_color)`: approaches a visible colored target.
- `my_go_to_visual(ctx, target_color)`: baseline visual navigation to a cube of a target color.

Coordinate-based helpers may be used only with coordinates estimated by the student system. `my_go_to_global` is not project-safe because it uses `scene_state` and exact entity IDs.

The visual navigation helpers are baselines. They may fail when obstacles block the path or the target leaves the camera view, so improving them is part of the project.

### Required Text LLM Decision

All submissions must use a text LLM for high-level agent decisions. The LLM should choose task-level actions, not raw velocity commands.

Recommended decision schema:

```json
{
  "next_action": "search_cube",
  "target_color": "red",
  "reason": "Red is highest priority and has not been completed."
}
```

Allowed `next_action` values:

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

Student code must validate the LLM decision before executing robot actions. `target_color` may be `null` when irrelevant. Additional fields such as `recovery_strategy`, `retry_limit`, and `memory_update` are allowed.

A single LLM call at startup is not enough. The LLM must participate during the observe-decide-act loop, especially for target choice, recovery, skip, and stop decisions.

### LLM and VLM

```python
from menlo_runner.llm import (
    call_llm,
    ask_vlm,
    parse_tool_call,
    build_system_prompt,
)
```

- `call_llm(messages, api_key=...)`: calls the text LLM.
- `ask_vlm(jpeg_bytes, prompt, api_key=...)`: asks a VLM about a camera frame. It may be used to read or verify fixed destination signage from observations.
- `parse_tool_call(text)`: parses a JSON tool call from an LLM response.
- `build_system_prompt(tools)`: builds a tool-use system prompt.

Example VLM sign-reading prompt:

```python
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
```

Use VLM output as observation evidence for the text-LLM decision loop and validation checks. Do not use VLMs to bypass the required structured decision object, and do not combine VLM output with `scene_state`, exact entity IDs, or ground-truth coordinates.

The default `WorkshopAgent` is a learning example, not a submission-ready project agent, because its default tools use `scene_state` and exact entity IDs. You may adapt its structure with project-safe tools.

## Result Checking

Workbook examples use `result.status`. Workshop 4 also checks `result.error.message` when a call fails. Use a defensive pattern:

```python
result = await session.invoke("set_velocity", {"vx": 0.25, "duration_s": 1.0})
print(result.status)

if getattr(result, "error", None):
    print(result.error.message)
```

After important actions, verify with robot status and camera observations rather than relying only on `result.status`.


