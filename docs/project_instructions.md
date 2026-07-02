# Project Instructions

## Task

All teams receive the same natural-language task:

> Find and sort cubes from the source area into their matching destination pads.

Teams must build one general LLM-assisted robot agent that can complete the task under randomized cube colors and randomized robot starting positions without source-code changes.

The LLM serves as the high-level task supervisor throughout execution. It decides what the robot should do next based on observations, memory, and recent action outcomes. Deterministic code should handle low-level perception, navigation, manipulation, validation, and safety.

## Environment

Each evaluation uses a static warehouse environment with:

- Cubes presented from the conveyor/cube source area
- Randomized cube colors
- Fixed destination pad locations
- Fixed destination signage and color backgrounds
- Fixed obstacle layout
- Fixed color-to-pad matching rules
- Randomized robot starting position

The fixed destination signage is:

| Sign | Meaning |
| --- | --- |
| A | Conveyor/cube source area, not a destination pad |
| B with red background | Red cube destination |
| C with green background | Green cube destination |
| D with blue background | Blue cube destination |
| E with yellow background | Yellow cube destination |

At the start of each run, the cube queue is available at the conveyor/cube source area. Teams may choose any available cube to pick first, although picking the first available cube is usually the simplest strategy. After a cube is successfully picked, the system may present the next cube at the same pickup area.

## Project Levels

Teams may attempt one of three project levels. The selected project level contributes to the overall evaluation score.

### Level 0: Full-State Agent

Complete environment information is available through `scene_state`.

Students may use:

- `scene_state`
- Entity IDs such as `cube_2` or `pad_C`
- Entity-target navigation with `go_to`
- Camera observations, optionally

Students are expected to focus on:

- LLM task planning
- High-level decision-making
- Navigation using `go_to` with entity targets
- Pick and place execution
- Recovery from failed actions

No perception or localization is required.

Main challenge: design an LLM-driven task planner that solves the task using complete environment information.

Relevant codebase helpers include `menlo_runner.scene`, `menlo_runner.basics.go_to_entity`, and the Workshop 4 `WorkshopAgent` structure. When using raw SDK calls, entity navigation has this shape:

```python
result = await ctx.invoke(
    "go_to",
    {"target": {"kind": "entity", "entity_id": "pad_C"}},
    timeout_s=300,
)
```

### Level 1: Adaptive Navigation Agent

`scene_state` is not available.

Students must detect cubes and destination pads using camera observations. They are expected to:

- Detect targets visually
- Navigate toward targets
- Use manual robot movement with `set_velocity` to approach until pick/place succeeds
- Use coordinate-based `go_to` only after suitable coordinates have been estimated or recorded by the student system
- Optionally use memory to improve future navigation

Main challenge: combine perception, memory, coordinate estimation, navigation, and LLM reasoning to improve performance over time.

The current Level 1 starter is `menlo_runner/programs/project/en/level_1_starter.py`.

### Level 2: Autonomous Vision Agent

`scene_state` is not available. Coordinate-based `go_to` is not permitted.

Students must rely on camera observations and manual robot control for navigation. They are expected to:

- Detect and track cubes and destination pads
- Navigate using `set_head`, `set_velocity`, and closed-loop visual feedback
- Avoid obstacles
- Recover from failed navigation, target loss, and failed manipulation
- Use the LLM for high-level planning and decision-making

Main challenge: develop a fully autonomous vision-based navigation system capable of completing the task without coordinate-based navigation.

The current Level 2 starter is `menlo_runner/programs/project/en/level_2_starter.py`.

Closed-loop navigation should follow:

```text
observe -> move briefly -> observe again -> correct or stop
```

## Allowed Information

All submitted agents may use:

- Camera observations
- The natural-language task
- The fixed color-to-pad and sign-to-pad matching rules
- `robot_status`, including robot pose and neck state
- Action results
- Project-allowed SDK skills and helper functions
- LLM outputs for high-level decision-making

All project agents require both `MENLO_API_KEY` and `TOKAMAK_API_KEY` during development and evaluation. `MENLO_API_KEY` connects to the robot platform. `TOKAMAK_API_KEY` is required for the text LLM decision loop, and also for optional VLM calls.

By default, `menlo_runner.llm.call_llm(...)` uses `minimaxai/minimax-m3`. Teams may choose another approved model without editing package source code:

```python
import os

os.environ["MENLO_LLM_MODEL"] = "minimaxai/minimax-m3"
os.environ["MENLO_VLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# Approved alternatives for either variable:
# os.environ["MENLO_LLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# os.environ["MENLO_VLM_MODEL"] = "minimaxai/minimax-m3"
```

For notebook users, put this in a cell after setup and before the agent starts. Local IDE users may set `MENLO_LLM_MODEL` and `MENLO_VLM_MODEL` in `.env` or pass `model=...` directly to `call_llm(...)` / `ask_vlm(...)`.

Additional information by level:

| Data source or capability | Level 0 | Level 1 | Level 2 |
| --- | --- | --- | --- |
| `scene_state` | Allowed | Not allowed | Not allowed |
| Exact entity IDs from the scene | Allowed | Not allowed | Not allowed |
| `go_to` with entity target | Allowed | Not allowed | Not allowed |
| `go_to` with student-estimated world pose | Allowed | Allowed | Not allowed |
| `set_velocity` | Allowed | Allowed | Allowed |
| `set_head` | Allowed | Allowed | Allowed |
| Camera observations | Allowed | Required | Required |
| Text LLM decision loop | Required | Required | Required |
| VLM observations | Optional | Optional | Optional |

Levels 1 and 2 must derive target information from camera observations and other allowed inputs. They must not use raw `scene_state`, ground-truth object coordinates, exact cube or pad entity IDs, or a global asset map.

Fixed action sequences that only work for one start pose or one cube-color setup are not allowed at any level.

## Required LLM Agent Structure

All teams must implement an LLM-assisted decision loop. The LLM must perform meaningful high-level reasoning rather than generating low-level robot commands.

The required execution loop is:

```text
observe -> decide -> validate -> act -> verify -> update memory -> continue
```

The LLM should make decisions such as:

- Selecting the next cube
- Prioritizing targets
- Choosing the next high-level action
- Determining recovery behavior after failed navigation, pick, or place actions
- Deciding whether to retry, skip, or stop
- Using memory to improve future decisions

Student code must validate every LLM response before execution.

Minimum response schema:

```json
{
  "next_action": "search_cube",
  "target_color": "red",
  "reason": "A red cube is visible and has not been attempted recently."
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

`target_color` may be `null` when the action does not need a color. Additional fields such as `retry_limit`, `memory_update`, or `recovery_strategy` are permitted.

VLM usage is optional. Teams may use VLMs for richer scene understanding, including reading destination signs from camera frames. The required AI-agent component is still the structured text-LLM decision loop.

A single LLM call at the beginning is not enough. The LLM must participate during task execution, especially for target selection, recovery, skipping, and stopping.

## Natural-Language Task Input

The submitted agent must accept the natural-language task as input rather than hard-coding only the default objective.

Examples of useful memory fields include:

- `delivered_count`
- `held_color`
- `completed_colors`
- `failed_attempts`
- `recent_outcomes`

## Starter Code and Helpers

Students may use or adapt helpers from:

- `menlo_runner.scene`
- `menlo_runner.basics`
- `menlo_runner.perception`
- `menlo_runner.navigation`
- `menlo_runner.llm`
- `menlo_runner.agents`

Common SDK calls:

```python
# Camera frame and robot status
jpeg = await ctx.get_vision("pov")
status = await ctx.state("robot_status")

# Head aiming
await ctx.invoke("set_head", {"yaw": 0.5, "pitch": 0.2}, timeout_s=10)

# Short manual movement
await ctx.invoke(
    "set_velocity",
    {"vx": 0.25, "vy": 0.0, "wz": 0.0, "duration_s": 1.0},
    timeout_s=30,
)

# Cancel an active runtime action
await ctx.invoke("cancel", {})
```

Pick and place actions:

```python
# Pick the nearest intended cube after the robot has been positioned close to it.
pick_result = await ctx.invoke(
    "pick_entity",
    {"target": {"kind": "entity", "entity_id": "cube"}},
    timeout_s=300,
)

# Place on the nearest zone after the robot has reached the matching pad.
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

Level 1 coordinate navigation:

```python
result = await ctx.invoke(
    "go_to",
    {
        "target": {
            "kind": "pose",
            "pose": {
                "frame_id": "world",
                "position": [x, y, 0],
            },
        }
    },
    timeout_s=300,
)
```

The `set_velocity` parameters are:

- `vx`: forward velocity in m/s
- `vy`: left velocity in m/s
- `wz`: yaw rate in rad/s
- `duration_s`: command duration in seconds

Commands are clipped to the trained policy ranges: `|vx|, |vy| <= 1.5`, `|wz| <= 0.6`. A new `set_velocity` command preempts any active `go_to` or `set_velocity` command.

Project-relevant helper functions:

- `menlo_runner.perception.detect_color_blobs(jpeg_bytes)`: detect colored blobs with centroid, bounding box, angle, and area.
- `menlo_runner.perception.perceive(ctx)`: capture the POV camera and return a compact color observation.
- `menlo_runner.perception.annotate_detections(jpeg_bytes)`: create a debug image with detections drawn on it.
- `menlo_runner.perception.estimate_depth_map(jpeg_bytes, depth_pipe)`: optional depth-estimation hook.
- `menlo_runner.navigation.center_on_color(ctx, target_color)`: baseline visual centering.
- `menlo_runner.navigation.drive_toward_color(ctx, target_color)`: baseline visual approach.
- `menlo_runner.navigation.my_go_to_visual(ctx, target_color)`: baseline visual navigation to a colored target.
- `menlo_runner.llm.call_llm(...)`: text LLM call for structured decisions.
- `menlo_runner.llm.ask_vlm(...)`: optional VLM call for scene or sign understanding.
- `menlo_runner.llm.parse_tool_call(...)`: parse JSON-like tool calls from model output.
- `menlo_runner.scene.visible_cubes(ctx)`: Level 0 helper for visible cube IDs, colors, and positions.
- `menlo_runner.scene.held_cube_info(ctx)`: Level 0 helper for the held cube.
- `menlo_runner.scene.delivered_cube_ids(ctx)`: Level 0 helper for delivered cube IDs.

Important restrictions:

- Level 0 may use `scene_state`, entity IDs, and entity-target `go_to`.
- Level 1 may use coordinate `go_to` only with coordinates estimated or recorded by the student system.
- Level 2 must not call `go_to`.
- `my_go_to_global` uses `scene_state` and exact entity IDs, so it is Level 0 only.
- The default `WorkshopAgent` is a learning example. It may be adapted for Level 0, but its default tools are not valid for Levels 1 or 2 because they use `scene_state` and exact entity IDs.

## Evaluation Setup

### Practice

Teams may develop and test using randomized cube-color orders and randomized robot starting positions.

### Interim Evaluation

The interim evaluation uses:

- One hidden cube-color configuration
- One hidden robot starting position

The published task remains:

> Find and sort cubes from the source area into their matching destination pads.

All teams within the same project level are evaluated using the same hidden setup.

No source-code changes are permitted during evaluation. Teams may use the results and feedback to improve their systems before the final evaluation.

### Final Evaluation

The final evaluation uses:

- A different hidden cube-color configuration
- A different hidden robot starting position

The published task remains:

> Find and sort cubes from the source area into their matching destination pads.

The final setup is different from the interim setup. No source-code changes are permitted during evaluation. Final results are used for judging.

## Common Requirements

All teams must:

- Accept the natural-language task as input.
- Use concepts from all four workshops.
- Implement the required LLM-assisted decision loop.
- Produce structured LLM outputs and validate them before execution.
- Verify execution outcomes using robot status, action results, and camera observations.
- Maintain execution logs containing observations, LLM decisions, executed actions, and outcomes.
- Recover appropriately from failed navigation, pick, and place actions.
- Demonstrate the robot behavior by running the project code during the interim and final presentations.
- Use presentation slides to summarize the required interim and final presentation topics.

## Evaluation Criteria

Evaluation is based on round-by-round cube delivery, code implementation, and presentations. Each delivery round counts at most 12 successful cube deliveries. The round timer is strict: if time expires while code is waiting for an LLM/VLM/API result or robot action result, the run stops immediately and the current delivered-cube count and delivery score are printed. If the robot falls and cannot recover without restarting, the run stops, the remaining time is recorded, and the team may restart within the remaining time for that round.

| Category | Evaluation metric | Points |
| --- | --- | --- |
| 1. Round Delivery | Successful cube deliveries within each round, capped at 12 cubes per round | Level-based |
| 2. Code Implementation | LLM decision loop implementation and hard-coding check | Up to 10 |
| 3. Presentation | Interim and final presentation scores | Up to 30 |

### Round Timing

| Round | Time limit |
| --- | --- |
| Round 1 | 5 minutes |
| Round 2 | 10 minutes |
| Round 3 | 15 minutes |

All levels use the same round setup for a given round and setup option number. The start pose and cube color order are shared; only the scoring formula changes by level.

### 1. Round Delivery

Each round is scored by the number of successfully moved cubes, up to 12 cubes per round. Incorrect placements may reduce credit or terminate the run depending on the benchmark rules.

| Project level | Delivery score |
| --- | --- |
| Level 0: Full-State Agent | 5 points per successfully moved cube |
| Level 1: Adaptive Navigation Agent | 60 points for the first successful cube move, then 20 points per additional successfully moved cube |
| Level 2: Autonomous Vision Agent | 60 points for the first successful cube move, then 40 points per additional successfully moved cube |

### 2. Code Implementation: 10 Points

Judges award up to 10 points for the required LLM judgment loop implementation and for confirming that the solution is not a hard-coded fixed script.

Evaluation elements:

- Compliance with the selected level's allowed-information rules
- General strategy rather than a fixed hard-coded sequence
- Meaningful LLM decision loop and structured output validation
- Clear observation, decision, action, verification, and memory separation
- Recovery behavior after failed navigation, pick, or place actions
- Readable code structure and useful execution logs

### 3. Presentation: 30 Points

Teams should demonstrate their robot behavior by running the project code. Presentation slides should focus on summary topics rather than detailed line-by-line implementation walkthroughs.

| Presentation | Points |
| --- | --- |
| Interim presentation | 10 |
| Final presentation | 20 |

Interim presentation:

- Implemented robot action flow
- Role of the LLM
- Current successes and limitations
- Improvement plan

Final presentation:

- Complete robot action flow
- Role of the LLM
- Improvements since the interim presentation and limitations of the current system
- How the solution could be used or extended for real-world AI-agent-based robotics
