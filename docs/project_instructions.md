# Project Instructions

## Task

All teams receive the same fixed natural-language task:

> Find and sort the six cubes in the warehouse into their matching destination pads.

The task is fixed across teams and evaluation runs. Teams must still build one general LLM-assisted robot agent that works with randomized cube colors and randomized robot starting positions without source-code changes.

The LLM is required as a high-level supervisor during execution. It should decide what the robot should do next from observations, memory, and recent action outcomes. The robot code should still handle low-level perception, localization, navigation, pick/place execution, and safety checks.

## Environment

Each run uses a static warehouse environment with:

- Six cubes
- Randomized cube colors
- Fixed destination-pad locations
- Fixed obstacle layout, unless additional layouts become available
- Fixed color-to-pad matching rules
- Randomized robot starting position, generated as an `(x, y)` position before student code starts

## Allowed Information

Submitted agents may use:

- Camera observations
- The fixed natural-language task
- `robot_status`, including robot pose and neck state
- Action results
- Project-allowed SDK skills and helper functions
- LLM outputs for high-level decision-making

Submitted agents may not use:

- Raw `scene_state`
- Ground-truth object or obstacle coordinates
- Exact cube or pad entity IDs
- Global asset map
- Fixed action sequences that only work for one setup

`scene_state` is for workshop learning, debugging, instructor-side evaluation, and scoring only. It must not be used in submitted agents.

## Required LLM Agent Structure

All teams must use an LLM for meaningful high-level decision-making. The LLM should not directly output low-level velocity commands. Low-level perception, coordinate estimation, navigation execution, and safety checks may be deterministic.

The agent must follow this loop:

```text
observe -> LLM decide -> validate -> act -> verify -> update memory -> continue
```

The LLM must be used for major high-level decisions, such as:

- selecting or prioritizing the next cube target
- choosing the next high-level action
- deciding whether to search, navigate, pick, place, recover, skip, or stop
- deciding what to do after failed navigation, pick, or place actions
- using memory to avoid repeatedly trying the same failed action

The LLM must return a structured decision object. Student code must validate the object before executing any robot action.

Required minimum schema:

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

`target_color` may be `null` when the action does not need a color. Extra fields such as `recovery_strategy`, `retry_limit`, or `memory_update` are allowed.

VLM use is optional. Teams may use VLMs for richer scene understanding, but the required AI-agent component is the text LLM decision loop.

## Level 1: Coordinate-Guided Sorting Agent

Completion target: correctly sort all six cubes. Partial success is rewarded based on the number of correctly sorted cubes.

- Perception: Detect cubes and destination pads from camera observations.
- Localization: Estimate target world coordinates using perception, depth, camera geometry, and `robot_status`.
- Navigation: Use coordinate-based `go_to` with visually estimated target coordinates.
- LLM decision-making: Select targets, choose high-level actions, decide recovery steps, and track progress using memory.
- Recovery: Re-observe and correct inaccurate localization or failed actions.
- Main challenge: Convert visual observations into sufficiently accurate world coordinates while using the LLM as the high-level task supervisor.
- Difficulty: Standard.

## Level 2: Vision-Guided Sorting Agent

Completion target: correctly sort all six cubes. Partial success is rewarded based on the number of correctly sorted cubes.

- Perception: Detect and track cubes and destination pads from camera observations.
- Navigation: Use closed-loop camera observations, `set_head`, and `set_velocity` to navigate to both cubes and pads.
- Coordinate navigation: Do not invoke `go_to`.
- LLM decision-making: Choose high-level search/navigation/recovery actions, maintain memory, and determine when to retry, skip, or stop.
- Obstacle handling: Detect obstacles, select detours, and re-identify targets where possible.
- Recovery: Handle target loss, overshoot, blocked movement, and failed actions.
- Main challenge: Implement reliable vision-only navigation while using the LLM as the high-level task supervisor.
- Difficulty: Advanced.

Closed-loop navigation should follow:

```text
observe -> move briefly -> observe again -> correct or stop
```

## Starter Code

Students may use or adapt project-safe helpers from:

- `menlo_runner.perception`
- `menlo_runner.navigation`
- `menlo_runner.llm`

See `docs/project_allowed_tools.md` for the exact allowed tools, helper functions, and usage examples.

Important restrictions:

- Coordinate helpers may use only coordinates estimated by the student system.
- `my_go_to_global` is not allowed in submitted agents because it uses `scene_state` and exact entity IDs.
- The default `WorkshopAgent` is a learning example, not a submission-ready project agent, because its default tools use `scene_state` and exact entity IDs.
- A single LLM call at the beginning is not enough. The LLM must participate in the decision loop during task execution.

## Evaluation Setup

### Practice

During development, students may test with randomly generated cube-color orders and randomized robot starting positions.

### Interim Evaluation

The interim evaluation uses one hidden cube-color order and one hidden robot starting position selected by the instructors.

- The fixed task is the same for all teams.
- The same hidden interim setup is used for all teams within the same level.
- No source-code changes are allowed during the evaluation run.
- Teams may use the results and feedback to improve their systems afterward.

### Final Evaluation

The final evaluation uses one separate hidden cube-color order and one separate hidden robot starting position selected by the instructors.

- The fixed task is the same for all teams.
- The same hidden final setup is used for all teams within the same level.
- The final setup is different from the interim setup.
- No source-code changes are allowed during the evaluation run.
- Final results are used for judging.

## Common Requirements

All teams must:

- Accept the fixed natural-language task as input.
- Use concepts from all four workshops.
- Implement an LLM-assisted observe-decide-act loop.
- Use structured LLM decisions and validate them before acting.
- Derive target information from current observations.
- Verify outcomes using action results, robot status, and camera observations.
- Recover appropriately from failures.
- Log observations, LLM decisions, executed actions, and outcomes.
- Explain their approach, results, and limitations.

## Evaluation Criteria

### 1. Task Performance

- Number of cubes correctly sorted
- Number of incorrect placements

### 2. LLM Agent Behavior

- Valid structured LLM decisions
- Meaningful use of observations, memory, action outcomes, and recovery reasoning
- Clear logs showing observation -> LLM decision -> action -> result
- Evidence that LLM decisions affect the executed high-level action sequence

### 3. Reliability

- Performance under instructor-selected evaluation conditions
- Ability to recover from failed actions
- Ability to run without source-code changes between evaluation runs

### 4. Engineering and Presentation

- Effective use of workshop concepts
- Code quality and system design
- Clear demonstration and explanation of results and limitations
