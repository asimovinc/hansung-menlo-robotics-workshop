# Menlo Robot Workshop Programs

[English](README.md) | [한국어](README.ko.md)

This codebase mirrors the four workshop notebooks:

- Workshop 1: SDK basics, viewer, robot state, scene state, simple actions
- Workshop 2: perception with camera frames, HSV color blobs, angle estimates, depth hooks
- Workshop 3: custom navigation with global state and vision-only control
- Workshop 4: tool-calling LLM agents

Student notebooks are separated by language:

- `notebooks/student/en/`: English student notebooks
- `notebooks/student/ko/`: Korean student notebooks

## Choose How to Work

### Option 1: Notebooks / Google Colab

Use this option if you want to complete the workshop entirely in a notebook. You do
not need to clone the repository, install Python locally, or set up the IDE scaffold.

1. Open the English or Korean notebook for the workshop.
2. Upload it to Google Colab, or open it in your preferred notebook environment.
3. Follow the setup cells in the notebook to install packages and configure your API
   keys. In Colab, store keys in the Secrets manager rather than writing them into a
   notebook cell.
4. Open the printed viewer URL in Google Chrome when instructed.

### Option 2: Local IDE Scaffold

Use this option if you prefer VS Code, PyCharm, or another local IDE. Clone this
repository and use the reusable modules under `menlo_runner/`.

## Local IDE Setup

Install the package from this folder:

```powershell
py -m pip install -e .
```

Copy `.env.example` to `.env` if present, or create `.env` with:

```text
MENLO_API_KEY=...
TOKAMAK_API_KEY=...
```

`TOKAMAK_API_KEY` is only needed for LLM/VLM agent examples.

## Run Workshop Demos

Use the long form if the `menlo-run` script is not on PATH:

```powershell
py -m menlo_runner.cli basics-demo
py -m menlo_runner.cli perception-demo
py -m menlo_runner.cli navigation-demo
py -m menlo_runner.cli agent-demo
py -m menlo_runner.cli level-1-starter
py -m menlo_runner.cli level-2-starter
```

The demos create a simulated robot, print a viewer URL, wait for you to open it in Chrome, run the selected program, and then clean up the robot.

## Interactive Session

To keep one robot and viewer open while running multiple workshop demos:

```powershell
py -m menlo_runner.cli session
```

Useful commands:

```text
programs                 List built-in programs
run <program>            Run a built-in program
custom <module>          Run a custom module with async def run(ctx)
scene                    Print robot, pad, and cube state
position                 Print robot position and status
screenshot [path]        Save the robot POV image
skills                   List viewer skills
viewer                   Print the viewer URL again
reset                    Use the reset button in the viewer UI
quit                     Disconnect, delete the robot, and exit
```

## Module Map

- `menlo_runner.basics`: simple SDK action wrappers used in Workshop 1
- `menlo_runner.perception`: Workshop 2 color blob detection, `perceive`, annotation, depth hooks
- `menlo_runner.navigation`: Workshop 3 `turn_to_face`, `my_go_to_global`, `my_go_to_visual`
- `menlo_runner.agents`: Workshop 4 tool registry, executor, ReAct-style `WorkshopAgent`
- `menlo_runner.scene`: scene-state helpers and cube/pad utilities
- `menlo_runner.programs`: runnable examples for concepts already introduced in the student notebooks

Exercise solutions are intentionally not included. Complete the exercise cells in the
student notebooks. For the final project, use the level-specific starters in
`notebooks/project/` or `menlo_runner/programs/project/en/level_1_starter.py` and
`menlo_runner/programs/project/en/level_2_starter.py`.

