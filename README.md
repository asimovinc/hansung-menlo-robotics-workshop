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
2. Upload it to Google Drive and open it in Google Colab.
3. Run the first setup cell. It installs `menlo_runner` from the GitHub repository.
4. Follow the setup cells in the notebook to configure your API
   keys. In Colab, store keys in the Secrets manager rather than writing them into a
   notebook cell.
5. Open the printed viewer URL in Google Chrome when instructed.
6. Fill in the TODO sections inside the notebook.
7. Run the final project cell. Project starter notebooks run the 10-minute scored
   simulation by default.

Required Colab secret names:

- `MENLO_API_KEY`: your Menlo API key from `platform.menlo.ai` -> Settings -> API Keys
- `TOKAMAK_API_KEY`: required for Workshop 4 and all project starter agents

Optional LLM model override:

```python
import os

os.environ["MENLO_LLM_MODEL"] = "minimaxai/minimax-m3"
os.environ["MENLO_VLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# Other approved choices for either variable:
# os.environ["MENLO_LLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# os.environ["MENLO_VLM_MODEL"] = "minimaxai/minimax-m3"
```

Put this in a notebook cell after setup and before the project agent runs. Local
IDE users can set the same value in `.env` or pass `model=...` directly to
`menlo_runner.llm.call_llm(...)`.

### Option 2: Local IDE With Starter Notebooks

Use this option when the repository is cloned locally, but you still want to work
mainly inside the project notebook files from `notebooks/project/`.

1. Clone this repository.
2. Install the local package from the repository folder with `py -m pip install -e .`.
3. Open a project notebook from `notebooks/project/` in your IDE.
4. Run the API key and robot context cells.
5. Open the printed Menlo viewer URL in Chrome.
6. Fill in the TODO sections in the notebook or edit the matching Python starter.
7. Run the final project cell. It uses the local cloned code and runs the
   10-minute scored simulation by default.

In this workflow, local edits to `menlo_runner/` are available to the notebook
because the package is installed in editable mode.

### Option 3: Local IDE With Python Starters

Use this option when you want to edit `.py` starter files directly and run them
from a terminal or IDE run configuration.

1. Clone this repository.
2. Install the local package from the repository folder with `py -m pip install -e .`.
3. Fill in the TODO sections in one of the Python starters under
   `menlo_runner/programs/project/`.
4. Run the matching starter command from a terminal, for example
   `py -m menlo_runner.cli level-1-starter`.
5. Open the printed Menlo viewer URL in Chrome.

Project starter commands run the same 10-minute scored simulation used by the
starter notebooks. Delivery points are uncapped: Level 0 gives 10 points per
delivery, Level 1 gives 20 points per delivery, and Level 2 gives 30 points per
delivery.

## Local IDE Setup

Install the package from this folder:

```powershell
py -m pip install -e .
```

Copy `.env.example` to `.env` if present, or create `.env` with:

```text
MENLO_API_KEY=...
TOKAMAK_API_KEY=...
MENLO_LLM_MODEL=minimaxai/minimax-m3
MENLO_VLM_MODEL=qwen/qwen3.6-35b-a3b
```

`MENLO_API_KEY` comes from `platform.menlo.ai` -> Settings -> API Keys.
`TOKAMAK_API_KEY` is required for Workshop 4 and for all project starter agents because the project requires an LLM-assisted decision loop.

## Run Workshop Demos

Use the long form if the `menlo-run` script is not on PATH:

```powershell
py -m menlo_runner.cli basics-demo
py -m menlo_runner.cli perception-demo
py -m menlo_runner.cli navigation-demo
py -m menlo_runner.cli agent-demo
py -m menlo_runner.cli student-program
py -m menlo_runner.cli level-0-starter
py -m menlo_runner.cli level-1-starter
py -m menlo_runner.cli level-2-starter
py -m menlo_runner.cli level-0-starter-ko
py -m menlo_runner.cli level-1-starter-ko
py -m menlo_runner.cli level-2-starter-ko
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
student notebooks, or write the equivalent code in `student_program.py` when working in an IDE.

For the final project, use the level-specific starters in `notebooks/project/`.

Project instructions:

- English: `docs/project_instructions.md`
- Korean: `docs/project_instructions.ko.md`
- Presentation deck: `docs/project_instruction_kor_updated.pptx`
- Original presentation deck: `docs/project_instruction_kor.pptx`

English Python starters:

- `menlo_runner/programs/project/en/level_0_starter.py`
- `menlo_runner/programs/project/en/level_1_starter.py`
- `menlo_runner/programs/project/en/level_2_starter.py`

Korean Python starters:

- `menlo_runner/programs/project/ko/level_0_starter_ko.py`
- `menlo_runner/programs/project/ko/level_1_starter_ko.py`
- `menlo_runner/programs/project/ko/level_2_starter_ko.py`
