# Menlo Robot Workshop Programs

[English](README.md) | [한국어](README.ko.md)

이 저장소는 Menlo Robot Workshop 노트북과 로컬 IDE에서 실행할 수 있는 예제 프로그램을 함께 제공합니다.

- Workshop 1: SDK 기본 사용법, viewer, robot state, scene state, 기본 action
- Workshop 2: 카메라 프레임 기반 perception, HSV color blob, angle estimation, depth hook
- Workshop 3: global state와 vision-only control을 사용하는 custom navigation
- Workshop 4: tool-calling LLM agent

Student notebook은 언어별로 분리되어 있습니다.

- `notebooks/student/en/`: English student notebooks
- `notebooks/student/ko/`: Korean student notebooks

## 작업 방식 선택

### Option 1: Notebook / Google Colab

Google Drive에 올린 notebook을 Colab에서 열어 브라우저만으로 작업하려면 이 방식을 사용하세요. 저장소를 clone하거나 로컬 Python 환경을 준비할 필요가 없습니다.

1. 영어 또는 한국어 notebook을 Google Drive에 업로드하고 Colab에서 엽니다.
2. 첫 번째 setup cell을 실행합니다. 이 cell은 GitHub 저장소에서 `menlo_runner` package를 설치합니다.
3. API key와 robot context cell을 실행합니다. Colab에서는 API key를 notebook cell에 직접 쓰지 말고 Secrets manager에 저장하세요.
4. 출력된 Menlo viewer URL을 Chrome에서 엽니다.
5. notebook 안의 TODO 부분을 채웁니다.
6. 마지막 project 실행 cell을 실행합니다. Project starter notebook은 round timing을 묻고 최대 12개 cube delivery에서 멈춥니다.

필수 Colab secret 이름:

- `MENLO_API_KEY`: `platform.menlo.ai` -> Settings -> API Keys에서 발급받은 Menlo API key
- `TOKAMAK_API_KEY`: Workshop 4와 모든 project starter agent에 필요

선택 LLM model 설정:

```python
import os

os.environ["MENLO_LLM_MODEL"] = "minimaxai/minimax-m3"
os.environ["MENLO_VLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# 승인된 다른 선택지:
# os.environ["MENLO_LLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
# os.environ["MENLO_VLM_MODEL"] = "minimaxai/minimax-m3"
```

Notebook 사용자는 setup cell을 실행한 뒤 project agent를 실행하기 전에 위 cell을 추가해서 실행하세요. Local IDE 사용자는 `.env`에 같은 값을 설정하거나 `menlo_runner.llm.call_llm(...)`에 `model=...`을 직접 넘길 수 있습니다.

이 방식에서는 notebook 파일은 Google Drive에 있지만 `menlo_runner.completion` 같은 support code는 GitHub에서 설치됩니다. 따라서 GitHub 저장소가 최신으로 업데이트되어 있으면 Colab에서도 최신 코드를 사용할 수 있습니다. Colab이 오래된 코드를 쓰는 것처럼 보이면 runtime을 restart하고 setup cell을 다시 실행하세요.

### Option 2: Local IDE with Starter Notebooks

저장소를 로컬에 clone했지만 주로 `notebooks/project/` 아래의 project notebook 파일에서 작업하려면 이 방식을 사용하세요.

1. 이 저장소를 clone합니다.
2. 저장소 폴더에서 `py -m pip install -e .`로 local package를 설치합니다.
3. IDE에서 `notebooks/project/` 아래의 project notebook을 엽니다.
4. API key와 robot context cell을 실행합니다.
5. 출력된 Menlo viewer URL을 Chrome에서 엽니다.
6. notebook 안의 TODO 부분을 채웁니다.
7. 마지막 project 실행 cell을 실행합니다. 이 cell은 local clone의 코드를 사용하고, round timing을 묻고, 최대 12개 cube delivery에서 멈춥니다.

이 방식에서는 package가 editable mode로 설치되므로 `menlo_runner/` 아래의 로컬 수정사항을 notebook에서 바로 사용할 수 있습니다.

### Option 3: Local IDE with Python Starters

`.py` starter 파일을 직접 수정하고 terminal 또는 IDE run configuration으로 실행하려면 이 방식을 사용하세요.

1. 이 저장소를 clone합니다.
2. 저장소 폴더에서 `py -m pip install -e .`로 local package를 설치합니다.
3. `menlo_runner/programs/project/` 아래의 Python starter 파일에서 TODO 부분을 채웁니다.
4. terminal에서 맞는 starter command를 실행합니다. 예:
   `py -m menlo_runner.cli level-1-starter`
5. 출력된 Menlo viewer URL을 Chrome에서 엽니다.

Project starter command는 starter notebook과 같은 scored workflow를 실행합니다. Starter는 `round1`, `round2`, `round3`, 또는 `manual` timing을 묻습니다. Round 제한 시간은 각각 5분, 10분, 15분이며 모든 round는 최대 12개 cube delivery에서 멈춥니다. Level 0은 이동한 cube 1개당 5점입니다. Level 1은 첫 성공 cube 60점, 이후 추가 cube 1개당 20점입니다. Level 2는 첫 성공 cube 60점, 이후 추가 cube 1개당 40점입니다.

## Local IDE Setup

저장소 폴더에서 package를 설치합니다.

```powershell
py -m pip install -e .
```

`.env.example`을 `.env`로 복사하거나 다음 내용으로 `.env`를 만듭니다.

```text
MENLO_API_KEY=...
TOKAMAK_API_KEY=...
MENLO_LLM_MODEL=minimaxai/minimax-m3
MENLO_VLM_MODEL=qwen/qwen3.6-35b-a3b
```

`MENLO_API_KEY`는 `platform.menlo.ai` -> Settings -> API Keys에서 발급받습니다.
`TOKAMAK_API_KEY`는 Workshop 4와 모든 project starter agent에 필요합니다. Project starter는 LLM-assisted decision loop를 사용합니다.

## Workshop Demo 실행

`menlo-run` script가 PATH에 없다면 다음 long form을 사용하세요.

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

Demo는 simulated robot을 만들고 viewer URL을 출력합니다. Chrome에서 viewer를 연 뒤 선택한 program을 실행하고 마지막에 robot을 정리합니다.

## Interactive Session

하나의 robot과 viewer를 유지한 채 여러 workshop demo를 실행하려면 다음 명령을 사용하세요.

```powershell
py -m menlo_runner.cli session
```

사용 가능한 command:

```text
programs                 내장 program 목록 표시
run <program>            내장 program 실행
custom <module>          async def run(ctx)를 제공하는 custom module 실행
complete <program>       completion scoring과 time cap으로 project program 실행
scene                    robot, pad, cube 상태 요약 출력
position                 robot 위치와 상태 출력
screenshot [path]        robot POV image 저장
skills                   viewer skill 목록 표시
viewer                   viewer URL 다시 출력
reset                    viewer UI의 reset button 사용
quit                     연결 해제, robot 삭제, 종료
```

## Module Map

- `menlo_runner.basics`: Workshop 1의 기본 SDK action wrapper
- `menlo_runner.perception`: Workshop 2 color blob detection, `perceive`, annotation, depth hook
- `menlo_runner.navigation`: Workshop 3 `turn_to_face`, `my_go_to_global`, `my_go_to_visual`
- `menlo_runner.agents`: Workshop 4 tool registry, executor, ReAct-style `WorkshopAgent`
- `menlo_runner.scene`: scene-state helper와 cube/pad utility
- `menlo_runner.programs`: student notebook에서 배운 개념을 실행 가능한 예제로 정리한 module

Exercise solution은 의도적으로 포함하지 않습니다. Student notebook의 exercise cell을 직접 완성하거나 IDE에서 작업할 때는 같은 기능을 `student_program.py`에 작성할 수 있습니다.

최종 project는 `notebooks/project/`의 level별 starter notebook을 사용하세요.

Project instructions:

- English: `docs/project_instructions.md`
- Korean: `docs/project_instructions.ko.md`
- 발표 자료: `docs/project_instruction_kor_updated.pptx`
- 원본 발표 자료: `docs/project_instruction_kor.pptx`

English Python starters:

- `menlo_runner/programs/project/en/level_0_starter.py`
- `menlo_runner/programs/project/en/level_1_starter.py`
- `menlo_runner/programs/project/en/level_2_starter.py`

Korean Python starters:

- `menlo_runner/programs/project/ko/level_0_starter_ko.py`
- `menlo_runner/programs/project/ko/level_1_starter_ko.py`
- `menlo_runner/programs/project/ko/level_2_starter_ko.py`
