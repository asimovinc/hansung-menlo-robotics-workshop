# Menlo 로봇 워크숍 프로그램

[English](README.md) | [한국어](README.ko.md)

이 코드베이스는 네 개의 워크숍 노트북과 같은 흐름으로 구성되어 있습니다.

- 워크숍 1: SDK 기초, 뷰어, 로봇 상태, 장면 상태, 기본 동작
- 워크숍 2: 카메라 프레임을 이용한 인식, HSV 색상 블롭, 각도 추정, 깊이 처리
- 워크숍 3: 전역 상태 및 비전 전용 제어를 이용한 사용자 정의 탐색
- 워크숍 4: 도구 호출형 LLM 에이전트

학생용 노트북은 언어별로 나뉩니다.

- `notebooks/student/en/`: 영문 학생용 노트북
- `notebooks/student/ko/`: 한국어 학생용 노트북

## 학습 방식 선택

### 옵션 1: 노트북 / Google Colab

워크숍 전체를 노트북에서 진행하려면 이 방식을 선택하세요. 저장소를 복제하거나
Python을 로컬에 설치하거나 IDE scaffold를 설정할 필요가 없습니다.

1. 원하는 워크숍의 영문 또는 한국어 노트북을 엽니다.
2. Google Colab에 업로드하거나 원하는 노트북 환경에서 엽니다.
3. 노트북의 설정 셀에 따라 패키지를 설치하고 API 키를 설정합니다. Colab에서는
   키를 노트북 셀에 직접 적지 말고 Secrets 관리자에 저장하세요.
4. 안내가 나오면 출력된 뷰어 URL을 Google Chrome에서 엽니다.

### 옵션 2: 로컬 IDE Scaffold

VS Code, PyCharm 또는 다른 로컬 IDE에서 작업하려면 이 방식을 선택하세요. 이
저장소를 복제한 뒤 `menlo_runner/`의 재사용 가능한 모듈을 이용합니다.

## 로컬 IDE 설치

이 폴더에서 패키지를 설치합니다.

```powershell
py -m pip install -e .
```

`.env.example`을 `.env`로 복사하거나 다음 내용으로 `.env` 파일을 만드세요.

```text
MENLO_API_KEY=...
TOKAMAK_API_KEY=...
```

`TOKAMAK_API_KEY`는 LLM/VLM 에이전트 예제에서만 필요합니다.

## 워크숍 데모 실행

`menlo-run` 명령이 PATH에 등록되지 않았다면 다음과 같이 긴 형식으로 실행하세요.

```powershell
py -m menlo_runner.cli basics-demo
py -m menlo_runner.cli perception-demo
py -m menlo_runner.cli navigation-demo
py -m menlo_runner.cli agent-demo
py -m menlo_runner.cli student-program
```

데모는 시뮬레이션 로봇을 생성하고 뷰어 URL을 출력합니다. Chrome에서 뷰어를 열 때까지 기다린 뒤 선택한 프로그램을 실행하고 마지막에 로봇을 정리합니다.

## 대화형 세션

하나의 로봇과 뷰어를 유지한 채 여러 워크숍 데모를 실행하려면 다음 명령을 사용하세요.

```powershell
py -m menlo_runner.cli session
```

사용할 수 있는 명령은 다음과 같습니다.

```text
programs                 내장 프로그램 목록 표시
run <program>            내장 프로그램 실행
custom <module>          async def run(ctx)를 제공하는 사용자 모듈 실행
scene                    로봇, 패드, 큐브 상태 출력
position                 로봇 위치와 상태 출력
screenshot [path]        로봇 POV 이미지 저장
skills                   뷰어 스킬 목록 표시
viewer                   뷰어 URL 다시 출력
reset                    뷰어 UI의 초기화 버튼 사용
quit                     연결 해제, 로봇 삭제 후 종료
```

## 모듈 구성

- `menlo_runner.basics`: 워크숍 1에서 사용하는 기본 SDK 동작 래퍼
- `menlo_runner.perception`: 워크숍 2의 색상 블롭 검출, `perceive`, 주석 표시, 깊이 처리
- `menlo_runner.navigation`: 워크숍 3의 `turn_to_face`, `my_go_to_global`, `my_go_to_visual`
- `menlo_runner.agents`: 워크숍 4의 도구 레지스트리, 실행기, ReAct 방식 `WorkshopAgent`
- `menlo_runner.scene`: 장면 상태 및 큐브/패드 도우미
- `menlo_runner.programs`: 학생 노트북에서 이미 배운 개념을 실행하는 예제

연습 문제 해답은 의도적으로 포함하지 않았습니다. 학생용 노트북의 연습 문제 셀을 직접 완성하거나 IDE에서 작업할 때 `student_program.py`에 같은 기능을 작성하세요.
