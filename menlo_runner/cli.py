# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.
# 스타터 코드: 프로젝트 제출을 위해 이 공용 파일을 직접 수정하지 마세요.
# 프로젝트 코드는 프로젝트 노트북 또는 menlo_runner/programs/의 새 파일에 작성하세요.

from __future__ import annotations

import argparse
import asyncio
import importlib
from typing import Any, Awaitable, Callable

from menlo_runner.basics import print_position, screenshot
from menlo_runner.config import load_config
from menlo_runner.context import MenloAuthError, open_robot_context
from menlo_runner.scene import get_scene_text


Program = Callable[[Any], Awaitable[None]]


PROGRAMS = {
    "basics-demo": ("menlo_runner.programs.basics_demo", False),
    "perception-demo": ("menlo_runner.programs.perception_demo", False),
    "navigation-demo": ("menlo_runner.programs.navigation_demo", False),
    "agent-demo": ("menlo_runner.programs.agent_demo", True),
    "student-program": ("menlo_runner.programs.student_program", False),
    "level-0-starter": ("menlo_runner.programs.project.en.level_0_starter", True),
    "level-1-starter": ("menlo_runner.programs.project.en.level_1_starter", True),
    "level-2-starter": ("menlo_runner.programs.project.en.level_2_starter", True),
    "level-0-starter-ko": ("menlo_runner.programs.project.ko.level_0_starter_ko", True),
    "level-1-starter-ko": ("menlo_runner.programs.project.ko.level_1_starter_ko", True),
    "level-2-starter-ko": ("menlo_runner.programs.project.ko.level_2_starter_ko", True),
}


def _load_program(module_name: str) -> Program:
    module = importlib.import_module(module_name)
    run = getattr(module, "run", None)
    if run is None:
        raise RuntimeError(f"{module_name} does not expose async def run(ctx).")
    return run


async def _run_program(module_name: str, *, require_tokamak: bool) -> None:
    config = load_config(require_tokamak=require_tokamak)
    ctx = await open_robot_context(config, name_prefix=module_name.rsplit(".", 1)[-1])
    try:
        program = _load_program(module_name)
        await program(ctx)
    finally:
        await ctx.close()
        print("Cleaned up robot and closed the client.")


def _program_requires_tokamak(module_name: str) -> bool:
    for registered_module, requires_tokamak in PROGRAMS.values():
        if module_name == registered_module:
            return requires_tokamak
    return False


async def _run_program_in_existing_context(ctx: Any, module_name: str) -> None:
    if _program_requires_tokamak(module_name) and not ctx.config.tokamak_api_key:
        print("This program requires TOKAMAK_API_KEY. Add it to .env and start a new session.")
        return
    program = _load_program(module_name)
    await program(ctx)


def _print_session_help() -> None:
    print(
        """
Commands:
  programs                 List built-in programs
  run <program>            Run a built-in program
  custom <module>          Run a custom module with async def run(ctx)
  scene                    Print a text summary of robot, pads, and cubes
  position                 Print robot position and status
  screenshot [path]        Save the robot POV image
  skills                   List currently advertised viewer skills
  viewer                   Print the viewer URL again
  reset                    Use the reset button in the viewer UI
  help                     Show this help
  quit                     Disconnect, delete the robot, and exit
""".strip()
    )


async def _interactive_session() -> None:
    config = load_config(require_tokamak=False)
    ctx = await open_robot_context(config, name_prefix="interactive-session")
    print("\nSame-viewer session is ready. Type 'help' for commands.")
    try:
        while True:
            raw = input("menlo> ").strip()
            if not raw:
                continue

            parts = raw.split()
            command = parts[0].lower()
            args = parts[1:]

            try:
                if command in {"quit", "exit", "q"}:
                    break
                if command == "help":
                    _print_session_help()
                elif command == "programs":
                    for name in PROGRAMS:
                        print(f"  {name}")
                elif command == "viewer":
                    print(ctx.viewer_url)
                elif command == "skills":
                    skills = await ctx.session.discover_skills()
                    for skill in skills:
                        print(f"  - {skill.name}")
                elif command == "position":
                    await print_position(ctx, "CURRENT")
                elif command == "scene":
                    print(await get_scene_text(ctx))
                elif command == "screenshot":
                    path = args[0] if args else "outputs/session-screenshot.jpg"
                    await screenshot(ctx, "Robot POV:", path)
                elif command == "reset":
                    print("Use the reset button in the viewer UI, then continue from this prompt.")
                elif command == "run":
                    if not args:
                        print("Usage: run <program>")
                        continue
                    program_name = args[0]
                    if program_name not in PROGRAMS:
                        print(f"Unknown program '{program_name}'. Try: programs")
                        continue
                    module_name, _ = PROGRAMS[program_name]
                    await _run_program_in_existing_context(ctx, module_name)
                elif command == "custom":
                    if not args:
                        print("Usage: custom <module>")
                        continue
                    await _run_program_in_existing_context(ctx, args[0])
                else:
                    print(f"Unknown command '{command}'. Type 'help'.")
            except Exception as exc:
                print(f"ERROR: {exc}")
    finally:
        await ctx.close()
        print("Cleaned up robot and closed the client.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Menlo robot SDK programs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in PROGRAMS:
        subparsers.add_parser(command)

    custom = subparsers.add_parser("custom", help="Run a module that exposes async def run(ctx).")
    custom.add_argument("module", help="Import path, for example menlo_runner.programs.student_program")
    custom.add_argument(
        "--tokamak",
        action="store_true",
        help="Require TOKAMAK_API_KEY for the custom program.",
    )

    subparsers.add_parser(
        "session",
        help="Keep one robot/viewer alive and run multiple commands against it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "session":
            asyncio.run(_interactive_session())
            return

        if args.command == "custom":
            asyncio.run(_run_program(args.module, require_tokamak=args.tokamak))
            return

        module_name, require_tokamak = PROGRAMS[args.command]
        asyncio.run(_run_program(module_name, require_tokamak=require_tokamak))
    except MenloAuthError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

