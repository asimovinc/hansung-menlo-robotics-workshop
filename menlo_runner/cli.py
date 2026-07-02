# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

import argparse
import asyncio
import inspect
import importlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from menlo_runner.basics import print_position, screenshot
from menlo_runner.completion import (
    DEFAULT_MAX_DELIVERED_CUBES,
    DEFAULT_ROUND,
    CompletionConfig,
    completion_config_for_round,
    level_from_program_name,
)
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


@dataclass(frozen=True)
class ProgramSpec:
    module_name: str
    require_tokamak: bool


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


def _program_spec(program_name: str) -> ProgramSpec:
    if program_name in PROGRAMS:
        module_name, require_tokamak = PROGRAMS[program_name]
        return ProgramSpec(module_name=module_name, require_tokamak=require_tokamak)
    return ProgramSpec(module_name=program_name, require_tokamak=False)


async def _run_program_in_existing_context(ctx: Any, module_name: str) -> None:
    if _program_requires_tokamak(module_name) and not ctx.config.tokamak_api_key:
        print("This program requires TOKAMAK_API_KEY. Add it to .env and start a new session.")
        return
    program = _load_program(module_name)
    await program(ctx)


async def _run_completion_in_existing_context(
    ctx: Any,
    program_name: str,
    *,
    level: int | None,
    round_name: str,
    max_delivered_cubes: int | None,
    max_elapsed_s: float | None,
    max_cycles: int | None,
) -> None:
    resolved_level = level if level is not None else level_from_program_name(program_name)
    completion = (
        CompletionConfig(
            level=resolved_level,
            max_delivered_cubes=max_delivered_cubes,
            max_elapsed_s=max_elapsed_s,
        )
        if max_elapsed_s is not None
        else completion_config_for_round(
            resolved_level,
            round_name=round_name,
            max_delivered_cubes=(
                DEFAULT_MAX_DELIVERED_CUBES
                if max_delivered_cubes is None
                else max_delivered_cubes
            ),
        )
    )
    spec = _program_spec(program_name)
    if spec.require_tokamak and not ctx.config.tokamak_api_key:
        print("This program requires TOKAMAK_API_KEY. Add it to .env and start a new session.")
        return

    module = importlib.import_module(spec.module_name)
    run_agent = getattr(module, "run_agent", None)
    if run_agent is None:
        raise RuntimeError(f"{program_name} does not expose run_agent(ctx, ...).")

    signature = inspect.signature(run_agent)
    if "completion" not in signature.parameters:
        raise RuntimeError(
            f"{program_name} does not support completion runs yet. "
            "Add a completion parameter to its run_agent function."
        )

    kwargs: dict[str, Any] = {"completion": completion}
    if max_cycles is not None and "max_cycles" in signature.parameters:
        kwargs["max_cycles"] = max_cycles
    if "task" in signature.parameters:
        resolve_task = getattr(module, "resolve_task", None)
        task = resolve_task(ctx) if resolve_task is not None else getattr(module, "TASK", "")
        kwargs["task"] = task
        if task:
            print(task)
    print(f"Running completion wrapper for {program_name}")
    await run_agent(ctx, **kwargs)


async def _run_completion(
    program_name: str,
    *,
    level: int | None,
    round_name: str,
    max_delivered_cubes: int | None,
    max_elapsed_s: float | None,
    max_cycles: int | None,
) -> None:
    spec = _program_spec(program_name)
    config = load_config(require_tokamak=spec.require_tokamak)
    ctx = await open_robot_context(config, name_prefix=f"complete-{program_name}")
    try:
        await _run_completion_in_existing_context(
            ctx,
            program_name,
            level=level,
            round_name=round_name,
            max_delivered_cubes=max_delivered_cubes,
            max_elapsed_s=max_elapsed_s,
            max_cycles=max_cycles,
        )
    finally:
        await ctx.close()
        print("Cleaned up robot and closed the client.")


def _print_session_help() -> None:
    print(
        """
Commands:
  programs                 List built-in programs
  run <program>            Run a built-in program
  custom <module>          Run a custom module with async def run(ctx)
  complete <program>       Run a built-in program/module with completion scoring
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
                elif command == "complete":
                    parser = build_completion_parser()
                    try:
                        complete_args = parser.parse_args(args)
                    except SystemExit:
                        continue
                    await _run_completion_in_existing_context(
                        ctx,
                        complete_args.program,
                        level=complete_args.level,
                        round_name=complete_args.round,
                        max_delivered_cubes=complete_args.cubes,
                        max_elapsed_s=complete_args.seconds,
                        max_cycles=complete_args.max_cycles,
                    )
                else:
                    print(f"Unknown command '{command}'. Type 'help'.")
            except Exception as exc:
                print(f"ERROR: {exc}")
    finally:
        await ctx.close()
        print("Cleaned up robot and closed the client.")


def build_completion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="complete",
        description="Run a project program with scoring and the simulation time cap.",
    )
    parser.add_argument(
        "program",
        help="Built-in program name or import path, for example level-0-starter.",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Project level for scoring. Omit to infer from the program name.",
    )
    parser.add_argument(
        "--round",
        choices=("round1", "round2", "round3"),
        default=DEFAULT_ROUND,
        help=f"Round timing to use when --seconds is omitted (default: {DEFAULT_ROUND}).",
    )
    parser.add_argument(
        "--cubes",
        type=int,
        default=DEFAULT_MAX_DELIVERED_CUBES,
        help=f"Stop target for delivered cubes (default: {DEFAULT_MAX_DELIVERED_CUBES}).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Manual time limit in seconds. Omit to use --round timing.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10_000,
        help="Maximum agent cycles before stopping (default: 10000).",
    )
    return parser


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

    complete = subparsers.add_parser(
        "complete",
        help="Run a project program with scoring and the simulation time cap.",
    )
    complete.add_argument(
        "program",
        help="Built-in program name or import path, for example level-0-starter.",
    )
    complete.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Project level for scoring. Omit to infer from the program name.",
    )
    complete.add_argument(
        "--round",
        choices=("round1", "round2", "round3"),
        default=DEFAULT_ROUND,
        help=f"Round timing to use when --seconds is omitted (default: {DEFAULT_ROUND}).",
    )
    complete.add_argument(
        "--cubes",
        type=int,
        default=DEFAULT_MAX_DELIVERED_CUBES,
        help=f"Stop target for delivered cubes (default: {DEFAULT_MAX_DELIVERED_CUBES}).",
    )
    complete.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Manual time limit in seconds. Omit to use --round timing.",
    )
    complete.add_argument(
        "--max-cycles",
        type=int,
        default=10_000,
        help="Maximum agent cycles before stopping (default: 10000).",
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

        if args.command == "complete":
            asyncio.run(
                _run_completion(
                    args.program,
                    level=args.level,
                    round_name=args.round,
                    max_delivered_cubes=args.cubes,
                    max_elapsed_s=args.seconds,
                    max_cycles=args.max_cycles,
                )
            )
            return

        module_name, require_tokamak = PROGRAMS[args.command]
        asyncio.run(_run_program(module_name, require_tokamak=require_tokamak))
    except MenloAuthError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

