# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

import math
from typing import Any

from menlo_runner.llm import ask_vlm, build_system_prompt, call_llm, parse_tool_call
from menlo_runner.scene import COLOR_TO_PAD, held_cube_info


DEFAULT_TOOLS = {
    "set_velocity": {
        "description": (
            "Move the robot for a fixed duration. Args: vx (forward m/s, max 1.5), "
            "vy (sideways m/s), wz (turn rad/s, max 0.6), duration_s. "
            "The robot cannot spin in place; pair turns with vx=0.2."
        )
    },
    "go_to": {
        "description": (
            "Walk to a named entity using built-in pathfinding. Args: entity_id, "
            "for example 'pad_C' or 'cube_2'."
        )
    },
    "pick_cube": {
        "description": (
            "Pick up the nearest visible cube. No args. Precondition: the robot "
            "should be close to a cube. Returns cube id and color on success."
        )
    },
    "place": {
        "description": (
            "Deliver the currently held cube to its correct pad. No args. "
            "Routing: red->pad_B, green->pad_C, blue->pad_D, yellow->pad_E."
        )
    },
    "look": {
        "description": "Capture the robot POV camera and return a VLM text description. No args."
    },
    "get_scene_summary": {
        "description": (
            "Get a compact scene-state summary of visible cubes: ids, colors, and "
            "distances from the robot. No args. Use as a baseline/debug tool."
        )
    },
    "check_held_object": {
        "description": "Check what cube the robot is currently holding. No args."
    },
}


class WorkshopAgent:
    """Workshop 4 style ReAct agent: LLM tool call, SDK execute, observe result."""

    def __init__(
        self,
        ctx: Any,
        *,
        tokamak_api_key: str,
        tools: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.ctx = ctx
        self.tokamak_api_key = tokamak_api_key
        self.tools = dict(tools or DEFAULT_TOOLS)
        self.tool_log: list[dict[str, Any]] = []

    async def _visible_cube_summary(self) -> str:
        scene = await self.ctx.state("scene_state")
        status = await self.ctx.state("robot_status")
        rx, ry = status.robot.pose.position[0], status.robot.pose.position[1]
        lines = []
        for entity_id, entity in sorted(scene.entities.items()):
            if not entity_id.startswith("cube_") or not entity.visible or entity.attached_to:
                continue
            distance = math.hypot(entity.pose.position[0] - rx, entity.pose.position[1] - ry)
            color = entity.state.get("color", "unknown") if entity.state else "unknown"
            lines.append(f"  {entity_id}: {color}, {distance:.1f}m away")
        return "Visible cubes:\n" + "\n".join(lines) if lines else "No visible cubes."

    async def _nearest_visible_cube_id(self) -> tuple[str, str] | None:
        scene = await self.ctx.state("scene_state")
        status = await self.ctx.state("robot_status")
        rx, ry = status.robot.pose.position[0], status.robot.pose.position[1]
        candidates = []
        for entity_id, entity in scene.entities.items():
            if not entity_id.startswith("cube_") or not entity.visible or entity.attached_to:
                continue
            distance = math.hypot(entity.pose.position[0] - rx, entity.pose.position[1] - ry)
            color = entity.state.get("color", "unknown") if entity.state else "unknown"
            candidates.append((distance, entity_id, color))
        if not candidates:
            return None
        _, entity_id, color = min(candidates)
        return entity_id, color

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "set_velocity":
            result = await self.ctx.invoke("set_velocity", args, timeout_s=60)
            return f"set_velocity finished: status={result.status}"

        if name == "go_to":
            entity_id = args["entity_id"]
            result = await self.ctx.invoke(
                "go_to",
                {"target": {"kind": "entity", "entity_id": entity_id}},
                timeout_s=300,
            )
            if result.status != "done":
                err = f" ({result.error.message})" if result.error else ""
                return f"go_to {entity_id} failed{err}"
            return f"Reached {entity_id}"

        if name == "look":
            jpeg = await self.ctx.get_vision("pov")
            return ask_vlm(
                jpeg,
                "Describe visible objects, cube colors, pad signs, obstacles, and rough positions.",
                api_key=self.tokamak_api_key,
            )

        if name == "get_scene_summary":
            return await self._visible_cube_summary()

        if name == "pick_cube":
            target = await self._nearest_visible_cube_id()
            if target is None:
                return "No visible cubes to pick."
            cube_id, color = target
            result = await self.ctx.invoke(
                "pick_entity",
                {"target": {"kind": "entity", "entity_id": cube_id}},
                timeout_s=300,
            )
            if result.status != "done":
                err = f": {result.error.message}" if result.error else ""
                return f"pick_entity {cube_id} failed{err}"
            return f"Picked {cube_id} ({color})"

        if name == "check_held_object":
            held = await held_cube_info(self.ctx)
            if held is None:
                return "Holding: nothing"
            cube_id, color = held
            return f"Holding: {cube_id} ({color})"

        if name == "place":
            held = await held_cube_info(self.ctx)
            if held is None:
                return "Not holding anything."
            cube_id, color = held
            pad_id = COLOR_TO_PAD.get(color)
            if pad_id is None:
                return f"Unknown held color {color!r}; cannot choose a pad."
            result = await self.ctx.invoke(
                "place_entity",
                {"target": {"kind": "entity", "entity_id": pad_id}},
                timeout_s=300,
            )
            if result.status != "done":
                err = f": {result.error.message}" if result.error else ""
                return f"place_entity {cube_id} on {pad_id} failed{err}"
            return f"Placed {color} cube on {pad_id}"

        return f"ERROR: unknown tool {name!r}"

    async def run(self, task: str, *, max_turns: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(self.tools)},
            {"role": "user", "content": task},
        ]
        self.tool_log = []

        for turn in range(1, max_turns + 1):
            reply = call_llm(messages, api_key=self.tokamak_api_key)
            messages.append({"role": "assistant", "content": reply})
            call = parse_tool_call(reply) or {
                "tool": "error",
                "args": {"message": "Could not parse tool call.", "raw_reply": reply[:200]},
            }
            tool_name = call["tool"]
            tool_args = call.get("args", {})
            history_chars = sum(len(str(message["content"])) for message in messages)
            print(f"turn {turn} | tool={tool_name} | args={tool_args} | history~{history_chars:,} chars")

            if tool_name == "done":
                print(f"Agent done: {tool_args.get('summary', '')}")
                break
            if tool_name == "error":
                result = f"parse error: {tool_args.get('message', '')}"
            else:
                result = await self.execute_tool(tool_name, tool_args)
            self.tool_log.append({"turn": turn, "tool": tool_name, "result": result})
            print(f"  -> {result[:160]}")
            messages.append({"role": "user", "content": result})

        return messages, self.tool_log

RobotAgent = WorkshopAgent

