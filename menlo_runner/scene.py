from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


COLOR_TO_PAD = {"green": "pad_C", "blue": "pad_D", "red": "pad_B", "yellow": "pad_E"}


@dataclass(frozen=True)
class CubeInfo:
    entity_id: str
    color: str
    position: tuple[float, float, float]
    distance_from_robot: float


async def get_scene(ctx: Any) -> Any:
    return await ctx.state("scene_state")


async def get_robot_status(ctx: Any) -> Any:
    return await ctx.state("robot_status")


async def get_scene_text(ctx: Any) -> str:
    """Compact text snapshot of the warehouse for people or an LLM agent."""
    scene = await get_scene(ctx)
    lines: list[str] = []
    for eid, entity in sorted(scene.entities.items()):
        pose = getattr(entity, "pose", None)
        if not pose:
            continue
        p = pose.position
        if eid == "robot":
            lines.append(f"robot at ({p[0]:+.2f}, {p[1]:+.2f}), facing yaw={pose.yaw_deg:.0f} deg")
        elif eid.startswith("pad_"):
            lines.append(f"{eid} at ({p[0]:+.2f}, {p[1]:+.2f})")
        elif eid.startswith("cube_") and entity.visible:
            color = entity.state.get("color", "?") if entity.state else "?"
            held = " (HELD BY ROBOT)" if entity.attached_to else ""
            lines.append(f"{eid} color={color} at ({p[0]:+.2f}, {p[1]:+.2f}){held}")
    return "\n".join(lines)


async def visible_cubes(ctx: Any, color: str | None = None) -> list[CubeInfo]:
    scene = await get_scene(ctx)
    robot_position = scene.entities["robot"].pose.position
    cubes: list[CubeInfo] = []

    for eid, entity in scene.entities.items():
        if not eid.startswith("cube_") or not entity.visible:
            continue
        cube_color = entity.state.get("color", "?") if entity.state else "?"
        if color is not None and cube_color != color:
            continue
        p = entity.pose.position
        distance = hypot(p[0] - robot_position[0], p[1] - robot_position[1])
        cubes.append(CubeInfo(eid, cube_color, tuple(p), distance))

    return sorted(cubes, key=lambda cube: cube.distance_from_robot)


async def held_cube_info(ctx: Any) -> tuple[str, str] | None:
    scene = await get_scene(ctx)
    for eid, entity in scene.entities.items():
        if eid.startswith("cube_") and entity.attached_to:
            color = entity.state.get("color", "?") if entity.state else "?"
            return eid, color
    return None


async def delivered_cube_ids(ctx: Any) -> list[str]:
    scene = await get_scene(ctx)
    return [
        eid
        for eid, entity in scene.entities.items()
        if eid.startswith("cube_")
        and not entity.visible
        and not str(eid).startswith("cube_pool")
    ]


