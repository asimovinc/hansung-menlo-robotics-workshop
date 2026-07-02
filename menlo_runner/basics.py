# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

from pathlib import Path
from typing import Any

async def print_position(ctx: Any, label: str = "") -> None:
    state = await ctx.state("robot_status")
    p = state.robot.pose.position
    print(
        f"{label:20s} pos=({p[0]:+.2f}, {p[1]:+.2f}) "
        f"yaw={state.robot.pose.yaw_deg:+.1f} deg status={state.robot.status}"
    )


async def screenshot(ctx: Any, label: str = "", path: str | Path | None = None) -> bytes:
    jpeg = await ctx.get_vision("pov")
    if label:
        print(label)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg)
        print(f"Saved screenshot: {out}")
    return jpeg


async def set_velocity(ctx: Any, **params: float) -> Any:
    result = await ctx.invoke("set_velocity", params)
    print(f"set_velocity {params} -> {result.status}")
    return result


async def go_to_entity(ctx: Any, entity_id: str, *, timeout_s: float = 300) -> Any:
    target = {"kind": "entity", "entity_id": entity_id}
    result = await ctx.invoke("go_to", {"target": target}, timeout_s=timeout_s)
    print(f"go_to {entity_id} -> {result.status}")
    return result


async def pick_entity(ctx: Any, entity_id: str = "cube", *, timeout_s: float = 300) -> Any:
    target = {"kind": "entity", "entity_id": entity_id}
    result = await ctx.invoke("pick_entity", {"target": target}, timeout_s=timeout_s)
    print(f"pick_entity {entity_id} -> {result.status}")
    return result


async def place_entity(ctx: Any, entity_id: str, *, timeout_s: float = 300) -> Any:
    target = {"kind": "entity", "entity_id": entity_id}
    result = await ctx.invoke("place_entity", {"target": target}, timeout_s=timeout_s)
    print(f"place_entity {entity_id} -> {result.status}")
    return result

