# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

import math
from typing import Any

from menlo_runner.perception import perceive


def angle_error_deg(robot_xy: tuple[float, float], yaw_deg: float, target_xy: tuple[float, float]) -> float:
    bearing = math.degrees(math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0]))
    return (bearing - yaw_deg + 180) % 360 - 180


async def turn_to_face(ctx: Any, target_pos: Any, *, tolerance_deg: float = 5.0) -> bool:
    state = await ctx.state("robot_status")
    robot_xy = (state.robot.pose.position[0], state.robot.pose.position[1])
    yaw = state.robot.pose.yaw_deg
    target_xy = (target_pos[0], target_pos[1])
    error = angle_error_deg(robot_xy, yaw, target_xy)
    print(f"Heading error: {error:+.1f} deg")
    if abs(error) < tolerance_deg:
        return True

    wz = 0.4 if error > 0 else -0.4
    duration = min(abs(error) / (abs(wz) * 57.296), 8.0)
    await ctx.invoke(
        "set_velocity",
        {"vx": 0.15, "vy": 0.0, "wz": wz, "duration_s": duration},
        timeout_s=30,
    )
    return abs(error) < 30.0


async def drive_to_distance(
    ctx: Any,
    target_pos: Any,
    *,
    tolerance_m: float = 0.6,
    max_steps: int = 20,
) -> bool:
    for step in range(1, max_steps + 1):
        state = await ctx.state("robot_status")
        rx, ry = state.robot.pose.position[0], state.robot.pose.position[1]
        distance = math.hypot(target_pos[0] - rx, target_pos[1] - ry)
        print(f"Drive step {step}: distance={distance:.2f}m")
        if distance <= tolerance_m:
            return True
        vx = min(0.8, distance * 0.4)
        await ctx.invoke(
            "set_velocity",
            {"vx": vx, "vy": 0.0, "wz": 0.0, "duration_s": 0.8},
            timeout_s=30,
        )
    return False


async def my_go_to_global(
    ctx: Any,
    entity_id: str,
    *,
    tolerance_m: float = 0.6,
    max_iters: int = 10,
) -> bool:
    scene = await ctx.state("scene_state")
    entity = scene.entities.get(entity_id)
    if entity is None:
        print(f"Entity {entity_id!r} not found.")
        return False
    target_pos = entity.pose.position

    for iteration in range(1, max_iters + 1):
        state = await ctx.state("robot_status")
        rx, ry = state.robot.pose.position[0], state.robot.pose.position[1]
        distance = math.hypot(target_pos[0] - rx, target_pos[1] - ry)
        print(f"Global nav iter {iteration}: distance={distance:.2f}m to {entity_id}")
        if distance <= tolerance_m:
            return True
        await turn_to_face(ctx, target_pos)
        if await drive_to_distance(ctx, target_pos, tolerance_m=tolerance_m):
            return True
    return False


async def center_on_color(
    ctx: Any,
    target_color: str,
    *,
    angle_tolerance_deg: float = 10.0,
    max_steps: int = 12,
) -> bool:
    for step in range(1, max_steps + 1):
        obs = await perceive(ctx)
        if target_color not in obs:
            print(f"Center step {step}: {target_color} not visible; scanning")
            await ctx.invoke(
                "set_velocity",
                {"vx": 0.15, "vy": 0.0, "wz": 0.3, "duration_s": 1.0},
                timeout_s=15,
            )
            continue

        angle = float(obs[target_color]["angle_deg"])
        print(f"Center step {step}: {target_color} at {angle:+.1f} deg")
        if abs(angle) <= angle_tolerance_deg:
            return True

        wz = -0.3 if angle > 0 else 0.3
        await ctx.invoke(
            "set_velocity",
            {"vx": 0.15, "vy": 0.0, "wz": wz, "duration_s": 0.8},
            timeout_s=15,
        )
    return False


async def drive_toward_color(
    ctx: Any,
    target_color: str,
    *,
    arrival_area: int = 8000,
    max_steps: int = 15,
) -> bool:
    for step in range(1, max_steps + 1):
        obs = await perceive(ctx)
        if target_color not in obs:
            print(f"Approach step {step}: {target_color} not visible")
            return False

        area = int(obs[target_color]["blob_area"])
        angle = float(obs[target_color]["angle_deg"])
        print(f"Approach step {step}: area={area}px^2 angle={angle:+.1f} deg")
        if area >= arrival_area:
            return True

        if step % 2 == 0 and abs(angle) > 10.0:
            wz = -0.3 if angle > 0 else 0.3
            args = {"vx": 0.15, "vy": 0.0, "wz": wz, "duration_s": 0.5}
        else:
            args = {"vx": 0.5, "vy": 0.0, "wz": 0.0, "duration_s": 1.0}
        await ctx.invoke("set_velocity", args, timeout_s=15)
    return False


async def my_go_to_visual(ctx: Any, target_color: str, *, arrival_area: int = 8000) -> bool:
    print(f"Vision nav to {target_color} cube")
    if not await center_on_color(ctx, target_color):
        return False
    return await drive_toward_color(ctx, target_color, arrival_area=arrival_area)

