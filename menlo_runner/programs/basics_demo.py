# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

from menlo_runner.basics import print_position, screenshot
from menlo_runner.scene import get_scene_text


async def run(ctx) -> None:
    print("Robot status:")
    await print_position(ctx, "CURRENT")

    print("\nScene summary:")
    print(await get_scene_text(ctx))

    await screenshot(ctx, "Saved current POV:", "outputs/basics-demo-pov.jpg")

    print("\nSmall SDK movement demo:")
    await ctx.invoke("set_velocity", {"vx": 0.4, "vy": 0.0, "wz": 0.0, "duration_s": 1.0})
    await print_position(ctx, "AFTER")

