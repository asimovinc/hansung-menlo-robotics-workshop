from __future__ import annotations

from menlo_runner.perception import perceive
from menlo_runner.navigation import my_go_to_global, my_go_to_visual


async def run(ctx) -> None:
    print("Part A: custom global-state navigation to pad_C")
    reached = await my_go_to_global(ctx, "pad_C", tolerance_m=0.8, max_iters=3)
    print(f"Global navigation result: {reached}")

    print("\nPart B: vision-only navigation to the first visible cube color")
    obs = await perceive(ctx)
    if not obs:
        print("No visible cube colors. Move/reset the robot near the conveyor and try again.")
        return
    target_color = next(iter(obs))
    reached = await my_go_to_visual(ctx, target_color)
    print(f"Vision navigation result for {target_color}: {reached}")

