# Starter code: do not edit this shared file for project submissions.
# Put project code in a project notebook or a new file under menlo_runner/programs/.

from __future__ import annotations

from menlo_runner.agents import WorkshopAgent


TASK = (
    "Use get_scene_summary to find a visible cube, go to it, pick it up, "
    "check what you are holding, and place it on the correct pad. "
    "Call done after one successful delivery or if you cannot continue."
)


async def run(ctx) -> None:
    agent = WorkshopAgent(ctx, tokamak_api_key=ctx.config.tokamak_api_key)
    _messages, tool_log = await agent.run(TASK, max_turns=12)
    print("\nTool log:")
    for entry in tool_log:
        print(f"  turn {entry['turn']}: {entry['tool']} -> {entry['result'][:80]}")

