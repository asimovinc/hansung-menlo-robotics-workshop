from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from menlo_robot_sdk import AsyncClient, connect
from menlo_robot_sdk.experimental import generate_room_key

from menlo_runner.config import MenloConfig


@dataclass
class RobotContext:
    config: MenloConfig
    client: AsyncClient
    session: Any
    robot_id: str
    viewer_url: str

    @classmethod
    async def create(
        cls,
        config: MenloConfig,
        *,
        name_prefix: str = "sdk-program",
        model: str = "asimov-v0",
        join_livekit: bool = True,
    ) -> "RobotContext":
        client = AsyncClient(rcs_url=config.rcs_url, api_key=config.menlo_api_key)
        created = await client.robots.create(
            name=f"{name_prefix}-{int(time.time())}",
            model=model,
        )
        robot_id = created.robot.id
        session = await connect(
            client,
            robot_id,
            worker_names=[],
            rcw_identity_prefix="simplesim",
            join_livekit=join_livekit,
        )
        room_key = await generate_room_key(client, robot_id)
        viewer_url = f"{config.viewer_base_url}/?key={room_key}"
        return cls(config, client, session, robot_id, viewer_url)

    async def wait_for_skills(self, timeout_s: float = 180.0) -> list[Any]:
        """Poll until the browser viewer joins and exposes robot skills."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                found = await self.session.discover_skills()
                if found:
                    return found
            except (RuntimeError, TimeoutError):
                pass
            await asyncio.sleep(2.0)
        raise TimeoutError("Viewer did not join. Open the viewer URL in Chrome.")

    async def invoke(
        self,
        skill_name: str,
        args: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        if timeout_s is None:
            return await self.session.invoke(skill_name, args)
        return await self.session.invoke(skill_name, args, timeout_s=timeout_s)

    async def state(self, key: str) -> Any:
        return await self.session.state.get(key)

    async def get_vision(self, camera: str = "pov") -> bytes:
        return await self.session.get_vision(camera)

    async def save_screenshot(self, path: str | Path, camera: str = "pov") -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(await self.get_vision(camera))
        return out

    async def close(self, *, delete_robot: bool = True) -> None:
        try:
            await self.session.disconnect()
        finally:
            if delete_robot:
                await self.client.robots.delete(self.robot_id)
            await self.client.aclose()


async def open_robot_context(
    config: MenloConfig,
    *,
    name_prefix: str = "sdk-program",
) -> RobotContext:
    ctx = await RobotContext.create(config, name_prefix=name_prefix)
    print(f"Created robot: {ctx.robot_id}")
    print("=" * 60)
    print(f"VIEWER URL: {ctx.viewer_url}")
    print("=" * 60)
    input("Open the viewer URL in Chrome, wait for the warehouse to load, then press Enter...")
    skills = await ctx.wait_for_skills()
    print("Skills found:")
    for skill in skills:
        print(f"  - {skill.name}")
    return ctx


