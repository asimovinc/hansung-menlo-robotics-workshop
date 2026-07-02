import asyncio
import unittest
from types import SimpleNamespace

from menlo_runner.scene import delivered_cube_ids


def cube_state(
    *,
    visible: bool,
    parent_pad_id: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        visible=visible,
        state={"parent_pad_id": parent_pad_id},
    )


class FakeSceneContext:
    def __init__(self, entities) -> None:
        self.entities = entities

    async def state(self, name: str):
        if name != "scene_state":
            raise AssertionError(f"unexpected state read: {name}")
        return SimpleNamespace(entities=self.entities)


class DeliveredCubeIdsTest(unittest.TestCase):
    def test_ignores_hidden_pool_cubes_that_were_not_delivered(self):
        ctx = FakeSceneContext(
            {
                "cube_0": cube_state(visible=True, parent_pad_id="pad_A"),
                "cube_pool_0": cube_state(visible=False, parent_pad_id=None),
                "robot": SimpleNamespace(visible=True, state=None),
            }
        )

        self.assertEqual(asyncio.run(delivered_cube_ids(ctx)), [])

    def test_counts_invisible_cubes_on_destination_pads(self):
        ctx = FakeSceneContext(
            {
                "cube_0": cube_state(visible=False, parent_pad_id="pad_D"),
                "cube_pool_0": cube_state(visible=False, parent_pad_id="pad_E"),
                "cube_1": cube_state(visible=False, parent_pad_id="pad_A"),
            }
        )

        self.assertEqual(asyncio.run(delivered_cube_ids(ctx)), ["cube_0", "cube_pool_0"])


if __name__ == "__main__":
    unittest.main()
