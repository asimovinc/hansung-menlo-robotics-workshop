import asyncio
import unittest
from types import SimpleNamespace

from menlo_runner.completion import (
    CompletionConfig,
    CompletionTimeout,
    CompletionTracker,
    completion_config_for_round,
    level_from_program_name,
    round_time_limit_s,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cube_state(
    *,
    entity_id: str,
    visible: bool,
    attached_to: str | None = None,
    parent_pad_id: str | None = None,
) -> SimpleNamespace:
    state = {}
    if parent_pad_id is not None:
        state["parent_pad_id"] = parent_pad_id
    return SimpleNamespace(
        entity_id=entity_id,
        visible=visible,
        attached_to=attached_to,
        state=state,
    )


class FakeSceneContext:
    def __init__(self, entities=None, robot_status: str = "ready") -> None:
        self.entities = entities or {
            "cube_0": cube_state(entity_id="cube_0", visible=False),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "cube_1": cube_state(entity_id="cube_1", visible=True),
            "robot": SimpleNamespace(visible=True),
        }
        self.robot_status = robot_status

    async def state(self, name: str):
        if name == "scene_state":
            return SimpleNamespace(entities=self.entities)
        if name == "robot_status":
            return SimpleNamespace(robot=SimpleNamespace(status=self.robot_status))
        raise AssertionError(f"unexpected state read: {name}")


class CompletionConfigTest(unittest.TestCase):
    def test_requires_at_least_one_stop_condition(self):
        with self.assertRaises(ValueError):
            CompletionConfig().validate()

    def test_rejects_non_positive_time_limit(self):
        with self.assertRaises(ValueError):
            CompletionConfig(max_elapsed_s=0).validate()

    def test_rejects_negative_cube_limit(self):
        with self.assertRaises(ValueError):
            CompletionConfig(max_delivered_cubes=-1).validate()

    def test_level_sets_delivery_points_without_score_cap(self):
        config = CompletionConfig(level=2, max_elapsed_s=600)

        self.assertEqual(config.delivery_points(), 40)

    def test_round_time_limit_resolves_standard_rounds_and_manual(self):
        self.assertEqual(round_time_limit_s("round1"), 300)
        self.assertEqual(round_time_limit_s("2"), 600)
        self.assertEqual(round_time_limit_s("round3"), 900)
        self.assertEqual(round_time_limit_s("manual", manual_seconds=42), 42)

    def test_completion_config_for_round_caps_at_twelve_deliveries(self):
        config = completion_config_for_round(1, round_name="round3")

        self.assertEqual(config.max_delivered_cubes, 12)
        self.assertEqual(config.max_elapsed_s, 900)


class CompletionTrackerTest(unittest.TestCase):
    def test_elapsed_time_starts_at_first_cycle(self):
        clock = FakeClock()
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=5), clock=clock)

        clock.advance(100)
        self.assertEqual(tracker.elapsed_s(), 0.0)

        tracker.start_first_cycle()
        clock.advance(3)

        self.assertEqual(tracker.elapsed_s(), 3.0)
        self.assertIsNone(tracker.stop_reason(delivered_count=0))

    def test_stops_when_time_limit_is_reached(self):
        clock = FakeClock()
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=5), clock=clock)

        tracker.start_first_cycle()
        clock.advance(5)

        self.assertEqual(tracker.stop_reason(delivered_count=0), "elapsed 5.0/5.0 seconds")
        self.assertEqual(tracker.remaining_s(), 0.0)

    def test_wait_for_remaining_cuts_off_slow_step(self):
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=0.01))
        tracker.start_first_cycle()

        async def slow_step():
            await asyncio.sleep(1)

        with self.assertRaises(CompletionTimeout):
            asyncio.run(tracker.wait_for_remaining(slow_step(), "slow model call"))

    def test_stops_when_delivery_limit_is_reached(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=2))

        tracker.start_first_cycle()

        self.assertIsNone(tracker.stop_reason(delivered_count=1))
        self.assertEqual(tracker.stop_reason(delivered_count=2), "delivered 2/2 cubes")

    def test_delivery_score_uses_level_first_delivery_bonus(self):
        tracker = CompletionTracker(CompletionConfig(level=2, max_elapsed_s=600))

        self.assertEqual(tracker.delivery_score(delivered_count=0), 0)
        self.assertEqual(tracker.delivery_score(delivered_count=1), 60)
        self.assertEqual(tracker.delivery_score(delivered_count=5), 220)
        self.assertEqual(
            tracker.score_description(),
            "60 points for the first delivery, then 40 points per additional delivery",
        )

    def test_scene_delivered_count_ignores_initial_hidden_pool_cubes(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        ctx = FakeSceneContext()

        delivered_count = asyncio.run(tracker.scene_delivered_count(ctx))

        self.assertEqual(delivered_count, 0)

    def test_scene_delivered_count_counts_destination_pad_delivery(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        entities = {
            "cube_0": cube_state(entity_id="cube_0", visible=True),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 0)
        entities["cube_0"] = cube_state(
            entity_id="cube_0",
            visible=False,
            parent_pad_id="pad_B",
        )

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 1)

    def test_scene_delivered_count_ignores_source_pad_and_hidden_pool(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=1))
        entities = {
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "cube_0": cube_state(entity_id="cube_0", visible=False, parent_pad_id="pad_A"),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertEqual(asyncio.run(tracker.scene_delivered_count(ctx)), 0)

    def test_stop_reason_from_scene_uses_destination_pad_count(self):
        tracker = CompletionTracker(CompletionConfig(max_delivered_cubes=2))
        entities = {
            "cube_0": cube_state(entity_id="cube_0", visible=True),
            "cube_1": cube_state(entity_id="cube_1", visible=True),
            "cube_pool_0": cube_state(entity_id="cube_pool_0", visible=False),
            "robot": SimpleNamespace(visible=True),
        }
        ctx = FakeSceneContext(entities)

        self.assertIsNone(asyncio.run(tracker.stop_reason_from_scene(ctx)))
        entities["cube_0"] = cube_state(
            entity_id="cube_0",
            visible=False,
            parent_pad_id="pad_C",
        )
        self.assertIsNone(asyncio.run(tracker.stop_reason_from_scene(ctx)))
        entities["cube_1"] = cube_state(
            entity_id="cube_1",
            visible=False,
            parent_pad_id="pad_E",
        )
        reason = asyncio.run(tracker.stop_reason_from_scene(ctx))

        self.assertEqual(reason, "delivered 2/2 cubes")

    def test_stop_reason_from_scene_reports_fallen_with_remaining_time(self):
        clock = FakeClock()
        tracker = CompletionTracker(CompletionConfig(max_elapsed_s=100), clock=clock)
        tracker.start_first_cycle()
        clock.advance(12)
        ctx = FakeSceneContext(robot_status="fallen")

        reason = asyncio.run(tracker.stop_reason_from_scene(ctx))

        self.assertEqual(reason, "robot fallen; remaining 88.0 seconds")


class ProgramLevelTest(unittest.TestCase):
    def test_infers_level_from_program_name(self):
        self.assertEqual(level_from_program_name("level-0-starter"), 0)
        self.assertEqual(level_from_program_name("level-1-starter-ko"), 1)
        self.assertEqual(level_from_program_name("menlo_runner.programs.project.en.level_2_starter"), 2)
        self.assertIsNone(level_from_program_name("student-program"))


if __name__ == "__main__":
    unittest.main()
