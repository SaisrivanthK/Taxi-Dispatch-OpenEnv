import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import Action, ActionType, TASKS, TaxiEnv
from baseline.agent import HeuristicAgent


@pytest.fixture
def env():
    return TaxiEnv()


class TestOpenEnvAPI:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="ride_matching", seed=1)
        assert hasattr(obs, "waiting_riders")
        assert hasattr(obs, "available_cars")
        assert hasattr(obs, "driver_status")
        assert obs.current_time == 0.0

    def test_step_returns_step_result(self, env):
        env.reset(task_id="ride_matching", seed=1)
        result = env.step(Action(action_type=ActionType.NO_OP))
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "truncated")
        assert isinstance(result.reward, float)

    def test_state_returns_dict(self, env):
        env.reset(task_id="ride_matching", seed=1)
        s = env.state()
        assert isinstance(s, dict)
        assert s["env_id"] == TaxiEnv.ENV_ID

    def test_score_in_range(self, env):
        env.reset(task_id="ride_matching", seed=1)
        for _ in range(10):
            env.step(Action(action_type=ActionType.NO_OP))
        s = env.score()
        assert 0.0 <= s <= 1.0

    def test_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset(task_id="nope")

    def test_step_before_reset_raises(self):
        fresh = TaxiEnv()
        with pytest.raises(RuntimeError):
            fresh.step(Action(action_type=ActionType.NO_OP))


class TestAllTasks:
    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_full_episode_completes(self, task_id):
        env = TaxiEnv()
        env.reset(task_id=task_id, seed=42)
        done = truncated = False
        steps = 0
        while not (done or truncated):
            result = env.step(Action(action_type=ActionType.NO_OP))
            done, truncated = result.done, result.truncated
            steps += 1
        assert steps <= TASKS[task_id].max_steps + 1
        assert 0.0 <= env.score() <= 1.0

    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_heuristic_beats_noop(self, task_id):
        def run(use_heuristic):
            env = TaxiEnv()
            agent = HeuristicAgent(env) if use_heuristic else None
            env.reset(task_id=task_id, seed=99)
            done = truncated = False
            while not (done or truncated):
                action = agent.act(None) if agent else Action(action_type=ActionType.NO_OP)
                result = env.step(action)
                done, truncated = result.done, result.truncated
            return env.score()

        noop_score = run(False)
        heuristic_score = run(True)
        assert heuristic_score >= noop_score, f"[{task_id}] heuristic={heuristic_score:.4f} < noop={noop_score:.4f}"


class TestRewards:
    def test_correct_assignment_positive_reward(self, env):
        env.reset(task_id="ride_matching", seed=7)
        for _ in range(3):
            env.step(Action(action_type=ActionType.NO_OP))
        waiting = env.waiting_riders()
        if not waiting:
            pytest.skip("No riders available yet")
        rider = waiting[0]
        car = next(
            (
                c for c in env.available_cars()
                if (
                    (rider.tier.value == 'vip' and c.car_type.value == 'armored')
                    or (rider.group_size >= 4 and c.car_type.value == 'xuv')
                    or (rider.tier.value == 'premium' and rider.group_size < 4 and c.car_type.value == 'sedan')
                    or (rider.tier.value == 'normal' and rider.group_size < 4 and c.car_type.value == 'basic')
                )
            ),
            env.available_cars()[0],
        )
        driver = env.available_drivers()[0]
        result = env.step(Action(
            action_type=ActionType.ASSIGN,
            rider_id=rider.rider_id,
            car_id=car.car_id,
            driver_id=driver.driver_id,
        ))
        assert result.reward > 0

    def test_wrong_tier_car_gives_lower_reward(self, env):
        env.reset(task_id="ride_matching", seed=7)
        for _ in range(3):
            env.step(Action(action_type=ActionType.NO_OP))
        waiting = env.waiting_riders()
        if not waiting:
            pytest.skip("No riders available yet")
        rider = waiting[0]
        car = next(
            (
                c for c in env.available_cars()
                if not (
                    (rider.tier.value == 'vip' and c.car_type.value == 'armored')
                    or (rider.group_size >= 4 and c.car_type.value == 'xuv')
                    or (rider.tier.value == 'premium' and rider.group_size < 4 and c.car_type.value == 'sedan')
                    or (rider.tier.value == 'normal' and rider.group_size < 4 and c.car_type.value == 'basic')
                )
            ),
            env.available_cars()[0],
        )
        driver = env.available_drivers()[0]
        result = env.step(Action(
            action_type=ActionType.ASSIGN,
            rider_id=rider.rider_id,
            car_id=car.car_id,
            driver_id=driver.driver_id,
        ))
        assert result.reward <= 0.25


class TestGraders:
    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_grader_range(self, task_id):
        env = TaxiEnv()
        agent = HeuristicAgent(env)
        env.reset(task_id=task_id, seed=42)
        done = truncated = False
        while not (done or truncated):
            result = env.step(agent.act(None))
            done, truncated = result.done, result.truncated
        score = env.score()
        assert 0.0 <= score <= 1.0

    def test_difficulty_ordering(self):
        scores = {}
        for tid in TASKS:
            env = TaxiEnv()
            agent = HeuristicAgent(env)
            env.reset(task_id=tid, seed=42)
            done = truncated = False
            while not (done or truncated):
                result = env.step(agent.act(None))
                done, truncated = result.done, result.truncated
            scores[tid] = env.score()
        assert scores["ride_matching"] >= scores["surge_mobility"] - 0.25
