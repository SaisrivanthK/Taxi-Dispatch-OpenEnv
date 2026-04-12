from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .taxi_env import TaxiEnv


@dataclass
class TaskConfig:
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    sim_duration_seconds: float
    n_cars: int
    n_drivers: int
    arrival_rate: float
    enable_traffic: bool
    enable_pooling: bool
    enable_repositioning: bool
    reward_weights: dict[str, float]


TASKS: dict[str, TaskConfig] = {
    "ride_matching": TaskConfig(
        task_id="ride_matching",
        name="Ride Matching",
        description="Low-pressure dispatch with simple vehicle matching and minimal cancellations.",
        difficulty="easy",
        max_steps=50,
        sim_duration_seconds=3000.0,
        n_cars=8,
        n_drivers=8,
        arrival_rate=0.35,
        enable_traffic=False,
        enable_pooling=False,
        enable_repositioning=False,
        reward_weights={
            "match_rate": 0.50,
            "completion_rate": 0.30,
            "wait_efficiency": 0.20,
        },
    ),
    "dispatch_allocation": TaskConfig(
        task_id="dispatch_allocation",
        name="Dispatch Allocation",
        description="Moderate demand with traffic, cancellations, and mixed vehicle constraints.",
        difficulty="medium",
        max_steps=150,
        sim_duration_seconds=9000.0,
        n_cars=12,
        n_drivers=12,
        arrival_rate=0.60,
        enable_traffic=True,
        enable_pooling=False,
        enable_repositioning=True,
        reward_weights={
            "match_rate": 0.25,
            "completion_rate": 0.25,
            "wait_efficiency": 0.15,
            "cancellation_rate": 0.20,
            "utilization_rate": 0.15,
        },
    ),
    "surge_mobility": TaskConfig(
        task_id="surge_mobility",
        name="Surge Mobility",
        description="High-demand city surge with pooling and driver repositioning required.",
        difficulty="hard",
        max_steps=300,
        sim_duration_seconds=18000.0,
        n_cars=16,
        n_drivers=16,
        arrival_rate=0.85,
        enable_traffic=True,
        enable_pooling=True,
        enable_repositioning=True,
        reward_weights={
            "match_rate": 0.20,
            "completion_rate": 0.20,
            "wait_efficiency": 0.15,
            "cancellation_rate": 0.15,
            "utilization_rate": 0.10,
            "pooling_rate": 0.10,
            "reposition_rate": 0.10,
        },
    ),
}


class BaseGrader:
    def __init__(self, task: TaskConfig):
        self.task = task

    def score(self, env: "TaxiEnv") -> float:
        raise NotImplementedError

    @staticmethod
    def clamp(x: float) -> float:
        # Ensure STRICT (0,1) range
        eps = 1e-6
        return max(eps, min(1.0 - eps, x))


class RideMatchingGrader(BaseGrader):
    def score(self, env: "TaxiEnv") -> float:
        m = env.episode_metrics
        w = self.task.reward_weights

        score = (
            w["match_rate"] * m.get("match_rate", 0.0)
            + w["completion_rate"] * m.get("completion_rate", 0.0)
            + w["wait_efficiency"] * m.get("wait_efficiency", 0.0)
        )

        return self.clamp(round(score, 4))


class DispatchAllocationGrader(BaseGrader):
    def score(self, env: "TaxiEnv") -> float:
        m = env.episode_metrics
        w = self.task.reward_weights

        score = (
            w["match_rate"] * m.get("match_rate", 0.0)
            + w["completion_rate"] * m.get("completion_rate", 0.0)
            + w["wait_efficiency"] * m.get("wait_efficiency", 0.0)
            + w["cancellation_rate"] * m.get("cancellation_rate", 0.0)
            + w["utilization_rate"] * m.get("utilization_rate", 0.0)
        )

        return self.clamp(round(score, 4))


class SurgeMobilityGrader(BaseGrader):
    def score(self, env: "TaxiEnv") -> float:
        m = env.episode_metrics
        w = self.task.reward_weights

        score = (
            w["match_rate"] * m.get("match_rate", 0.0)
            + w["completion_rate"] * m.get("completion_rate", 0.0)
            + w["wait_efficiency"] * m.get("wait_efficiency", 0.0)
            + w["cancellation_rate"] * m.get("cancellation_rate", 0.0)
            + w["utilization_rate"] * m.get("utilization_rate", 0.0)
            + w["pooling_rate"] * m.get("pooling_rate", 0.0)
            + w["reposition_rate"] * m.get("reposition_rate", 0.0)
        )

        return self.clamp(round(score, 4))


GRADERS = {
    "ride_matching": RideMatchingGrader,
    "dispatch_allocation": DispatchAllocationGrader,
    "surge_mobility": SurgeMobilityGrader,
}


def get_grader(task_id: str) -> BaseGrader:
    task = TASKS[task_id]
    return GRADERS[task_id](task)