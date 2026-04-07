from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from .models import (
    Action,
    ActionType,
    CarState,
    CarType,
    DriverState,
    Observation,
    RiderState,
    RiderView,
    CarView,
    DriverView,
    StepResult,
    Tier,
)
from .tasks import TASKS, get_grader


TIER_PRIORITY = {
    Tier.VIP: 0,
    Tier.PREMIUM: 1,
    Tier.NORMAL: 2,
}


class TaxiEnv:
    ENV_ID = "taxi-dispatch-v1"
    ENV_VERSION = "1.0.0"

    ZONES = [0, 1, 2, 3, 4]

    def __init__(self) -> None:
        self._task = None
        self._rng = random.Random()
        self._reset_state()

    def reset(self, task_id: str = "ride_matching", seed: int | None = None) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS)}")
        self._task = TASKS[task_id]
        self._rng = random.Random(seed)
        self._reset_state()
        self._init_fleet()
        return self._build_observation()

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        if isinstance(action, dict):
            action = Action(**action)

        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        prev_metrics = dict(self._episode_metrics)
        events: list[str] = []
        reward_delta = 0.0

        reward_delta += self._process_trip_completions(events)

        self._sim_time += self._step_size
        self._step_count += 1

        self._update_traffic()
        events.extend(self._process_arrivals())
        self._process_waiting_and_cancellations(events)

        reward_delta += self._execute_action(action, events)
        self._update_driver_time_counters()
        self._update_metrics(prev_metrics)

        step_reward = self._compute_step_reward(reward_delta)
        self._cumulative_reward += step_reward

        done = self._sim_time >= self._task.sim_duration_seconds
        truncated = self._step_count >= self._task.max_steps
        obs = self._build_observation(events)
        info = {
            "step": self._step_count,
            "sim_time": self._sim_time,
            "episode_metrics": dict(self._episode_metrics),
            "cumulative_reward": self._cumulative_reward,
        }
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def state(self) -> dict[str, Any]:
        return {
            "env_id": self.ENV_ID,
            "task_id": self._task.task_id if self._task else None,
            "sim_time": self._sim_time,
            "step": self._step_count,
            "riders": {rid: asdict(r) for rid, r in self._riders.items()},
            "cars": {cid: asdict(c) for cid, c in self._cars.items()},
            "drivers": {did: asdict(d) for did, d in self._drivers.items()},
            "metrics": dict(self.episode_metrics),
            "traffic_factor": self._traffic_factor,
        }

    def score(self) -> float:
        if self._task is None:
            return 0.0
        return get_grader(self._task.task_id).score(self)

    def waiting_riders(self) -> list[RiderState]:
        riders = [r for r in self._riders.values() if r.waiting]
        riders.sort(key=lambda r: (TIER_PRIORITY[r.tier], -r.wait_time, -r.group_size))
        return riders

    def available_cars(self) -> list[CarState]:
        return [c for c in self._cars.values() if c.available]

    def available_drivers(self) -> list[DriverState]:
        return [d for d in self._drivers.values() if d.available and not d.on_break]

    @property
    def episode_metrics(self) -> dict[str, float]:
        return self._episode_metrics

    @property
    def current_time(self) -> float:
        return self._sim_time

    @property
    def _step_size(self) -> float:
        return self._task.sim_duration_seconds / self._task.max_steps

    def _reset_state(self) -> None:
        self._sim_time = 0.0
        self._step_count = 0
        self._riders: dict[str, RiderState] = {}
        self._cars: dict[str, CarState] = {}
        self._drivers: dict[str, DriverState] = {}
        self._traffic_factor = 1.0
        self._cumulative_reward = 0.0
        self._episode_metrics = {
            "total_arrived": 0,
            "completed": 0,
            "cancelled": 0,
            "unsafe_matches": 0,
            "match_rate": 0.0,
            "completion_rate": 0.0,
            "wait_efficiency": 0.0,
            "cancellation_rate": 0.0,
            "utilization_rate": 0.0,
            "pooling_rate": 0.0,
            "reposition_rate": 0.0,
            "safe_drop_rate": 0.0,
            "avg_wait_time": 0.0,
            "wait_time_sum": 0.0,
            "wait_count": 0,
            "assignments": 0,
            "matched_assignments": 0,
            "pool_actions": 0,
            "successful_pools": 0,
            "repositions": 0,
            "successful_repositions": 0,
            "driver_busy_time": 0.0,
            "driver_total_time": 0.0,
            "car_busy_time": 0.0,
            "car_total_time": 0.0,
            "safe_completed": 0,
        }

    def _init_fleet(self) -> None:
        task = self._task
        car_types = (
            [CarType.BASIC] * max(1, task.n_cars // 4)
            + [CarType.SEDAN] * max(1, task.n_cars // 4)
            + [CarType.XUV] * max(1, task.n_cars // 4)
            + [CarType.ARMORED] * max(1, task.n_cars - 3 * max(1, task.n_cars // 4))
        )
        for i in range(task.n_cars):
            ctype = car_types[i % len(car_types)]
            capacity = {CarType.BASIC: 2, CarType.SEDAN: 4, CarType.XUV: 6, CarType.ARMORED: 4}[ctype]
            zone = self._rng.choice(self.ZONES)
            self._cars[f"C{i+1:03d}"] = CarState(
                car_id=f"C{i+1:03d}",
                car_type=ctype,
                capacity=capacity,
                location_zone=zone,
                driver_id=None,
                available=True,
            )
        for i in range(task.n_drivers):
            zone = self._rng.choice(self.ZONES)
            self._drivers[f"D{i+1:03d}"] = DriverState(
                driver_id=f"D{i+1:03d}",
                zone=zone,
                available=True,
            )
        driver_ids = list(self._drivers.keys())
        for car, did in zip(self._cars.values(), driver_ids):
            car.driver_id = did
            self._drivers[did].zone = car.location_zone

    def _spawn_rider(self) -> None:
        idx = len(self._riders) + 1
        tier = self._rng.choices([Tier.NORMAL, Tier.PREMIUM, Tier.VIP], weights=[0.6, 0.25, 0.15])[0]
        urgency = "urgent" if (tier == Tier.VIP or self._rng.random() < 0.25) else "normal"
        group_size = self._rng.choices([1, 2, 3, 4, 5, 6], weights=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])[0]
        if group_size >= 4 and tier == Tier.NORMAL:
            tier = Tier.PREMIUM if self._rng.random() < 0.5 else Tier.NORMAL
        pickup_zone = self._rng.choice(self.ZONES)
        drop_zone = self._rng.choice([z for z in self.ZONES if z != pickup_zone])
        share_allowed = self._rng.random() < (0.8 if tier != Tier.VIP else 0.4)
        rid = f"R{idx:04d}"
        self._riders[rid] = RiderState(
            rider_id=rid,
            tier=tier,
            urgency=urgency,
            group_size=group_size,
            pickup_zone=pickup_zone,
            drop_zone=drop_zone,
            share_allowed=share_allowed,
            arrival_time=self._sim_time,
        )
        self._episode_metrics["total_arrived"] += 1

    def _arrival_prob(self) -> float:
        return self._task.arrival_rate

    def _process_arrivals(self) -> list[str]:
        events = []
        if self._rng.random() < self._arrival_prob():
            self._spawn_rider()
            events.append("New ride request arrived.")
            if self._task.difficulty == "hard" and self._rng.random() < 0.3:
                self._spawn_rider()
                events.append("Surge: second ride request arrived.")
        return events

    def _update_traffic(self) -> None:
        if not self._task.enable_traffic:
            self._traffic_factor = 1.0
            return
        if self._task.difficulty == "easy":
            self._traffic_factor = 1.0
        elif self._task.difficulty == "medium":
            self._traffic_factor = round(self._rng.uniform(0.9, 1.4), 2)
        else:
            spike = 1.0 + (0.8 if self._step_count % 25 < 10 else 0.0)
            self._traffic_factor = round(self._rng.uniform(0.9, 1.6) + spike * 0.25, 2)

    def _process_waiting_and_cancellations(self, events: list[str]) -> None:
        threshold = 8 if self._task.difficulty == "easy" else (6 if self._task.difficulty == "medium" else 4)
        wait_penalty_sum = 0.0

        for rider in self._riders.values():
            if rider.waiting:
                rider.wait_time += self._step_size
                self._episode_metrics["wait_time_sum"] += rider.wait_time
                self._episode_metrics["wait_count"] += 1
                wait_penalty_sum += 0.0003 * (1.0 if rider.urgency == "normal" else 1.7)
                if rider.wait_time > threshold * self._step_size:
                    rider.cancelled = True
                    self._episode_metrics["cancelled"] += 1
                    events.append(f"Cancelled: {rider.rider_id} after waiting too long.")

        self._cumulative_reward -= min(0.03, wait_penalty_sum)

    def _execute_action(self, action: Action, events: list[str]) -> float:
        if action.action_type == ActionType.NO_OP:
            return 0.0
        if action.action_type == ActionType.WAIT:
            return -0.005
        if action.action_type == ActionType.REPOSITION:
            return self._action_reposition(action, events)
        if action.action_type == ActionType.POOL:
            return self._action_pool(action, events)
        if action.action_type == ActionType.ASSIGN:
            return self._action_assign(action, events)
        return -0.01

    def _action_reposition(self, action: Action, events: list[str]) -> float:
        if not self._task.enable_repositioning:
            return -0.01
        driver = self._drivers.get(action.driver_id or "")
        if driver is None or not driver.available or driver.on_break:
            return -0.02
        target = action.target_zone
        if target is None or target not in self.ZONES:
            return -0.02

        self._episode_metrics["repositions"] += 1
        driver.zone = target
        car = next((c for c in self._cars.values() if c.driver_id == driver.driver_id), None)
        if car is not None:
            car.location_zone = target
        demand_zones = self._demand_heatmap()
        success = demand_zones and target == demand_zones[0][0]
        if success:
            self._episode_metrics["successful_repositions"] += 1
            events.append(f"Repositioned driver {driver.driver_id} to hot zone {target}.")
            return 0.06
        events.append(f"Repositioned driver {driver.driver_id} to zone {target}.")
        return 0.01

    def _action_assign(self, action: Action, events: list[str]) -> float:
        rider = self._riders.get(action.rider_id or "")
        car = self._cars.get(action.car_id or "")
        if rider is None or car is None:
            return -0.02
        driver = self._drivers.get(action.driver_id or car.driver_id or "")
        if driver is None or not car.available or not driver.available or driver.on_break:
            return -0.03
        if not rider.waiting:
            return -0.02

        self._episode_metrics["assignments"] += 1
        tier_ok = self._tier_match(rider, car)
        cap_ok = rider.group_size <= car.capacity
        if tier_ok and cap_ok:
            self._episode_metrics["matched_assignments"] += 1
        else:
            self._episode_metrics["unsafe_matches"] += 1

        if rider.tier == Tier.VIP and car.car_type != CarType.ARMORED:
            return -0.4
        if rider.group_size >= 4 and car.car_type != CarType.XUV and rider.tier != Tier.VIP:
            return -0.12
        if rider.tier == Tier.PREMIUM and rider.group_size < 4 and car.car_type != CarType.SEDAN:
            return -0.15
        if rider.group_size > car.capacity:
            return -0.3

        return self._launch_trip([rider], car, driver, pooled=False, events=events)

    def _action_pool(self, action: Action, events: list[str]) -> float:
        if not self._task.enable_pooling:
            return -0.02
        rider_ids = action.rider_ids or []
        if len(rider_ids) != 2:
            return -0.02
        riders = [self._riders.get(rid) for rid in rider_ids]
        if any(r is None for r in riders):
            return -0.02
        r1, r2 = riders
        car = self._cars.get(action.car_id or "")
        driver_id = action.driver_id or (car.driver_id if car else None)
        driver = self._drivers.get(driver_id or "") if driver_id else None
        if car is None or driver is None or not car.available or not driver.available:
            return -0.03
        if not all(r.waiting for r in riders):
            return -0.02
        if not all(r.share_allowed for r in riders):
            return -0.2
        if r1.pickup_zone != r2.pickup_zone:
            return -0.15

        self._episode_metrics["pool_actions"] += 1
        total_group = r1.group_size + r2.group_size
        if total_group > car.capacity:
            return -0.3

        compatible_drop = abs(r1.drop_zone - r2.drop_zone) <= 2
        if compatible_drop:
            self._episode_metrics["successful_pools"] += 1
            reward = self._launch_trip([r1, r2], car, driver, pooled=True, events=events)
            return reward + 0.08
        self._episode_metrics["unsafe_matches"] += 1
        return -0.1

    def _launch_trip(self, riders: list[RiderState], car: CarState, driver: DriverState, pooled: bool, events: list[str]) -> float:
        rider_ids = [r.rider_id for r in riders]
        car.available = False
        driver.available = False
        car.current_rider_ids = rider_ids
        for r in riders:
            r.assigned_car = car.car_id
            r.assigned_driver = driver.driver_id
            r.pooled_with = [x for x in rider_ids if x != r.rider_id]
        driver.zone = car.location_zone

        pickup_zone = riders[0].pickup_zone
        trip_zone = sum(abs(r.pickup_zone - r.drop_zone) for r in riders) / len(riders)
        to_pickup = abs(driver.zone - pickup_zone)
        base = 1.0 + trip_zone + to_pickup * 0.5
        duration_steps = max(1, int(round(base * self._traffic_factor)))
        busy_for = duration_steps * self._step_size

        car.busy_until = self._sim_time + busy_for
        driver.busy_until = self._sim_time + busy_for

        reward = 0.15
        if all(self._tier_match(r, car) for r in riders):
            reward += 0.15
        if all(r.group_size <= car.capacity for r in riders):
            reward += 0.10

        avg_wait = sum(r.wait_time for r in riders) / len(riders)
        reward += max(0.0, 0.30 - avg_wait * 0.01)

        if pooled and len(riders) == 2:
            reward += 0.05

        events.append(f"{'Pooled' if pooled else 'Assigned'} {','.join(rider_ids)} via {car.car_id} driven by {driver.driver_id}.")
        return reward

    def _process_trip_completions(self, events: list[str]) -> float:
        reward = 0.0
        for car in self._cars.values():
            if not car.available and car.busy_until <= self._sim_time:
                driver = self._drivers.get(car.driver_id or "")
                rider_ids = list(car.current_rider_ids)
                completed_riders = [self._riders[rid] for rid in rider_ids if rid in self._riders]
                final_drop_zone = completed_riders[0].drop_zone if completed_riders else car.location_zone

                if driver is not None:
                    driver.available = True
                    driver.busy_until = 0.0
                    driver.zone = final_drop_zone
                car.available = True
                car.current_rider_ids = []
                car.location_zone = final_drop_zone

                for rider in completed_riders:
                    rider.completed = True
                    rider.safe_drop = self._tier_match(rider, car)
                    self._episode_metrics["completed"] += 1
                    if rider.safe_drop:
                        self._episode_metrics["safe_completed"] += 1
                        reward += 0.05
                    else:
                        self._episode_metrics["unsafe_matches"] += 1
                        reward -= 0.02
                    events.append(f"Completed ride {rider.rider_id}.")

                reward += 0.04
        return reward

    def _update_driver_time_counters(self) -> None:
        for car in self._cars.values():
            self._episode_metrics["car_total_time"] += self._step_size
            if not car.available:
                self._episode_metrics["car_busy_time"] += self._step_size
        for driver in self._drivers.values():
            self._episode_metrics["driver_total_time"] += self._step_size
            if not driver.available:
                self._episode_metrics["driver_busy_time"] += self._step_size
            else:
                driver.idle_time += self._step_size

    def _update_metrics(self, prev_metrics: dict[str, float]) -> None:
        m = self._episode_metrics
        arrived = max(1, m["total_arrived"])
        assignments = max(1, m["assignments"])
        m["match_rate"] = m["matched_assignments"] / assignments
        m["completion_rate"] = m["completed"] / arrived
        m["cancellation_rate"] = max(0.0, 1.0 - (m["cancelled"] / arrived))
        avg_wait = m["wait_time_sum"] / max(1, m["wait_count"])
        m["avg_wait_time"] = avg_wait
        wait_norm = 1.0 - min(1.0, avg_wait / (self._task.sim_duration_seconds / 12.0))
        m["wait_efficiency"] = max(0.0, min(1.0, wait_norm))

        total_time = max(1.0, m["driver_total_time"])
        m["utilization_rate"] = max(0.0, min(1.0, m["driver_busy_time"] / total_time))
        m["pooling_rate"] = m["successful_pools"] / max(1, m["pool_actions"])
        m["reposition_rate"] = m["successful_repositions"] / max(1, m["repositions"])
        m["safe_drop_rate"] = m["safe_completed"] / max(1, m["completed"])

    def _compute_step_reward(self, reward_delta: float) -> float:
        penalty = 0.0
        for rider in self._riders.values():
            if rider.waiting:
                urgency_factor = 1.7 if rider.urgency == "urgent" else 1.0
                penalty += 0.0002 * urgency_factor * max(0.0, rider.wait_time / max(1e-9, self._step_size))
        return round(max(-0.1, reward_delta - penalty), 5)

    def _build_observation(self, events: list[str] | None = None) -> Observation:
        waiting = [
            RiderView(
                rider_id=r.rider_id,
                tier=r.tier.value,
                urgency=r.urgency,
                group_size=r.group_size,
                pickup_zone=r.pickup_zone,
                drop_zone=r.drop_zone,
                share_allowed=r.share_allowed,
                wait_time=r.wait_time,
                assigned_car=r.assigned_car,
                assigned_driver=r.assigned_driver,
                cancelled=r.cancelled,
                completed=r.completed,
                safe_drop=r.safe_drop,
            )
            for r in self.waiting_riders()
        ]
        cars = [
            CarView(
                car_id=c.car_id,
                car_type=c.car_type.value,
                capacity=c.capacity,
                location_zone=c.location_zone,
                driver_id=c.driver_id,
                available=c.available,
                busy_until=c.busy_until,
                current_rider_ids=list(c.current_rider_ids),
            )
            for c in self._cars.values()
        ]
        drivers = [
            DriverView(
                driver_id=d.driver_id,
                zone=d.zone,
                available=d.available,
                busy_until=d.busy_until,
                busy_time=d.busy_time,
                idle_time=d.idle_time,
                on_break=d.on_break,
            )
            for d in self._drivers.values()
        ]
        return Observation(
            current_time=self._sim_time,
            task_id=self._task.task_id if self._task else "",
            traffic_factor=self._traffic_factor,
            waiting_riders=waiting,
            available_cars=cars,
            driver_status=drivers,
            metrics=dict(self._episode_metrics),
            recent_events=events or [],
        )

    def _tier_match(self, rider: RiderState, car: CarState) -> bool:
        if rider.tier == Tier.VIP:
            return car.car_type == CarType.ARMORED
        if rider.group_size >= 4:
            return car.car_type == CarType.XUV
        if rider.tier == Tier.PREMIUM:
            return car.car_type == CarType.SEDAN
        return car.car_type == CarType.BASIC

    def _demand_heatmap(self) -> list[tuple[int, int]]:
        counts = {z: 0 for z in self.ZONES}
        for rider in self.waiting_riders():
            counts[rider.pickup_zone] += 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
