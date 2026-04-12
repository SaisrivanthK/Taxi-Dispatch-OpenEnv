from __future__ import annotations

import json
import os
from typing import Any

from env import Action, ActionType, TaxiEnv

MAX_STEPS = 60


# =========================
# SMART AGENT
# =========================
def score_assignment(env, rider, car, driver):
    distance = abs(driver.zone - rider.pickup_zone)

    tier_score = {
        "vip": 3.0,
        "premium": 2.0,
        "normal": 1.0,
    }[rider.tier.value]

    capacity_score = 1.0 - abs(car.capacity - rider.group_size) * 0.1
    wait_penalty = rider.wait_time * 0.02

    safe_bonus = 0.0
    if rider.tier.value == "vip" and car.car_type.value == "armored":
        safe_bonus += 2.0
    elif rider.group_size >= 4 and car.car_type.value == "xuv":
        safe_bonus += 1.5
    elif rider.tier.value == "premium" and car.car_type.value == "sedan":
        safe_bonus += 1.0

    return (
        tier_score
        + capacity_score
        + safe_bonus
        - distance * 0.5
        - wait_penalty
    )


def smart_action(env: TaxiEnv) -> Action:
    riders = env.waiting_riders()
    cars = env.available_cars()
    drivers = env.available_drivers()

    if not riders:
        if env._task and env._task.enable_repositioning and drivers:
            heat = env._demand_heatmap()
            if heat:
                return Action(
                    action_type=ActionType.REPOSITION,
                    driver_id=drivers[0].driver_id,
                    target_zone=heat[0][0],
                )
        return Action(action_type=ActionType.NO_OP)

    best = None
    best_score = -1e9

    for rider in riders[:5]:
        for car in cars:
            if car.capacity < rider.group_size:
                continue

            driver = next((d for d in drivers if d.driver_id == car.driver_id), None)
            if not driver:
                continue

            score = score_assignment(env, rider, car, driver)

            if score > best_score:
                best_score = score
                best = (rider, car, driver)

    if best:
        r, c, d = best
        return Action(
            action_type=ActionType.ASSIGN,
            rider_id=r.rider_id,
            car_id=c.car_id,
            driver_id=d.driver_id,
        )

    return Action(action_type=ActionType.NO_OP)


# =========================
# RUN TASK WITH LOGGING
# =========================
def run_task(task_id: str, seed: int):
    env = TaxiEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    print(f"[START] task={task_id}", flush=True)

    steps = 0

    for step in range(1, MAX_STEPS + 1):
        action = smart_action(env)

        result = env.step(action)
        obs = result.observation
        steps = step

        print(
            f"[STEP] step={step} reward={result.reward:.4f} done={str(result.done).lower()}",
            flush=True,
        )

        if result.done or result.truncated:
            break

    score = env.score()

    print(
        f"[END] task={task_id} score={score:.4f} steps={steps}",
        flush=True,
    )


# =========================
# MAIN
# =========================
def main():
    tasks = ["ride_matching", "dispatch_allocation", "surge_mobility"]

    for task in tasks:
        run_task(task, seed=42)


if __name__ == "__main__":
    main()