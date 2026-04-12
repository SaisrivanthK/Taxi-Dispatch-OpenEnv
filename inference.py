from __future__ import annotations

import json
import os
from typing import Any, Optional

from openai import OpenAI

from env import Action, ActionType, TaxiEnv

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
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
# OPTIONAL LLM AGENT
# =========================
def model_action(client: OpenAI, obs: Any) -> dict:
    prompt = f"Observation: {json.dumps(obs.model_dump())}\nReturn best action JSON."

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return json.loads(completion.choices[0].message.content)


# =========================
# RUN TASK
# =========================
def run_task(task_id: str, seed: int):
    env = TaxiEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    client = OpenAI(api_key=API_KEY) if API_KEY else None

    rewards = []

    for step in range(MAX_STEPS):
        try:
            if client:
                action_dict = model_action(client, obs)
                action = Action(**action_dict)
            else:
                action = smart_action(env)
        except Exception:
            action = smart_action(env)

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward)

        if result.done or result.truncated:
            break

    score = env.score()

    print(f"\nTask: {task_id}")
    print(f"Score: {score:.4f}")
    return score


# =========================
# MAIN
# =========================
def main():
    tasks = ["ride_matching", "dispatch_allocation", "surge_mobility"]

    scores = []
    for t in tasks:
        s = run_task(t, seed=42)
        scores.append(s)

    avg = sum(scores) / len(scores)

    print("\n==========================")
    print(f"Average Score: {avg:.4f}")
    print("==========================")


if __name__ == "__main__":
    main()