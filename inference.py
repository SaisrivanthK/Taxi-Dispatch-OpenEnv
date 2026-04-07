from __future__ import annotations

import json
import os
from typing import Any, Optional

from openai import OpenAI

from env import Action, ActionType, TASKS, TaxiEnv


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MAX_STEPS = 60
TEMPERATURE = 0.2
MAX_TOKENS = 220
BENCHMARK = "taxi_dispatch_v1"


SYSTEM_PROMPT = """
You are controlling a taxi dispatch agent.

Return ONE JSON object only with one of these forms:

1) {"action_type":"assign","rider_id":"...","car_id":"...","driver_id":"..."}
2) {"action_type":"pool","rider_ids":["...","..."],"car_id":"...","driver_id":"..."}
3) {"action_type":"reposition","driver_id":"...","target_zone":2}
4) {"action_type":"wait"}
5) {"action_type":"no_op"}

Rules:
- VIP riders require armored cars.
- Group size >= 4 should prefer xuv.
- Premium riders prefer sedan when group size is small.
- Use pooling only when compatible.
- Reposition idle drivers toward hot zones.
- Do not add commentary or markdown.
""".strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def obs_summary(obs: Any) -> str:
    return json.dumps(obs.model_dump(), separators=(",", ":"), ensure_ascii=False)


def heuristic_action(env: TaxiEnv) -> Action:
    waiting = env.waiting_riders()
    cars = env.available_cars()
    drivers = env.available_drivers()

    if not waiting:
        if env._task and env._task.enable_repositioning and drivers:
            hot = env._demand_heatmap()
            if hot and hot[0][1] > 0:
                return Action(action_type=ActionType.REPOSITION, driver_id=drivers[0].driver_id, target_zone=hot[0][0])
        return Action(action_type=ActionType.NO_OP)

    if env._task and env._task.enable_pooling and len(waiting) >= 2:
        r1 = waiting[0]
        for r2 in waiting[1:]:
            if r1.pickup_zone == r2.pickup_zone and r1.share_allowed and r2.share_allowed:
                car = next((c for c in cars if c.capacity >= r1.group_size + r2.group_size), None)
                driver = next((d for d in drivers if d.driver_id == car.driver_id), None) if car else None
                if car and driver:
                    return Action(action_type=ActionType.POOL, rider_ids=[r1.rider_id, r2.rider_id], car_id=car.car_id, driver_id=driver.driver_id)

    rider = waiting[0]
    preferred = []
    if rider.tier.value == "vip":
        preferred = ["armored"]
    elif rider.group_size >= 4:
        preferred = ["xuv"]
    elif rider.tier.value == "premium":
        preferred = ["sedan"]
    else:
        preferred = ["basic"]

    chosen_car = None
    for ctype in preferred:
        for car in cars:
            if car.car_type.value == ctype and car.capacity >= rider.group_size:
                chosen_car = car
                break
        if chosen_car:
            break
    if chosen_car is None:
        chosen_car = next((car for car in cars if car.capacity >= rider.group_size), None)

    if chosen_car:
        driver = next((d for d in drivers if d.driver_id == chosen_car.driver_id), None) or (drivers[0] if drivers else None)
        if driver:
            return Action(
                action_type=ActionType.ASSIGN,
                rider_id=rider.rider_id,
                car_id=chosen_car.car_id,
                driver_id=driver.driver_id,
            )

    if env._task and env._task.enable_repositioning and drivers:
        hot = env._demand_heatmap()
        if hot and hot[0][1] > 0:
            return Action(action_type=ActionType.REPOSITION, driver_id=drivers[0].driver_id, target_zone=hot[0][0])

    return Action(action_type=ActionType.NO_OP)


def model_action(client: OpenAI, env: TaxiEnv, obs: Any) -> Action:
    user_prompt = f"Task: {obs.task_id}\nObservation: {obs_summary(obs)}\nAvailable actions must respect vehicle tiers, capacity, pooling, and traffic.\nReturn JSON only."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return heuristic_action(env)


def run_task(task_id: str, seed: int) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    env = TaxiEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps = 0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            action = model_action(client, env, obs) if client else heuristic_action(env)
            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            steps = step

            log_step(step=step, action=json.dumps(action.to_dict(), separators=(",", ":")), reward=result.reward, done=result.done, error=None)

            if result.done or result.truncated:
                break

        score = env.score()
        success = score >= 0.1
    except Exception as exc:
        score = env.score()
        log_step(step=steps + 1, action="null", reward=0.0, done=True, error=str(exc))
        success = False
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    seed = int(os.getenv("SEED", "42"))
    for task_id in TASKS.keys():
        run_task(task_id, seed)


if __name__ == "__main__":
    main()
