from __future__ import annotations

import json
import os

from openai import OpenAI
from env import Action, ActionType, TaxiEnv

MAX_STEPS = 60

# =========================
# REQUIRED LLM CLIENT
# =========================
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)

MODEL = "gpt-4o-mini"


# =========================
# LLM ACTION
# =========================
def llm_action(obs) -> Action:
    prompt = f"""
You are a taxi dispatch agent.

Return ONLY JSON:

1) assign → {{"action_type":"assign","rider_id":"...","car_id":"...","driver_id":"..."}}
2) reposition → {{"action_type":"reposition","driver_id":"...","target_zone":1}}
3) wait → {{"action_type":"wait"}}
4) no_op → {{"action_type":"no_op"}}

Rules:
- VIP → armored
- group >=4 → xuv
- premium → sedan
- minimize wait time
- prefer closest driver

Observation:
{json.dumps(obs.model_dump())}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()

    try:
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return Action(action_type=ActionType.NO_OP)


# =========================
# RUN TASK
# =========================
def run_task(task_id: str, seed: int):
    env = TaxiEnv()
    obs = env.reset(task_id=task_id, seed=seed)

    print(f"[START] task={task_id}", flush=True)

    steps = 0

    for step in range(1, MAX_STEPS + 1):
        try:
            action = llm_action(obs)
        except Exception:
            action = Action(action_type=ActionType.NO_OP)

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
        f"[END] task={task_id} score={score:.6f} steps={steps}",
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