#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import TASKS, TaxiEnv
from baseline.agent import HeuristicAgent


def run_episode(task_id: str, seed: int, verbose: bool = False) -> dict:
    env = TaxiEnv()
    agent = HeuristicAgent(env)

    obs = env.reset(task_id=task_id, seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(obs)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        steps += 1

        if verbose and steps % 20 == 0:
            m = result.info["episode_metrics"]
            print(
                f"  Step {steps:4d} | sim={result.info['sim_time']/60:6.1f}m "
                f"| reward={result.reward:+.4f} | arrived={m['total_arrived']:.0f} "
                f"| completed={m['completed']:.0f} | match={m['match_rate']:.2f}"
            )

        if result.done or result.truncated:
            break

    final_score = env.score()
    metrics = env.episode_metrics

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": final_score,
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Taxi Dispatch Baseline Runner")
    parser.add_argument("--task", default=None, choices=list(TASKS), help="Task to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else list(TASKS.keys())

    print("=" * 65)
    print("  Taxi Dispatch OpenEnv — Baseline Evaluation")
    print("=" * 65)

    results = []
    for tid in tasks_to_run:
        task_cfg = TASKS[tid]
        print(f"\n▶  Task: {task_cfg.name}  [{task_cfg.difficulty.upper()}]")
        print(f"   {task_cfg.description[:80]}...")
        result = run_episode(tid, args.seed, args.verbose)
        results.append(result)

        print("\n   ┌─ RESULTS ─────────────────────────────────┐")
        print(f"   │  Steps         : {result['steps']}")
        print(f"   │  Total reward  : {result['total_reward']:+.4f}")
        print(f"   │  Final score   : {result['final_score']:.4f}  (0.0 – 1.0)")
        print(f"   │  Match rate    : {result['metrics']['match_rate']:.2%}")
        print(f"   │  Completion    : {result['metrics']['completion_rate']:.2%}")
        print(f"   │  Cancel rate   : {result['metrics']['cancellation_rate']:.2%}")
        print("   └────────────────────────────────────────────┘")

    print("\n" + "=" * 65)
    print(f"  SUMMARY (seed={args.seed})")
    print("=" * 65)
    print(f"  {'Task':<30} {'Difficulty':<10} {'Score':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for r in results:
        cfg = TASKS[r["task_id"]]
        print(f"  {cfg.name:<30} {cfg.difficulty:<10} {r['final_score']:>8.4f}")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print("=" * 65)

    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "results": results}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
