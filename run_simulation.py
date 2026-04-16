"""
run_simulation.py — Compare baseline policies on the CrossDock environment.
"""
from __future__ import annotations

import sys
import numpy as np
from typing import List, Dict

from crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from policies import RandomPolicy, FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_episode(
    env: CrossDockEnv,
    policies,
    seed: int,
) -> Dict:
    """Run one episode and return the final metrics dict."""
    env._seed = seed
    obs = env.reset()

    total_reward = 0.0
    for _ in range(env.episode_length):
        actions = [
            policies[k].act(obs[k], env.num_inbound_doors)
            for k in range(env.num_lanes)
        ]
        obs, rewards, done, info = env.step(actions)
        total_reward += sum(rewards)
        if done:
            break

    metrics = info.get("metrics", env.metrics.copy())
    metrics["total_reward"] = total_reward
    metrics["door_utilization"] = env.door_utilization
    metrics["avg_dwell_time"] = env.avg_dwell_time
    return metrics


def aggregate(results: List[Dict]) -> Dict:
    keys = list(results[0].keys())
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = {**DEFAULT_CONFIG}
    num_episodes = 20

    policy_factories = {
        "Random":    lambda: [RandomPolicy(np.random.default_rng(i))   for i in range(config["num_lanes"])],
        "FIFO":      lambda: [FIFOPolicy()                              for _ in range(config["num_lanes"])],
        "Greedy":    lambda: [GreedyPolicy()                            for _ in range(config["num_lanes"])],
        "Heuristic": lambda: [HeuristicPriorityPolicy()                 for _ in range(config["num_lanes"])],
    }

    env = CrossDockEnv(config, seed=0)

    print(f"{'Policy':<14}", end="")
    cols = ["total_throughput", "buffer_overflow_count", "total_reward",
            "door_utilization", "avg_dwell_time"]
    col_labels = ["throughput", "overflow", "reward", "door_util", "dwell_t"]
    for lbl in col_labels:
        print(f"{lbl:>14}", end="")
    print()
    print("-" * (14 + 14 * len(cols)))

    for name, factory in policy_factories.items():
        results = []
        for ep in range(num_episodes):
            policies = factory()
            metrics = run_episode(env, policies, seed=ep)
            results.append(metrics)

        agg = aggregate(results)
        print(f"{name:<14}", end="")
        for col in cols:
            mean = agg[col]["mean"]
            std  = agg[col]["std"]
            print(f"{mean:>10.2f}±{std:<3.1f}", end="")
        print()

    print()
    print("Done. Each policy ran for", num_episodes, "episodes.")


if __name__ == "__main__":
    main()
