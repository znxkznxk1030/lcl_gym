"""
run_simulation.py — Compare baseline policies on the CrossDock environment.

Verbose mode (--verbose):
    python run_simulation.py --verbose [--policy greedy] [--steps 10]
"""
from __future__ import annotations

import sys
import numpy as np
from typing import List, Dict

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import RandomPolicy, FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy


# ---------------------------------------------------------------------------
# Verbose step printer
# ---------------------------------------------------------------------------

def _door_str(env: CrossDockEnv) -> str:
    parts = []
    for d in env.doors:
        if d.is_busy:
            parts.append(f"D{d.door_id}[busy:{d.remaining_time}t]")
        else:
            parts.append(f"D{d.door_id}[idle]")
    return "  ".join(parts)


def _lane_str(env: CrossDockEnv) -> str:
    parts = []
    for lane in env.lanes:
        parts.append(
            f"L{lane.lane_id}(q={lane.queue_volume:.0f} timer={lane.dispatch_timer})"
        )
    return "  ".join(parts)


def run_verbose(
    env: CrossDockEnv,
    policies,
    max_steps: int = 10,
):
    """Run one episode step-by-step with detailed per-step output."""
    obs = env.reset()
    print("=" * 70)
    print(f"EPISODE START  (episode_length={env.episode_length}, "
          f"showing first {max_steps} steps)")
    print("=" * 70)

    for t in range(min(max_steps, env.episode_length)):
        actions = [
            policies[k].act(obs[k], env.num_inbound_doors)
            for k in range(env.num_lanes)
        ]

        # ── state before step ──────────────────────────────────────────
        print(f"\n[Step {t:>3}]")
        print(f"  Waiting trucks : {len(env.waiting_trucks)}")
        print(f"  Buffer         : {env.buffer:.0f} / {env.buffer_capacity}")
        print(f"  Doors          : {_door_str(env)}")
        print(f"  Lanes          : {_lane_str(env)}")

        action_labels = {0: "skip"}
        for i in range(env.num_inbound_doors):
            action_labels[i + 1] = f"door{i}"
        action_str = "  ".join(
            f"L{k}→{action_labels[a]}" for k, a in enumerate(actions)
        )
        print(f"  Actions        : {action_str}")

        obs, rewards, done, info = env.step(actions)

        # ── outcome ───────────────────────────────────────────────────
        reward_str = "  ".join(f"L{k}:{r:+.1f}" for k, r in enumerate(rewards))
        print(f"  Rewards        : {reward_str}")
        print(f"  Doors (after)  : {_door_str(env)}")
        print(f"  Lanes (after)  : {_lane_str(env)}")

        if done:
            print("\n[Episode done]")
            break

    print("\n" + "=" * 70)
    m = env.metrics
    print(f"Metrics so far  throughput={m['total_throughput']:.1f}  "
          f"overflow={m['buffer_overflow_count']}  "
          f"door_util={env.door_utilization:.2%}  "
          f"dwell_t={env.avg_dwell_time:.2f}")
    print("=" * 70)


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
    args = sys.argv[1:]
    verbose = "--verbose" in args

    # --verbose 옵션 파싱
    policy_name = "greedy"
    max_steps = 10
    if verbose:
        for i, a in enumerate(args):
            if a == "--policy" and i + 1 < len(args):
                policy_name = args[i + 1].lower()
            if a == "--steps" and i + 1 < len(args):
                max_steps = int(args[i + 1])

        config = {**DEFAULT_CONFIG}
        env = CrossDockEnv(config, seed=42)
        policy_map = {
            "random":    [RandomPolicy(np.random.default_rng(i)) for i in range(env.num_lanes)],
            "fifo":      [FIFOPolicy()       for _ in range(env.num_lanes)],
            "greedy":    [GreedyPolicy()     for _ in range(env.num_lanes)],
            "heuristic": [HeuristicPriorityPolicy() for _ in range(env.num_lanes)],
        }
        policies = policy_map.get(policy_name, policy_map["greedy"])
        print(f"Policy: {policy_name.upper()}")
        run_verbose(env, policies, max_steps=max_steps)
        return

    # 기본: 벤치마크 모드
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
