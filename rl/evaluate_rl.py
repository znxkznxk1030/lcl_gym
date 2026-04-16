"""
evaluate_rl.py — 학습된 RL 에이전트 vs 베이스라인 비교 평가

실행 (프로젝트 루트에서):
    python rl/evaluate_rl.py
    python rl/evaluate_rl.py --weights checkpoints/weights_final
    python rl/evaluate_rl.py --episodes 50
"""
from __future__ import annotations

import os
import sys

# 프로젝트 루트를 sys.path에 추가 (어느 위치에서 실행해도 동작)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import RandomPolicy, FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy
from rl.networks import NumpyMLP
from rl.rl_policy import QLearningPolicy
from run_simulation import run_episode, aggregate


# ---------------------------------------------------------------------------
# RL 에이전트 로드
# ---------------------------------------------------------------------------

def load_rl_agents(weights_path: str, env: CrossDockEnv) -> List[QLearningPolicy]:
    net = NumpyMLP(
        obs_size=env.obs_size,
        hidden=64,
        n_actions=env.num_inbound_doors + 1,
    )
    net.load(weights_path)
    return [
        QLearningPolicy(net=net, epsilon=0.0, rng=np.random.default_rng(k))
        for k in range(env.num_lanes)
    ]


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(
    weights_path: str = "checkpoints/weights_final",
    num_episodes: int = 20,
    seed_offset: int = 1000,
):
    env = CrossDockEnv({**DEFAULT_CONFIG}, seed=0)

    policy_factories = {
        "Random":    lambda: [RandomPolicy(np.random.default_rng(i)) for i in range(env.num_lanes)],
        "FIFO":      lambda: [FIFOPolicy()            for _ in range(env.num_lanes)],
        "Greedy":    lambda: [GreedyPolicy()          for _ in range(env.num_lanes)],
        "Heuristic": lambda: [HeuristicPriorityPolicy() for _ in range(env.num_lanes)],
        "RL":        lambda: load_rl_agents(weights_path, env),
    }

    cols       = ["total_throughput", "buffer_overflow_count", "total_reward",
                  "door_utilization", "avg_dwell_time"]
    col_labels = ["throughput", "overflow", "reward", "door_util", "dwell_t"]

    print(f"\n평가 결과 ({num_episodes} 에피소드, 가중치: {weights_path})\n")
    print(f"{'Policy':<14}", end="")
    for lbl in col_labels:
        print(f"{lbl:>14}", end="")
    print()
    print("-" * (14 + 14 * len(cols)))

    results_all: Dict[str, Dict] = {}

    for name, factory in policy_factories.items():
        results = []
        for ep in range(num_episodes):
            metrics = run_episode(env, factory(), seed=seed_offset + ep)
            results.append(metrics)

        agg = aggregate(results)
        results_all[name] = agg

        print(f"{name:<14}", end="")
        for col in cols:
            print(f"{agg[col]['mean']:>10.2f}±{agg[col]['std']:<3.1f}", end="")
        print()

    print()
    if "RL" in results_all:
        tp_delta = results_all["RL"]["total_throughput"]["mean"] - results_all["Greedy"]["total_throughput"]["mean"]
        ov_delta = results_all["RL"]["buffer_overflow_count"]["mean"] - results_all["FIFO"]["buffer_overflow_count"]["mean"]
        print(f"RL throughput vs Greedy : {tp_delta:+.2f}  ({'개선' if tp_delta >= 0 else '저하'})")
        print(f"RL overflow   vs FIFO   : {ov_delta:+.2f}  ({'개선' if ov_delta <= 0 else '저하'})")

    return results_all


# ---------------------------------------------------------------------------
# 학습 곡선 출력
# ---------------------------------------------------------------------------

def print_training_curve(save_dir: str = "checkpoints"):
    paths = {k: os.path.join(save_dir, f) for k, f in [
        ("rewards",    "episode_rewards.npy"),
        ("throughput", "throughput_log.npy"),
        ("overflow",   "overflow_log.npy"),
        ("loss",       "td_loss_log.npy"),
    ]}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"학습 로그 없음: {missing}")
        return

    rewards    = np.load(paths["rewards"])
    throughput = np.load(paths["throughput"])
    overflow   = np.load(paths["overflow"])
    loss       = np.load(paths["loss"])
    n = len(rewards)

    print(f"\n학습 곡선 요약 (총 {n} 에피소드, 이동평균 window={min(100, n//5 or 1)})\n")
    print(f"{'구간':<20} {'reward':>10} {'throughput':>12} {'overflow':>10} {'td_loss':>10}")
    print("-" * 65)
    for label, sl in [
        ("초반 (1~20%)",   slice(0,       n//5)),
        ("중반 (40~60%)",  slice(2*n//5,  3*n//5)),
        ("후반 (80~100%)", slice(4*n//5,  n)),
    ]:
        print(f"{label:<20} {np.mean(rewards[sl]):>10.1f} {np.mean(throughput[sl]):>12.1f} "
              f"{np.mean(overflow[sl]):>10.1f} {np.mean(loss[sl]):>10.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    args = sys.argv[1:]
    kwargs = {"weights_path": "checkpoints/weights_final", "num_episodes": 20}
    for i, a in enumerate(args):
        if a == "--weights"  and i + 1 < len(args): kwargs["weights_path"] = args[i+1]
        if a == "--episodes" and i + 1 < len(args): kwargs["num_episodes"] = int(args[i+1])
    return kwargs


if __name__ == "__main__":
    kwargs = _parse_args()
    print_training_curve()
    evaluate(**kwargs)
