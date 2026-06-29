"""
run_ablation.py — Reward Shaping Ablation Study

4가지 variant를 각각 학습 후 벤치마크:
  - full:       inbound + outbound shaping (기준)
  - no_out:     outbound shaping 제거 (action=2 guidance 없음)
  - no_in:      inbound shaping 제거  (action=1 guidance 없음)
  - no_shaping: shaping 없음 (R_env만 사용)

사용법:
    python rl/run_ablation.py
    python rl/run_ablation.py --episodes 1000 --bench 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from rl.train_rl import train as rl_train
from rl.rl_policy import QLearningPolicy

CFG = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 3,
    "num_outbound_doors": 3,
    "buffer_capacity": 80.0,
    "arrival_count_min": 50,
    "arrival_count_max": 70,
    "all_trucks_at_start": False,
    "arrival_pattern": "clustered",
    "arrival_cluster_count": 4,
    "arrival_time_window": 300,
    "compound_trucks": False,
    "use_truck_selection": False,
    "enable_disruptions": True,
    "disruption_door_failure": True,
    "disruption_door_failure_prob": 0.02,
    "disruption_door_failure_duration_min": 10,
    "disruption_door_failure_duration_max": 20,
}

VARIANTS = {
    "full":       {"use_outbound_shaping": True,  "use_inbound_shaping": True},
    "no_out":     {"use_outbound_shaping": False, "use_inbound_shaping": True},
    "no_in":      {"use_outbound_shaping": True,  "use_inbound_shaping": False},
    "no_shaping": {"use_outbound_shaping": False, "use_inbound_shaping": False},
}


def run_episode(net, cfg, seed):
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    n = env.num_lanes
    policy = QLearningPolicy(net=net, epsilon=0.0)
    policies = [policy] * n
    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(n)]
        obs, _, done, _ = env.step(actions)
    m = env.metrics.copy()
    m["total_ticks"] = float(env.t)
    return m


def aggregate(results):
    keys = list(results[0].keys())
    return {k: {"mean": float(np.mean([r[k] for r in results])),
                "std":  float(np.std( [r[k] for r in results]))} for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--bench",    type=int, default=20)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=str, default="rl/ablation_results.json")
    args = parser.parse_args()

    print("=" * 62)
    print(f"[Ablation] Reward Shaping — {args.episodes} ep 학습 / {args.bench} ep 벤치마크")
    print("=" * 62)

    all_results = {}

    for name, flags in VARIANTS.items():
        label = (
            "inbound+outbound" if flags["use_inbound_shaping"] and flags["use_outbound_shaping"]
            else "outbound only" if flags["use_outbound_shaping"]
            else "inbound only"  if flags["use_inbound_shaping"]
            else "없음 (R_env만)"
        )
        print(f"\n[{name}] shaping={label}")

        result = rl_train(
            num_episodes=args.episodes,
            lr=5e-4,
            seed=args.seed,
            save_dir=f"checkpoints_ablation_{name}",
            env_config=CFG,
            log_interval=args.episodes,  # 마지막에만 출력
            **flags,
        )
        net = result["net"]

        bench = [run_episode(net, CFG, seed=200 + ep) for ep in range(args.bench)]
        agg = aggregate(bench)
        all_results[name] = {
            "flags": flags,
            "label": label,
            "benchmark": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                          for k, v in agg.items()},
        }

        tk = agg["total_ticks"]
        ed = agg["empty_departures"]
        print(f"  → ticks={tk['mean']:.1f}±{tk['std']:.1f}  empty={ed['mean']:.2f}±{ed['std']:.1f}")

    # 결과 출력
    print("\n" + "=" * 72)
    print("[결과] Reward Shaping Ablation (벤치마크 평균)")
    print("=" * 72)
    header = f"{'variant':14s}  {'shaping':22s}  {'Ticks':>10s}  {'Std':>6s}  {'빈출발':>8s}"
    print(header)
    print("-" * len(header))

    full_ticks = all_results["full"]["benchmark"]["total_ticks"]["mean"]
    for name, res in all_results.items():
        tk = res["benchmark"]["total_ticks"]
        ed = res["benchmark"]["empty_departures"]
        delta = tk["mean"] - full_ticks
        marker = "" if name == "full" else f"  ({delta:+.1f})"
        print(f"  {name:14s}  {res['label']:22s}  {tk['mean']:>8.1f}  {tk['std']:>6.1f}  {ed['mean']:>6.2f}{marker}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    main()
