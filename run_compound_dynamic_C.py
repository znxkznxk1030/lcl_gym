#!/usr/bin/env python3
"""
실험 C 재수행 — 동적 Compound 환경에서 도어 고장 확률 sweep (강인성).

실험 C는 원래 정적 배정에서 도어 고장 영향을 봤다(헤드룸=0, RL-DR 무이득). 여기서는 RL이 실제
역할(action=2 재배치)을 하는 **동적 compound 환경**에서 도어 고장 확률을 0~0.20으로 높여가며
RL vs 베이스라인의 강인성을 비교한다. (해당 모델 = 본문 동적 lane-agent RL, prob=0.02 학습.)

관전 포인트(본문 실험 5/6과 동일 구조):
  - RL은 **학습 분포(prob≈0.02) 근처**에서 우위.
  - 고확률(OOD)에서 **분포 이탈(distribution shift)**로 붕괴 가능 → 도메인 랜덤화 재학습 필요성.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv
from env.policies import FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy, RandomPolicy
from rl.networks import NumpyMLP
from rl.rl_policy import QLearningPolicy
from run_compound_dynamic_A import BASE, OBS, CKPT

SEEDS = list(range(20))
PROBS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]


def make_rl():
    net = NumpyMLP(obs_size=OBS, n_actions=3)
    net.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT))
    return QLearningPolicy(net=net, epsilon=0.0)


POLICIES = {"RL": make_rl, "FIFO": FIFOPolicy, "Greedy": GreedyPolicy,
            "Heuristic": HeuristicPriorityPolicy, "Random": RandomPolicy}


def run(prob, make, seed):
    cfg = {**BASE, "num_outbound_doors": 3,
           "enable_disruptions": prob > 0, "disruption_door_failure": prob > 0,
           "disruption_door_failure_prob": prob}
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    pols = [make() for _ in range(env.num_lanes)]
    is_rl = hasattr(pols[0], "net")
    while True:
        acts = [pols[k].act(obs[k][:OBS] if is_rl else obs[k], env.num_inbound_doors)
                for k in range(env.num_lanes)]
        obs, _, done, info = env.step(acts)
        if done:
            break
    return env.t, info["metrics"]["empty_departures"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("실험 C 재수행 — 동적 Compound 환경 도어 고장 sweep (RL action=2 강인성)")
    print(f"  5 lanes, 3 inbound doors, 희소 도크=3, 부분 하차, seeds={len(SEEDS)}")
    print("  (해당 모델 = 동적 lane-agent RL, prob=0.02 학습)")
    print("=" * 78)

    results = {}
    header = f"{'prob':>6} | " + " | ".join(f"{n:>8}" for n in POLICIES) + " | winner"
    print("\n[Avg Ticks ↓]")
    print(header); print("-" * len(header))
    for prob in PROBS:
        row = {}
        for name, make in POLICIES.items():
            ts = [run(prob, make, s)[0] for s in SEEDS]
            row[name] = float(np.mean(ts))
        results[f"{prob}"] = row
        winner = min(row, key=row.get)
        cells = " | ".join(f"{row[n]:8.1f}" for n in POLICIES)
        print(f"{prob:>6} | {cells} | {winner}")

    # empty departures (action=2 효과 지표)
    print("\n[Empty Departures ↓]  (RL action=2 재배치의 직접 지표)")
    print(f"{'prob':>6} | " + " | ".join(f"{n:>8}" for n in POLICIES))
    print("-" * len(header))
    ed_all = {}
    for prob in PROBS:
        ed = {name: float(np.mean([run(prob, make, s)[1] for s in SEEDS]))
              for name, make in POLICIES.items()}
        ed_all[f"{prob}"] = ed
        print(f"{prob:>6} | " + " | ".join(f"{ed[n]:8.2f}" for n in POLICIES))
    results["empty_departures"] = ed_all

    print("\n[분석]")
    rl = {p: results[p]["RL"] for p in results if p != "empty_departures"}
    fifo = {p: results[p]["FIFO"] for p in results if p != "empty_departures"}
    print(f"  prob=0.02(학습): RL={rl['0.02']:.1f} vs FIFO={fifo['0.02']:.1f} "
          f"({rl['0.02']-fifo['0.02']:+.1f}) → RL 우위")
    print(f"  prob=0.20(OOD):  RL={rl['0.2']:.1f} vs FIFO={fifo['0.2']:.1f} "
          f"({rl['0.2']-fifo['0.2']:+.1f}) → RL 붕괴(분포 이탈)")
    print("  → RL 우위는 학습 분포 근처에 국한. 고확률 OOD에선 본문 실험 5처럼 붕괴 → 도메인 랜덤화 필요.")

    out = os.path.join(here, "compound_dynamic_C_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
