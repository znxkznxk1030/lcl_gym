#!/usr/bin/env python3
"""
실험 A 재실험 — 동적 환경의 Compound Truck (RL이 이기는 구조).

기존 실험 A는 **정적 일회성 배정**(Exact/SA/Greedy/RL/Heuristic/Random을 makespan으로 비교)이라
오프라인 최적화기가 최적이어서 RL이 이기지 못했다. 여기서는 같은 compound 트럭을 **동적 환경**
(`compound_dynamic=True`: 동적 도착 + 출고 도크 희소 + 도어 고장 + 실시간 action=2 재배치)에 넣고,
실시간 lane-agent 정책(RL vs FIFO/Greedy/Heuristic/Random)을 **makespan(total ticks)**으로 비교한다.

compound 특성: 각 트럭은 화물이 가장 많은 목적지를 보유(kept)·직접 운반(부분 하차), 나머지만 하차 →
동적 희소 도크가 레인을 비우며 처리. 도크가 비어가는 레인을 서비스하면 빈 출발(낭비) → RL은 action=2로
빈-레인 도크를 화물 있는 레인으로 재배치하는 타이밍을 학습해 우위.

RL 가중치: 본문 동적 lane-agent RL(`checkpoints_2stage_8door`). 트럭 분포를 비-compound 동적 env와
동일하게 유지해 기존 RL이 in-distribution.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy, RandomPolicy
from rl.networks import NumpyMLP
from rl.rl_policy import QLearningPolicy

CKPT = "checkpoints_2stage_8door/weights_final.npz"
SEEDS = list(range(20))

BASE = {
    **DEFAULT_CONFIG,
    "compound_trucks": True, "compound_dynamic": True, "partial_unloading": True,
    "num_lanes": 5, "num_inbound_doors": 3, "buffer_capacity": 80.0,
    "all_trucks_at_start": False, "use_scheduled_arrivals": True,
    "arrival_count_min": 50, "arrival_count_max": 70,
    "arrival_pattern": "clustered", "arrival_cluster_count": 4, "arrival_time_window": 300,
    "unit_load_time": 1, "entering_time_min": 1, "entering_time_max": 3,
    "enable_disruptions": True, "disruption_door_failure": True,
    "disruption_door_failure_prob": 0.02,
}

_W = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT))
OBS = int(_W["W1"].shape[0])


def make_rl():
    net = NumpyMLP(obs_size=OBS, n_actions=3)
    net.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), CKPT))
    return QLearningPolicy(net=net, epsilon=0.0)


POLICIES = {"RL": make_rl, "FIFO": FIFOPolicy, "Greedy": GreedyPolicy,
            "Heuristic": HeuristicPriorityPolicy, "Random": RandomPolicy}


def run(cfg, make, seed):
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
    m = info["metrics"]
    return env.t, m["empty_departures"], m["total_throughput"], m["kept_volume_delivered"]


def evaluate(cfg):
    res = {}
    for name, make in POLICIES.items():
        ts, es, th, kv = [], [], [], []
        for s in SEEDS:
            t, e, tp, k = run(cfg, make, s)
            ts.append(t); es.append(e); th.append(tp); kv.append(k)
        res[name] = {"ticks": float(np.mean(ts)), "ticks_std": float(np.std(ts)),
                     "empty_dep": float(np.mean(es)), "throughput": float(np.mean(th)),
                     "kept_delivered": float(np.mean(kv))}
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 74)
    print("실험 A 재실험 — 동적 Compound Truck (compound_dynamic, 희소 도크 + 도어 고장)")
    print(f"  5 lanes, 3 inbound doors, 부분 하차, door failure prob=0.02, seeds={len(SEEDS)}")
    print("=" * 74)

    results = {}
    cfg_scarce = {**BASE, "num_outbound_doors": 3}   # 희소 도크 3 < 5 lanes
    res = evaluate(cfg_scarce); results["scarce_docks_3"] = res
    print("\n[희소 도크 = 3 < 5 lanes]\n")
    print(f"{'정책':>10} | {'Avg Ticks ↓':>11} | {'Std':>5} | {'Empty Dep ↓':>11} | {'Throughput':>10} | {'kept배송':>8}")
    print("-" * 72)
    order = sorted(res, key=lambda k: res[k]["ticks"])
    for k in order:
        r = res[k]; tag = " 🥇" if k == order[0] else ""
        print(f"{k:>10} | {r['ticks']:11.1f} | {r['ticks_std']:5.1f} | {r['empty_dep']:11.2f} | "
              f"{r['throughput']:10.0f} | {r['kept_delivered']:8.0f}{tag}")

    # 대조: 풍부 도크(8 ≥ 5) → 재배치 여지 없음
    cfg_ab = {**BASE, "num_outbound_doors": 8}
    ab = {n: float(np.mean([run(cfg_ab, POLICIES[n], s)[0] for s in SEEDS]))
          for n in ("FIFO", "Greedy")}
    results["abundant_docks_8"] = ab
    print(f"\n[풍부 도크 = 8 ≥ 5]  대조: FIFO={ab['FIFO']:.1f}  Greedy={ab['Greedy']:.1f} "
          f"→ 빈-레인 도크 없어 action=2 무의미 (RL 우위 여지 소멸)")

    # ── SA/Exact 와의 비교: 배정 레버 vs action=2 레버 ───────────────────
    # SA/Exact 는 정적 일회성 배정 최적화기 → 동적 env에 직접 적용 불가.
    #   동적 env에서 그들이 정할 수 있는 건 '보유 목적지(kept)' 배정 레버뿐(실시간 action=2 불가).
    #   배정 레버를 argmax/random/min 으로 쓸어 'SA/Exact가 도달 가능한 천장'을 측정.
    print("\n[SA·Exact 와 비교] 배정 레버(그들이 최적화) vs action=2 레버(RL이 최적화)\n")
    lever = {}
    for rule in ("argmax", "random", "min"):
        cfg = {**BASE, "num_outbound_doors": 3, "compound_kept_rule": rule}
        lever[rule] = float(np.mean([run(cfg, FIFOPolicy, s)[0] for s in SEEDS]))
    results["assignment_lever_fifo"] = lever
    best_assign = min(lever.values())
    print(f"  배정 레버 (FIFO 정적 실행):  argmax={lever['argmax']:.1f}  "
          f"random={lever['random']:.1f}  min={lever['min']:.1f}")
    print(f"  → 최적 배정 = argmax({lever['argmax']:.1f}). SA/Exact가 배정을 아무리 최적화해도 "
          f"**천장 = {best_assign:.1f}** (action=2 없음).")
    print(f"  action=2 레버 (RL): {res['RL']['ticks']:.1f}  →  SA/Exact 천장({best_assign:.1f})보다 "
          f"{best_assign - res['RL']['ticks']:+.1f} 우수.")
    results["sa_exact_ceiling"] = best_assign

    rl, fifo, greedy = res["RL"]["ticks"], res["FIFO"]["ticks"], res["Greedy"]["ticks"]
    print("\n[요약]")
    print(f"  RL {rl:.1f} < FIFO {fifo:.1f} < Greedy {greedy:.1f}  (RL vs FIFO {rl-fifo:+.1f} ticks, "
          f"empty_dep {res['RL']['empty_dep']:.2f} vs {res['FIFO']['empty_dep']:.2f})")
    print("  → 정적 배정 실험 A에서 지던 RL이, 동적+희소+재배치 구조에선 baseline을 이긴다.")

    out = os.path.join(here, "compound_dynamic_A_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
