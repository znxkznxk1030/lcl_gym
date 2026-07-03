#!/usr/bin/env python3
"""
Door Disruption 하에서의 Compound 배정 강인성 실험.

도어 고장(door failure)을 추가한 환경에서, 각 배정 전략이 만든 배정을 **시뮬레이터**로 평가한다.
(disruption이 있으면 해석 makespan 공식이 무효 → 실제 step 시뮬레이션 makespan 사용.)

- 결정적 베이스라인(Exact/SA/Greedy/Heuristic/Random): 무결(nominal) 모델로 배정 결정 후 disruption 하 평가.
- RL          : disruption 없이 학습된 RL 배정 (분포 이탈 확인용).
- RL-DR       : door disruption 하에서 새로 학습된 RL 배정 (강인성).

각 seed(0~19)는 트럭+고장 realization을 함께 고정 → 전 전략 동일 조건 비교.
makespan(@nominal)은 참고용 해석값, makespan(@disrupt)이 핵심 지표.

산출물: 표 + compound_disruption_results.json + viz 재생 JSON
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv
from env.policies import FIFOPolicy
from viz.export_simulation import capture_frame
from compound_baselines import (makespan_analytic, assign_exact, assign_sa,
                                 assign_greedy, assign_heuristic_vam, assign_random)
from rl_compound_assignment import (disrupt_cfg, load_policy, load_policy_dr,
                                    assign_rl, DR_DEMAND_MAX, DR_T_K, DR_PROB)

SEEDS = list(range(20))
NOMINAL = disrupt_cfg(prob=0.0)        # 고장 없음
DISRUPT = disrupt_cfg(prob=DR_PROB)    # 도어 고장 prob=0.05


def peek(cfg, seed):
    env = CrossDockEnv(config=cfg, seed=seed); env.reset()
    comp = [t for t in env.waiting_trucks if t.truck_type == "compound"]
    out = [t for t in env.outbound_waiting if t.truck_type == "outbound"]
    return comp, out, env.t_k, env.num_destinations


def sim(cfg, seed, override, frames_out=None):
    c = {**cfg, "compound_dest_override": override}
    env = CrossDockEnv(config=c, seed=seed); obs = env.reset()
    pols = [FIFOPolicy() for _ in range(env.num_lanes)]
    if frames_out is not None:
        fr = capture_frame(env, [0] * env.num_lanes, [0.0] * env.num_lanes)
        fr["outbound_waiting_count"] = len(env.outbound_waiting); frames_out.append(fr)
    done = False
    while not done:
        a = [pols[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
        obs, _, done, info = env.step(a)
        if frames_out is not None:
            fr = capture_frame(env, a, [0.0] * env.num_lanes)
            fr["outbound_waiting_count"] = len(env.outbound_waiting); frames_out.append(fr)
    return env.t, info["metrics"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rl = load_policy(); rldr = load_policy_dr()

    # 배정 전략: 이름 → assignment 생성 함수(comp,out,tk,nD,rng)
    strats = {
        "Exact":     lambda comp, out, tk, nD, rng: assign_exact(comp, out, tk, nD),
        "SA":        lambda comp, out, tk, nD, rng: assign_sa(comp, out, tk, nD, rng),
        "Greedy":    lambda comp, out, tk, nD, rng: assign_greedy(comp, nD),
        "Heuristic": lambda comp, out, tk, nD, rng: assign_heuristic_vam(comp, nD),
        "Random":    lambda comp, out, tk, nD, rng: assign_random(comp, nD, rng),
    }
    if rl is not None:
        strats["RL"] = lambda comp, out, tk, nD, rng: assign_rl(rl, comp, DR_DEMAND_MAX, nD)
    if rldr is not None:
        strats["RL-DR"] = lambda comp, out, tk, nD, rng: assign_rl(rldr, comp, DR_DEMAND_MAX, nD)

    print("=" * 74)
    print(f"Door Disruption 하 Compound 배정 강인성 (prob={DR_PROB}, "
          f"t_k={DR_T_K}, demand_max={DR_DEMAND_MAX}, seeds={len(SEEDS)})")
    if rldr is None:
        print("  ⚠️ RL-DR 가중치 없음 — `python -c \"from rl_compound_assignment import "
              "train_disrupt; train_disrupt()\"` 로 학습 후 재실행")
    print("=" * 74)

    nominal = {k: [] for k in strats}   # 해석 makespan (고장 없음)
    disrupt = {k: [] for k in strats}   # 시뮬 makespan (고장 있음)
    for seed in SEEDS:
        comp, out, tk, nD = peek(DISRUPT, seed)   # 트럭은 disruption 무관(동일 생성)
        rng = np.random.default_rng(2000 + seed)
        for name, fn in strats.items():
            a = fn(comp, out, tk, nD, rng)
            nominal[name].append(makespan_analytic(comp, out, tk, nD, a, partial=True))
            mk, _ = sim(DISRUPT, seed, a)
            disrupt[name].append(mk)

    order = [k for k in ["Exact", "SA", "Greedy", "RL", "RL-DR", "Heuristic", "Random"] if k in strats]
    best_d = min(np.mean(disrupt[k]) for k in order)
    print(f"\n{'전략':>10} | {'makespan@nominal':>16} | {'makespan@disrupt':>16} | "
          f"{'증가율':>7} | {'best 대비':>8}")
    print("-" * 74)
    results = {}
    for k in order:
        n = float(np.mean(nominal[k])); d = float(np.mean(disrupt[k]))
        inc = 100.0 * (d - n) / n
        gap = 100.0 * (d - best_d) / best_d
        results[k] = {"nominal": n, "disrupt": d, "increase_pct": inc, "gap_to_best_pct": gap,
                      "disrupt_std": float(np.std(disrupt[k]))}
        tag = " 🏅" if abs(d - best_d) < 1e-9 else ""
        print(f"{k:>10} | {n:16.1f} | {d:16.1f} | {inc:+6.1f}% | {gap:+7.2f}%{tag}")

    out_json = os.path.join(here, "compound_disruption_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out_json}")

    # 시각화 — 대표 seed 에서 Greedy vs RL-DR (disruption 하) 재생
    viz_seed = 3
    comp, out, tk, nD = peek(DISRUPT, viz_seed)
    for name in (["Greedy"] + (["RL-DR"] if rldr is not None else [])):
        a = strats[name](comp, out, tk, nD, np.random.default_rng(2000 + viz_seed))
        frames = []
        mk, metrics = sim(DISRUPT, viz_seed, a, frames_out=frames)
        data = {"meta": {"policy": f"disrupt-{name.lower()}", "strategy": name, "mode": "disrupt",
                         "seed": viz_seed, "makespan": int(mk), "num_steps": len(frames),
                         "num_lanes": nD, "num_inbound_doors": 5, "num_outbound_doors": nD,
                         "outbound_loading_time_max": DISRUPT["outbound_loading_time_max"],
                         "final_metrics": {kk: float(v) if isinstance(v, (int, float)) else v
                                           for kk, v in metrics.items()}},
                "frames": frames}
        p = os.path.join(here, "viz", f"sim_compound_disrupt_{name.lower().replace('-','')}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[viz] {name}: makespan={mk} (door_failures={int(metrics['disruption_door_failures'])})"
              f"  →  {os.path.relpath(p, here)}")


if __name__ == "__main__":
    main()
