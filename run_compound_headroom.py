#!/usr/bin/env python3
"""
Door Disruption 하 배정 Headroom 측정 (진단 실험).

질문: disruption 환경에서 '배정을 잘 고르면' makespan을 줄일 여지가 정말 있는가?
방법: 각 seed에서
  - a_nom  = nominal(고장 없음) 최적 배정 (해석식 전수탐색)
  - a_best = disruption 하 최적 배정 (시뮬레이터로 2520개 전수탐색, 해당 seed의 고장 realization 기준)
  를 구하고, 둘을 disruption 하에서 비교한다.

  headroom = makespan(a_nom)@disrupt − makespan(a_best)@disrupt   (≥ 0)

a_best는 그 seed의 고장 패턴을 '미리 아는' clairvoyant 최적 → headroom의 상한(upper bound).
이 상한조차 작다면, 미래 고장을 모르는 학습 정책(RL-DR)이 배정으로 얻을 수 있는 이득은 ~0이다.
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compound_baselines import makespan_analytic, assign_exact, assign_greedy
from rl_compound_assignment import disrupt_cfg, peek_env_trucks, sim_makespan_env, DR_PROB

SEEDS = list(range(6))
DISRUPT = disrupt_cfg(prob=DR_PROB)
NOMINAL = disrupt_cfg(prob=0.0)
I, nD = 5, 7


def peek_all(cfg, seed):
    from env.crossdock_env import CrossDockEnv
    env = CrossDockEnv(config=cfg, seed=seed); env.reset()
    comp = [t for t in env.waiting_trucks if t.truck_type == "compound"]
    out = [t for t in env.outbound_waiting if t.truck_type == "outbound"]
    return comp, out, env.t_k, env.num_destinations


def main():
    print("=" * 76)
    print(f"Headroom 측정 — disruption(prob={DR_PROB}) 하 배정 최적화 여지 (seeds={len(SEEDS)})")
    print("=" * 76)
    print(f"{'seed':>4} | {'nomExact@dis':>12} | {'greedy@dis':>10} | {'best@dis':>9} | "
          f"{'headroom':>9} | {'head%':>6}")
    print("-" * 70)

    rows = []
    for seed in SEEDS:
        comp, out, t_k, _ = peek_all(DISRUPT, seed)
        a_nom = assign_exact(comp, out, t_k, nD)                 # nominal 최적 (해석)
        a_grd = assign_greedy(comp, nD)
        mk_nom = sim_makespan_env(DISRUPT, seed, a_nom)
        mk_grd = sim_makespan_env(DISRUPT, seed, a_grd)
        # disruption 하 전수탐색 (clairvoyant 최적)
        best, mk_best = None, float("inf")
        t0 = time.time()
        for perm in itertools.permutations(range(nD), I):
            a = {i: perm[i] for i in range(I)}
            m = sim_makespan_env(DISRUPT, seed, a)
            if m < mk_best:
                mk_best, best = m, a
        head = mk_nom - mk_best
        rows.append((mk_nom, mk_grd, mk_best, head))
        print(f"{seed:>4} | {mk_nom:12.0f} | {mk_grd:10.0f} | {mk_best:9.0f} | "
              f"{head:9.0f} | {100*head/mk_nom:5.1f}%  ({time.time()-t0:.0f}s, 2520 sims)",
              flush=True)

    arr = np.array(rows, float)
    nom, grd, best, head = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()
    print("-" * 70)
    print(f"{'평균':>4} | {nom:12.1f} | {grd:10.1f} | {best:9.1f} | {head:9.1f} | {100*head/nom:5.1f}%")
    print()
    print(f"[해석] nominal-최적 배정의 disruption makespan = {nom:.1f}")
    print(f"       clairvoyant(고장 패턴 미리 앎) 최적 배정 = {best:.1f}")
    print(f"       → 배정 최적화의 '상한 여지(headroom)' = {head:.1f} ticks ({100*head/nom:.1f}%)")
    print(f"       미래 고장을 모르는 RL-DR이 얻을 수 있는 실제 이득은 이 상한보다 작다.")


if __name__ == "__main__":
    main()
