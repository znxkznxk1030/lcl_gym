#!/usr/bin/env python3
"""
Compound Truck 배정 베이스라인 실험 — 논문 Shahmardan & Sajadieh (2020) 재현.

논문이 makespan 최소화를 위해 비교한 솔루션 접근(Exact / Heuristic / SA / Random)을 본 시뮬레이터의
compound 목적지 배정 문제에 적용하여 makespan으로 비교한다(논문 Table 2/4 형식).

검증: 배정이 정해지면 makespan은 결정적이며 해석적 공식이 시뮬레이터 makespan과 정확히 일치함을
대표 seed에서 실제 step 시뮬레이션으로 재확인한다.

산출물: 콘솔/마크다운 표 + compound_baseline_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import FIFOPolicy
from viz.export_simulation import capture_frame
from compound_baselines import STRATEGIES, makespan_analytic
from rl_compound_assignment import load_policy, assign_rl

# 학습된 RL 배정 정책 (있으면 비교에 포함). 없으면 `python rl_compound_assignment.py` 로 학습.
_RL_NET = load_policy()

# 전략 집합: 논문 베이스라인(STRATEGIES, demand_max 무시) + RL(ours, demand_max 사용)
#   통일 시그니처: fn(comp, out, t_k, nD, rng, demand_max) -> assignment dict
STRATS = {name: (lambda comp, out, t_k, nD, rng, dmax, _f=fn: _f(comp, out, t_k, nD, rng))
          for name, fn in STRATEGIES.items()}
if _RL_NET is not None:
    STRATS["RL"] = lambda comp, out, t_k, nD, rng, dmax: assign_rl(_RL_NET, comp, dmax, nD)

BASE_CFG = {
    **DEFAULT_CONFIG,
    "compound_trucks": True,
    "partial_unloading": True,
    "all_trucks_at_start": True,
    "use_scheduled_arrivals": True,
    "num_lanes": 7,
    "num_destinations": 7,
    "num_inbound_doors": 5,
    "num_outbound_doors": 7,
    "num_compound_trucks": 5,
    "outbound_capacity": 1e6,
    "buffer_capacity": 1e9,
    "episode_length": 1_000_000,
    "enable_disruptions": False,
}

T_K = 8
DEMAND_MAX = 20
SEEDS = list(range(20))


def peek_trucks(cfg, seed):
    """reset 후 compound/outbound 트럭과 t_k·nD 추출 (배정 알고리즘 입력)."""
    env = CrossDockEnv(config=cfg, seed=seed)
    env.reset()
    comp = [t for t in env.waiting_trucks if t.truck_type == "compound"]
    out = [t for t in env.outbound_waiting if t.truck_type == "outbound"]
    return comp, out, env.t_k, env.num_destinations


def sim_makespan(cfg, seed, override):
    """실제 step 시뮬레이션 makespan (해석 공식 검증용)."""
    c = {**cfg, "compound_dest_override": override}
    env = CrossDockEnv(config=c, seed=seed)
    obs = env.reset()
    pols = [FIFOPolicy() for _ in range(env.num_lanes)]
    while True:
        obs, _, done, _ = env.step(
            [pols[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)])
        if done:
            break
    return env.t


def make_cfg(t_k, demand_max):
    return {**BASE_CFG, "unit_load_time": t_k, "demand_min": 0, "demand_max": demand_max}


def export_viz(cfg, seed, override, strategy, makespan, here):
    """배정 override 로 시뮬레이션을 재생하며 프레임 캡처 → viz JSON 저장."""
    c = {**cfg, "compound_dest_override": override}
    env = CrossDockEnv(config=c, seed=seed)
    obs = env.reset()
    pols = [FIFOPolicy() for _ in range(env.num_lanes)]
    frames = [capture_frame(env, [0] * env.num_lanes, [0.0] * env.num_lanes)]
    frames[-1]["outbound_waiting_count"] = len(env.outbound_waiting)
    done = False
    while not done:
        actions = [pols[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
        obs, _, done, info = env.step(actions)
        fr = capture_frame(env, actions, [0.0] * env.num_lanes)
        fr["outbound_waiting_count"] = len(env.outbound_waiting)
        frames.append(fr)
    data = {
        "meta": {
            "policy": f"assign-{strategy.lower()}",
            "strategy": strategy,
            "assignment": {int(k): int(v) for k, v in override.items()},
            "seed": seed,
            "makespan": int(env.t),
            "num_steps": len(frames),
            "num_lanes": c["num_lanes"],
            "num_inbound_doors": c["num_inbound_doors"],
            "num_outbound_doors": c["num_outbound_doors"],
            "outbound_loading_time_max": c["outbound_loading_time_max"],
            "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                              for k, v in info["metrics"].items()},
        },
        "frames": frames,
    }
    out = os.path.join(here, "viz", f"sim_compound_assign_{strategy.lower()}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out, int(env.t)


def evaluate(t_k, demand_max, verify_sim=False):
    """각 전략을 모든 seed에 적용 → makespan 통계 + Exact 대비 gap."""
    cfg = make_cfg(t_k, demand_max)
    per = {name: [] for name in STRATS}
    runtime = {name: 0.0 for name in STRATS}
    mismatches = 0
    for seed in SEEDS:
        comp, out, tk, nD = peek_trucks(cfg, seed)
        rng = np.random.default_rng(1000 + seed)
        for name, fn in STRATS.items():
            t0 = time.time()
            a = fn(comp, out, tk, nD, rng, demand_max)
            runtime[name] += time.time() - t0
            m = makespan_analytic(comp, out, tk, nD, a, partial=True)
            per[name].append(m)
            if verify_sim:
                sm = sim_makespan(cfg, seed, a)
                if abs(sm - m) > 1e-6:
                    mismatches += 1
    return per, runtime, mismatches


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rl_on = "RL" in STRATS
    print("=" * 72)
    print("Compound Truck 배정 베이스라인 (Exact / SA / Greedy(ours) / "
          + ("RL(ours) / " if rl_on else "") + "Heuristic / Random)")
    print(f"  5 compound + 2 outbound = 7 목적지, 5 도어, t_k={T_K},"
          f" demand_max={DEMAND_MAX}, seeds={len(SEEDS)}")
    if not rl_on:
        print("  ⚠️ RL 가중치 없음 — `python rl_compound_assignment.py` 로 학습 후 재실행하면 RL 포함")
    print("=" * 72)

    # 메인 비교 (sim 검증 포함)
    per, runtime, mismatches = evaluate(T_K, DEMAND_MAX, verify_sim=True)
    assert mismatches == 0, f"해석 makespan != sim makespan ({mismatches}건)"
    print(f"\n[검증] 해석 makespan == 시뮬레이션 makespan "
          f"({len(STRATS)*len(SEEDS)}건 전부 일치) ✅\n")

    opt = np.array(per["Exact"], float)
    order = ["Exact", "SA", "Greedy"] + (["RL"] if rl_on else []) + ["Heuristic", "Random"]
    print(f"{'전략':>10} | {'avg makespan':>12} | {'std':>6} | {'gap to Exact':>12} | {'ms/seed':>8}")
    print("-" * 62)
    results = {"main": {}, "sweep": {}}
    for name in order:
        arr = np.array(per[name], float)
        gap = 100.0 * np.mean((arr - opt) / opt)
        ms = 1000.0 * runtime[name] / len(SEEDS)
        results["main"][name] = {
            "avg": float(arr.mean()), "std": float(arr.std()),
            "gap_pct": float(gap), "ms_per_seed": float(ms)}
        print(f"{name:>10} | {arr.mean():12.1f} | {arr.std():6.1f} | "
              f"{gap:+11.2f}% | {ms:8.2f}")

    # t_k / demand 스윕 (Exact 대비 gap 추이)
    print(f"\n[스윕] 설정별 Exact 대비 gap(%) — SA / Greedy(ours) / Heuristic / Random\n")
    print(f"{'(t_k, dmax)':>14} | {'Exact avg':>10} | {'SA':>7} | {'Greedy':>8} | {'Heuristic':>10} | {'Random':>8}")
    print("-" * 66)
    for t_k in (4, 8):
        for dmax in (10, 20, 30):
            p, _, _ = evaluate(t_k, dmax)
            o = np.array(p["Exact"], float)
            def g(n): return 100.0 * np.mean((np.array(p[n], float) - o) / o)
            results["sweep"][f"{t_k}_{dmax}"] = {
                "exact_avg": float(o.mean()),
                "sa_gap": float(g("SA")),
                "greedy_gap": float(g("Greedy")),
                "heuristic_gap": float(g("Heuristic")),
                "random_gap": float(g("Random"))}
            print(f"{'(%d, %d)'%(t_k,dmax):>14} | {o.mean():10.0f} | "
                  f"{g('SA'):+6.2f}% | {g('Greedy'):+7.2f}% | {g('Heuristic'):+9.2f}% | {g('Random'):+7.2f}%")

    out_json = os.path.join(here, "compound_baseline_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out_json}")

    # 시각화용 재생 JSON — 대표 1 seed 에서 전략별 배정으로 시뮬레이션 캡처
    viz_seed = 3
    cfg = make_cfg(T_K, DEMAND_MAX)
    comp, out, tk, nD = peek_trucks(cfg, viz_seed)
    print(f"\n[viz] seed={viz_seed} 전략별 배정 재생 JSON 저장 (동일 트럭, 배정만 상이)")
    for name, fn in STRATEGIES.items():
        rng = np.random.default_rng(1000 + viz_seed)
        a = fn(comp, out, tk, nD, rng)
        m_an = makespan_analytic(comp, out, tk, nD, a, partial=True)
        path, m_sim = export_viz(cfg, viz_seed, a, name, m_an, here)
        assert abs(m_sim - m_an) < 1e-6
        print(f"  {name:>10}: makespan={m_sim}  →  {os.path.relpath(path, here)}")


if __name__ == "__main__":
    main()
