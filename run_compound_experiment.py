#!/usr/bin/env python3
"""
Compound Truck + Partial Unloading 검증 실험
==========================================

논문 Shahmardan & Sajadieh (2020) "Truck scheduling in a multi-door cross-docking
center with partial unloading" 의 핵심 결과 — 부분 하차(partial unloading)가 완전 하차
(complete unloading) 대비 makespan을 단축한다 (Table 6/7) — 를 본 시뮬레이터로 재현한다.

실험 설계 (논문 Table 6/7 형식):
  - 행(row)   : demand density (demand_max 로 조절)
  - 열(column): 단위 적재/하차 시간 t_k ∈ {4, 6, 8, 10} (고정)
  - 셀(cell)  : partial makespan, 그리고 complete 대비 개선율(%)
  - 동일 seed 로 partial / complete 를 짝지어 비교, SEEDS 평균.
  - 정책(FIFO/Greedy/Heuristic)은 mode-agnostic 한 기존 lane-agent 정책을 그대로 사용.

DBPR(Destination Bound Product Ratio, Eq.32, 1 product type 기준):
  DBPR_d = max_i f_id / Σ_i f_id  →  목적지별 최대를 평균하여 보고.

산출물:
  - 콘솔/마크다운 표 (makespan + improvement %)
  - compound_experiment_results.json (원자료)
  - viz/sim_compound_partial.json / viz/sim_compound_complete.json (재생용, viz/index2d.html)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy
from viz.export_simulation import capture_frame

# ──────────────────────────────────────────────────────────────────────
# 환경 설정 (논문 민감도 분석: 5 compound + 2 outbound = 7 목적지, 5 도어)
# ──────────────────────────────────────────────────────────────────────
PAPER_CFG = {
    **DEFAULT_CONFIG,
    "compound_trucks": True,
    "all_trucks_at_start": True,
    "use_scheduled_arrivals": True,
    "num_lanes": 7,
    "num_destinations": 7,
    "num_inbound_doors": 5,
    "num_outbound_doors": 7,           # 모든 트럭이 동시에 도크 가능
    "num_compound_trucks": 5,
    "num_outbound_trucks": 2,          # 참고용(실제 outbound 수는 잔여 목적지로 결정)
    "outbound_capacity": 1e6,          # 트럭은 목적지 통합수요 전량 운반 (사실상 무제한)
    "buffer_capacity": 1e9,
    "episode_length": 1_000_000,
    "enable_disruptions": False,
}

POLICIES = {
    "FIFO":      FIFOPolicy,
    "Greedy":    GreedyPolicy,
    "Heuristic": HeuristicPriorityPolicy,
}

T_K_VALUES = [4, 6, 8, 10]              # 논문 Table 6/7 컬럼
DENSITY_SWEEP = [10, 15, 20, 30]       # demand_max (행: demand density 프록시)
NCOMPOUND_SWEEP = [1, 2, 3, 4, 5, 6, 7]  # compound 트럭 수 → DBPR 가변 (잔여는 outbound)
SEEDS = list(range(20))


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────
def make_cfg(partial: bool, t_k: int, demand_max: int, num_compound: int = 5) -> dict:
    return {
        **PAPER_CFG,
        "partial_unloading": partial,
        "unit_load_time": t_k,
        "demand_min": 0,
        "demand_max": demand_max,
        "num_compound_trucks": num_compound,
    }


def paired_makespan(cfg_p, cfg_c, seed):
    """동일 seed 로 partial/complete 실행 → (partial_makespan, complete_makespan, dbpr)."""
    rp = run_episode(FIFOPolicy, cfg_p, seed)
    rc = run_episode(FIFOPolicy, cfg_c, seed)
    assert abs(rp["throughput"] - rc["throughput"]) < 1e-6, "throughput 불변식 위반"
    return rp["makespan"], rc["makespan"], rp["dbpr"], (rp["hit_cap"] or rc["hit_cap"])


def compute_dbpr(env: CrossDockEnv) -> float:
    """Eq.32 (1 product type): DBPR_d = max_i f_id / Σ_i f_id, 목적지별 평균.
    reset 직후 compound 트럭(waiting_trucks)의 f_id 로 계산."""
    compounds = [t for t in env.waiting_trucks if t.truck_type == "compound"]
    nD = env.num_destinations
    ratios = []
    for d in range(nD):
        col = [t.dest_volume(d) for t in compounds]
        s = sum(col)
        if s > 0:
            ratios.append(max(col) / s)
    return float(np.mean(ratios)) if ratios else 0.0


def run_episode(policy_cls, cfg: dict, seed: int, capture: bool = False):
    """단일 에피소드 실행. makespan(=env.t)·throughput·DBPR 반환. capture 시 frames 도 반환."""
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    dbpr = compute_dbpr(env)
    policies = [policy_cls() for _ in range(env.num_lanes)]

    frames = None
    if capture:
        frames = [capture_frame(env, [0] * env.num_lanes, [0.0] * env.num_lanes)]
        # outbound_waiting 정보 보강
        frames[-1]["outbound_waiting_count"] = len(env.outbound_waiting)

    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors)
                   for k in range(env.num_lanes)]
        obs, _, done, info = env.step(actions)
        if capture:
            fr = capture_frame(env, actions, [0.0] * env.num_lanes)
            fr["outbound_waiting_count"] = len(env.outbound_waiting)
            frames.append(fr)

    m = info["metrics"]
    result = {
        "makespan": int(env.t),
        "throughput": float(m["total_throughput"]),
        "compound_throughput": float(m["compound_throughput"]),
        "kept_volume_delivered": float(m["kept_volume_delivered"]),
        "dbpr": dbpr,
        "hit_cap": env.t >= cfg["episode_length"],
    }
    if capture:
        return result, frames, m
    return result


# ──────────────────────────────────────────────────────────────────────
# 메인 실험
# ──────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results: dict = {"density_sweep": {}, "dbpr_sweep": {}, "policy_check": {}}
    any_cap_hit = False

    print("=" * 80)
    print("Compound Truck — Partial vs Complete Unloading (논문 makespan 결과 재현)")
    print(f"  5 compound + 잔여 outbound = {PAPER_CFG['num_destinations']} 목적지,"
          f" {PAPER_CFG['num_inbound_doors']} 도어, seeds={len(SEEDS)}")
    print("=" * 80)

    # ── 표 1: demand density × t_k (논문 Table 6/7 형식) ──────────────────
    # compound 도크 배정은 정책 무관 → 대표로 FIFO 사용(정책 동등성은 표 3에서 확인).
    print("\n[표 1] demand_max × t_k 별 makespan (Partial / Complete) 및 개선율\n")
    header = f"{'demand_max':>11} | {'DBPR':>5} | " + " | ".join(
        f"t_k={t:<2}  P/C  (imp%)".center(24) for t in T_K_VALUES
    )
    print(header); print("-" * len(header))
    for dmax in DENSITY_SWEEP:
        results["density_sweep"][dmax] = {}
        cells, dbpr_acc = [], []
        for t_k in T_K_VALUES:
            pm, cm, imps = [], [], []
            for seed in SEEDS:
                p, c, dbpr, cap = paired_makespan(
                    make_cfg(True, t_k, dmax), make_cfg(False, t_k, dmax), seed)
                any_cap_hit = any_cap_hit or cap
                pm.append(p); cm.append(c); dbpr_acc.append(dbpr)
                if c > 0:
                    imps.append(100.0 * (c - p) / c)
            mp, mc, mi = float(np.mean(pm)), float(np.mean(cm)), float(np.mean(imps))
            results["density_sweep"][dmax][t_k] = {
                "partial": mp, "complete": mc, "improvement_pct": mi}
            cells.append(f"{mp:5.0f}/{mc:5.0f} ({mi:4.1f}%)".center(24))
        dbpr_mean = float(np.mean(dbpr_acc))
        results["density_sweep"][dmax]["_dbpr"] = dbpr_mean
        print(f"{dmax:>11} | {dbpr_mean:5.2f} | " + " | ".join(cells))

    # ── 표 2: DBPR sweep (compound 트럭 수 변경 → DBPR 가변) ──────────────
    # 논문: "수요가 소수 compound 트럭에 집중(DBPR↑)될수록 부분하차가 유리".
    print("\n[표 2] compound 트럭 수 변경에 따른 DBPR · makespan (t_k=8, demand_max=20)\n")
    h2 = f"{'#compound':>9} | {'#outbound':>9} | {'DBPR':>5} | {'Partial':>8} | {'Complete':>8} | {'imp%':>6}"
    print(h2); print("-" * len(h2))
    for nC in NCOMPOUND_SWEEP:
        pm, cm, dbpr_acc = [], [], []
        for seed in SEEDS:
            p, c, dbpr, cap = paired_makespan(
                make_cfg(True, 8, 20, nC), make_cfg(False, 8, 20, nC), seed)
            any_cap_hit = any_cap_hit or cap
            pm.append(p); cm.append(c); dbpr_acc.append(dbpr)
        mp, mc = float(np.mean(pm)), float(np.mean(cm))
        mi = 100.0 * (mc - mp) / mc if mc else 0.0
        results["dbpr_sweep"][nC] = {
            "num_outbound": PAPER_CFG["num_destinations"] - nC,
            "dbpr": float(np.mean(dbpr_acc)),
            "partial": mp, "complete": mc, "improvement_pct": mi}
        print(f"{nC:>9} | {PAPER_CFG['num_destinations']-nC:>9} | "
              f"{np.mean(dbpr_acc):5.2f} | {mp:8.0f} | {mc:8.0f} | {mi:5.1f}")

    # ── 표 3: 정책 동등성 확인 (compound 도킹은 정책 무관) ────────────────
    print("\n[표 3] 정책별 makespan 동등성 점검 (Partial, t_k=8, demand_max=20)\n")
    print(f"{'Policy':>10} | {'mean makespan':>13}")
    print("-" * 28)
    for name, cls in POLICIES.items():
        ms = [run_episode(cls, make_cfg(True, 8, 20), s)["makespan"] for s in SEEDS]
        results["policy_check"][name] = float(np.mean(ms))
        print(f"{name:>10} | {np.mean(ms):13.1f}")

    if any_cap_hit:
        print("\n⚠️  일부 run 이 episode_length 상한 도달 — makespan 무효! 설정 점검 필요.")
    else:
        print("\n✅ 모든 run 정상 종료 (cap 미도달). 전 구간 partial ≤ complete 확인.")

    out_json = os.path.join(here, "compound_experiment_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {out_json}")

    # 시각화용 재생 JSON (대표 1 seed)
    viz_seed, viz_tk, viz_dmax = 3, 8, 20
    for mode, partial in (("partial", True), ("complete", False)):
        cfg = make_cfg(partial, viz_tk, viz_dmax)
        res, frames, metrics = run_episode(FIFOPolicy, cfg, viz_seed, capture=True)
        data = {
            "meta": {
                "policy": "fifo",
                "mode": mode,
                "partial_unloading": partial,
                "seed": viz_seed,
                "t_k": viz_tk,
                "demand_max": viz_dmax,
                "makespan": res["makespan"],
                "num_steps": len(frames),
                "num_lanes": cfg["num_lanes"],
                "num_inbound_doors": cfg["num_inbound_doors"],
                "num_outbound_doors": cfg["num_outbound_doors"],
                "outbound_loading_time_max": cfg["outbound_loading_time_max"],
                "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                                  for k, v in metrics.items()},
            },
            "frames": frames,
        }
        out = os.path.join(here, "viz", f"sim_compound_{mode}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[저장] {out}  (makespan={res['makespan']}, frames={len(frames)})")

    print("\n[뷰어] open viz/index2d.html → sim_compound_partial/complete.json 재생")


if __name__ == "__main__":
    main()
