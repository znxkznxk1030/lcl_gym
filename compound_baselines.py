"""
Compound Truck 목적지 배정(assignment) 베이스라인 — 논문 Shahmardan & Sajadieh (2020).

논문이 makespan 최소화를 위해 비교한 솔루션 접근(Exact / Heuristic / Simulated Annealing)을
본 시뮬레이터의 compound 목적지 배정 문제에 적용한다. 배정이 결정되면 makespan은 결정적이며,
해석적 공식이 시뮬레이터 makespan과 정확히 일치(검증됨)하므로 탐색에 해석적 평가를 사용한다.

배정(assignment) = {compound_idx: dest} (서로 다른 목적지). 잔여 목적지는 outbound 트럭이 담당.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# makespan 해석 모델 (시뮬레이터와 동일, 검증됨)
# ──────────────────────────────────────────────────────────────────────
def makespan_analytic(comp, out, t_k: float, nD: int, assign: Dict[int, int],
                      partial: bool = True) -> float:
    """병렬 2단계(하차→적재, 게이트) makespan.

    inbound  = max_i (DE_i + unloaded_i · t_k)         (compound ≤ 도어 → 병렬)
    outbound = max over 담당트럭 (DL + reload · t_k)
      - compound 담당 d: reload = 통합수요_d − kept (partial) / 통합수요_d (complete)
      - outbound 담당 d: reload = 통합수요_d
    makespan = inbound + outbound
    """
    cons = {d: sum(c.dest_volume(d) for c in comp) for d in range(nD)}
    inbound = 0.0
    for i, c in enumerate(comp):
        kept = c.dest_volume(assign[i]) if partial else 0.0
        inbound = max(inbound, c.DE + (c.total_volume() - kept) * t_k)
    served = set(assign.values())
    outbound = 0.0
    for i, c in enumerate(comp):
        d = assign[i]
        kept = c.dest_volume(d) if partial else 0.0
        outbound = max(outbound, c.DL + (cons[d] - kept) * t_k)
    oi = 0
    for d in range(nD):
        if d in served:
            continue
        DL = out[oi].DL if oi < len(out) else 5
        oi += 1
        outbound = max(outbound, DL + cons[d] * t_k)
    return inbound + outbound


# ──────────────────────────────────────────────────────────────────────
# 배정 전략 (논문 베이스라인)
# ──────────────────────────────────────────────────────────────────────
def assign_random(comp, nD: int, rng) -> Dict[int, int]:
    """무작위 distinct 배정 (성능 하한 기준)."""
    I = len(comp)
    dests = list(rng.permutation(nD))[:I]
    return {i: int(dests[i]) for i in range(I)}


def assign_greedy(comp, nD: int) -> Dict[int, int]:
    """Greedy (ours): 본 환경 `_build_compound_schedule` 기본 배정.

    모든 (트럭 i, 목적지 d, 보유량 f_i,d)를 보유량 내림차순 정렬 후, distinct 제약을
    지키며 큰 것부터 탐욕적으로 배정 (보유량 합 최대화). VAA보다 단순하지만 makespan 최적에 더 근접.
    """
    I = len(comp)
    cand = sorted(
        [(i, d, comp[i].dest_volume(d)) for i in range(I) for d in range(nD)],
        key=lambda x: -x[2],
    )
    assign: Dict[int, int] = {}
    used: set = set()
    for i, d, _v in cand:
        if i in assign or d in used:
            continue
        assign[i] = d
        used.add(d)
        if len(assign) == min(I, nD):
            break
    return assign


def assign_heuristic_vam(comp, nD: int) -> Dict[int, int]:
    """Heuristic (H): Vogel 근사법(VAA)으로 보유량(f_i,d) 합을 최대화하는 배정.

    논문 Step 1–2: 각 compound 트럭을 부분 하차/적재 시간이 최소가 되는 목적지에 배정
    (= 그 트럭이 가장 많이 싣고 온 목적지를 보유 → 처리시간 최소화). 후회값(regret) 기반.
    """
    I = len(comp)
    benefit = np.array([[comp[i].dest_volume(d) for d in range(nD)] for i in range(I)], float)
    rows = set(range(I))
    cols = set(range(nD))
    assign: Dict[int, int] = {}
    while rows:
        best_regret, best_kind, best_idx = -1.0, None, None
        # 행(트럭) 후회값
        for i in rows:
            vals = sorted((benefit[i][d] for d in cols), reverse=True)
            reg = (vals[0] - vals[1]) if len(vals) > 1 else vals[0]
            if reg > best_regret:
                best_regret, best_kind, best_idx = reg, "row", i
        # 열(목적지) 후회값
        for d in cols:
            vals = sorted((benefit[i][d] for i in rows), reverse=True)
            reg = (vals[0] - vals[1]) if len(vals) > 1 else vals[0]
            if reg > best_regret:
                best_regret, best_kind, best_idx = reg, "col", d
        if best_kind == "row":
            i = best_idx
            d = max(cols, key=lambda d: benefit[i][d])
        else:
            d = best_idx
            i = max(rows, key=lambda i: benefit[i][d])
        assign[i] = d
        rows.discard(i); cols.discard(d)
    return assign


def assign_exact(comp, out, t_k: float, nD: int, partial: bool = True) -> Dict[int, int]:
    """Exact: 모든 distinct 배정 전수탐색 → makespan 최소 (소규모 최적해)."""
    I = len(comp)
    best, best_a = float("inf"), None
    for perm in itertools.permutations(range(nD), I):
        a = {i: perm[i] for i in range(I)}
        m = makespan_analytic(comp, out, t_k, nD, a, partial)
        if m < best:
            best, best_a = m, a
    return best_a


def assign_sa(comp, out, t_k: float, nD: int, rng, partial: bool = True,
              iters: int = 400, T0: float = 200.0, alpha: float = 0.99) -> Dict[int, int]:
    """Simulated Annealing: Heuristic 초기해에서 이웃탐색(목적지 swap/reassign)으로 makespan 개선.

    논문과 동일하게 H 해를 초기해로 사용(좋은 출발점 → 짧은 수렴). 이웃구조:
      - reassign: 한 트럭을 미사용 목적지로 이동
      - swap    : 두 트럭의 목적지 교환
    """
    I = len(comp)
    cur = assign_heuristic_vam(comp, nD)
    cur_m = makespan_analytic(comp, out, t_k, nD, cur, partial)
    best, best_m = dict(cur), cur_m
    T = T0
    all_dests = set(range(nD))
    for _ in range(iters):
        cand = dict(cur)
        used = set(cand.values())
        free = list(all_dests - used)
        if free and rng.random() < 0.5:
            i = int(rng.integers(0, I))
            cand[i] = int(rng.choice(free))           # reassign
        else:
            i, j = rng.choice(I, size=2, replace=False)
            cand[int(i)], cand[int(j)] = cand[int(j)], cand[int(i)]  # swap
        cand_m = makespan_analytic(comp, out, t_k, nD, cand, partial)
        if cand_m <= cur_m or rng.random() < np.exp(-(cand_m - cur_m) / max(T, 1e-6)):
            cur, cur_m = cand, cand_m
            if cur_m < best_m:
                best, best_m = dict(cur), cur_m
        T *= alpha
    return best


STRATEGIES = {
    "Random":    lambda comp, out, t_k, nD, rng: assign_random(comp, nD, rng),
    "Heuristic": lambda comp, out, t_k, nD, rng: assign_heuristic_vam(comp, nD),
    "Greedy":    lambda comp, out, t_k, nD, rng: assign_greedy(comp, nD),   # 우리 모델 기본 배정
    "SA":        lambda comp, out, t_k, nD, rng: assign_sa(comp, out, t_k, nD, rng),
    "Exact":     lambda comp, out, t_k, nD, rng: assign_exact(comp, out, t_k, nD),
}
