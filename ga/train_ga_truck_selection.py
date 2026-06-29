"""
train_ga_truck_selection.py — Truck Selection 모드 GA 학습

Lane Request 모드의 train_ga.py와 별도로 동작하며,
"어떤 트럭을 먼저 처리할 것인가"를 GA로 최적화한다.

사용법:
    python ga/train_ga_truck_selection.py
    python ga/train_ga_truck_selection.py --pop 50 --gen 100 --eval 8
    python ga/train_ga_truck_selection.py --compare   # Lane-mode GA와 성능 비교 출력
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import FIFOPolicy, HeuristicPriorityPolicy
from ga.truck_selection_policy import (
    TruckSelectionGAPolicy,
    GENE_BOUNDS,
    GENE_NAMES,
    HEURISTIC_GENES,
    N_GENES,
)
from env.policies import TruckSelFIFOPolicy, TruckSelHeuristicPolicy


# ─────────────────────────────────────────────────────────────────
# 환경 설정 — Lane-mode 실험(20260531_005)과 동일 조건 (비교 기준)
# ─────────────────────────────────────────────────────────────────

CFG_TRUCK_SEL: dict = {
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
    "use_truck_selection": True,      # Truck Selection 모드
    "top_k_trucks": 15,
    "enable_disruptions": True,
    "disruption_door_failure": True,
    "disruption_door_failure_prob": 0.02,
    "disruption_door_failure_duration_min": 10,
    "disruption_door_failure_duration_max": 20,
}

BOUNDS_LO: np.ndarray = GENE_BOUNDS[:, 0]
BOUNDS_HI: np.ndarray = GENE_BOUNDS[:, 1]

# ─────────────────────────────────────────────────────────────────
# GA 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────

POP_SIZE   = 30
N_GEN      = 50
N_EVAL     = 5
TOURN_SIZE = 3
CROSS_RATE = 0.8
MUT_SIGMA  = 0.2
ELITE_K    = 3


# ─────────────────────────────────────────────────────────────────
# 적합도 함수
# ─────────────────────────────────────────────────────────────────

def evaluate(
    genes: np.ndarray,
    seeds: list[int],
    env: CrossDockEnv,
) -> float:
    """N 에피소드 평균 tick 수의 음수 반환 (적을수록 높은 점수)."""
    n_agents  = env.top_k_trucks
    num_doors = env.num_inbound_doors
    total_ticks = 0
    for seed in seeds:
        env._seed = seed
        obs = env.reset()
        policies = [TruckSelectionGAPolicy(genes) for _ in range(n_agents)]
        done = False
        while not done:
            actions = [policies[k].act(obs[k], num_doors) for k in range(n_agents)]
            obs, _, done, _ = env.step(actions)
        total_ticks += env.t
    return -(total_ticks / len(seeds))


# ─────────────────────────────────────────────────────────────────
# GA 연산자
# ─────────────────────────────────────────────────────────────────

def tournament_select(
    pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=TOURN_SIZE)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < CROSS_RATE:
        mask = rng.random(N_GENES) < 0.5
        return np.where(mask, p1, p2)
    return p1.copy()


def mutate(genes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, MUT_SIGMA, size=N_GENES)
    return np.clip(genes + noise, BOUNDS_LO, BOUNDS_HI)


# ─────────────────────────────────────────────────────────────────
# 메인 GA 루프
# ─────────────────────────────────────────────────────────────────

def run_ga(
    pop_size: int = POP_SIZE,
    n_gen:    int = N_GEN,
    n_eval:   int = N_EVAL,
    seed:     int = 0,
    verbose:  bool = True,
) -> tuple[np.ndarray, float, list[dict]]:
    rng        = np.random.default_rng(seed)
    eval_seeds = list(range(n_eval))
    env        = CrossDockEnv(config=CFG_TRUCK_SEL, seed=0)

    # 초기 개체군 — Heuristic 유전자를 씨앗으로 포함
    pop    = rng.uniform(BOUNDS_LO, BOUNDS_HI, size=(pop_size, N_GENES))
    pop[0] = HEURISTIC_GENES.copy()

    if verbose:
        print(f"[GA-TruckSel] 개체군 초기화 완료 (pop={pop_size}, gen={n_gen}, eval={n_eval})")
        print("[GA-TruckSel] 초기 적합도 계산 중...")

    t0 = time.perf_counter()
    fitness = np.array([evaluate(pop[i], eval_seeds, env) for i in range(pop_size)])
    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"[GA-TruckSel] 초기 평가 완료 ({elapsed:.1f}s) | "
            f"best={-fitness.max():.1f}tick  mean={-fitness.mean():.1f}tick\n"
        )

    best_fitness = float(fitness.max())
    best_genes   = pop[np.argmax(fitness)].copy()
    history: list[dict] = []

    for gen in range(n_gen):
        elite_idx = np.argsort(fitness)[-ELITE_K:]
        new_pop   = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            child = mutate(
                crossover(
                    tournament_select(pop, fitness, rng),
                    tournament_select(pop, fitness, rng),
                    rng,
                ),
                rng,
            )
            new_pop.append(child)

        pop     = np.array(new_pop[:pop_size])
        fitness = np.array([evaluate(pop[i], eval_seeds, env) for i in range(pop_size)])

        gen_best = float(fitness.max())
        gen_mean = float(fitness.mean())
        improved = gen_best > best_fitness
        if improved:
            best_fitness = gen_best
            best_genes   = pop[np.argmax(fitness)].copy()

        history.append({"gen": gen + 1, "best": gen_best, "mean": gen_mean})

        if verbose:
            marker = " ★" if improved else ""
            print(
                f"Gen {gen+1:3d}/{n_gen} | "
                f"best={-gen_best:.1f}tick  mean={-gen_mean:.1f}tick | "
                f"all-time best={-best_fitness:.1f}tick{marker}"
            )

    return best_genes, best_fitness, history


# ─────────────────────────────────────────────────────────────────
# 벤치마크
# ─────────────────────────────────────────────────────────────────

def benchmark(
    genes: np.ndarray,
    n_episodes: int = 20,
    seed_offset: int = 200,
) -> dict[str, dict]:
    """
    GA-TruckSel vs TruckSel-FIFO vs TruckSel-Heuristic 비교.

    Returns
    -------
    {"GA-TruckSel": agg, "TruckSel-FIFO": agg, "TruckSel-Heuristic": agg}
    """
    env       = CrossDockEnv(config=CFG_TRUCK_SEL, seed=0)
    n_agents  = env.top_k_trucks
    num_doors = env.num_inbound_doors

    policies_map = {
        "GA-TruckSel":       lambda: [TruckSelectionGAPolicy(genes)   for _ in range(n_agents)],
        "TruckSel-FIFO":     lambda: [TruckSelFIFOPolicy()            for _ in range(n_agents)],
        "TruckSel-Heuristic":lambda: [TruckSelHeuristicPolicy()       for _ in range(n_agents)],
    }

    results: dict[str, list[dict]] = {k: [] for k in policies_map}

    for ep in range(n_episodes):
        seed = seed_offset + ep
        for name, make in policies_map.items():
            env._seed = seed
            obs = env.reset()
            policies = make()
            done = False
            while not done:
                actions = [policies[k].act(obs[k], num_doors) for k in range(n_agents)]
                obs, _, done, _ = env.step(actions)
            m = env.metrics.copy()
            m["total_ticks"] = env.t
            results[name].append(m)

    def agg(lst: list[dict]) -> dict:
        keys = list(lst[0].keys())
        return {
            k: {
                "mean": float(np.mean([r[k] for r in lst])),
                "std":  float(np.std( [r[k] for r in lst])),
            }
            for k in keys
        }

    return {name: agg(lst) for name, lst in results.items()}


def _print_benchmark(bm: dict[str, dict]) -> None:
    cols = [
        ("total_ticks",          "Ticks"),
        ("total_throughput",     "처리량(CBM)"),
        ("avg_fill_rate",        "탑재율"),
        ("empty_departures",     "빈출발"),
        ("outbound_departures",  "출발횟수"),
    ]
    header = f"{'정책':20s}" + "".join(f"{lbl:>18s}" for _, lbl in cols)
    print(header)
    print("-" * len(header))
    for name, agg_d in sorted(bm.items(), key=lambda x: x[1]["total_ticks"]["mean"]):
        row = f"{name:20s}"
        for key, _ in cols:
            if key not in agg_d:
                row += f"  {'N/A':>14s}"
                continue
            mv = agg_d[key]
            if key == "avg_fill_rate":
                row += f"  {mv['mean']*100:>7.1f}%±{mv['std']*100:<5.1f}"
            else:
                row += f"  {mv['mean']:>9.1f}±{mv['std']:<5.1f}"
        print(row)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truck Selection 모드 GA 학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pop",            type=int,  default=POP_SIZE)
    parser.add_argument("--gen",            type=int,  default=N_GEN)
    parser.add_argument("--eval",           type=int,  default=N_EVAL)
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--bench-episodes", type=int,  default=20)
    parser.add_argument("--output",         type=str,  default="ga/best_genes_truck_selection.json")
    args = parser.parse_args()

    print("=" * 62)
    print("[GA-TruckSel] Truck Selection Policy Optimization")
    print(f"     pop={args.pop}  gen={args.gen}  eval={args.eval}  seed={args.seed}")
    print("=" * 62 + "\n")

    # ── GA 학습 ──────────────────────────────────────────────────
    best_genes, best_fitness, history = run_ga(
        pop_size=args.pop,
        n_gen=args.gen,
        n_eval=args.eval,
        seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 62)
    print("[결과] 최적 유전자")
    print("=" * 62)
    for name, val in zip(GENE_NAMES, best_genes):
        print(f"  {name:20s} = {val:+.4f}")
    print(f"\n  학습 피트니스 = {-best_fitness:.1f} tick (eval seeds 0~{args.eval-1})")

    # ── 벤치마크 ─────────────────────────────────────────────────
    print(f"\n[벤치마크] {args.bench_episodes} 에피소드 ...")
    bm = benchmark(best_genes, n_episodes=args.bench_episodes)
    print()
    _print_benchmark(bm)

    # ── 저장 ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = {
        "genes":      best_genes.tolist(),
        "gene_names": GENE_NAMES,
        "fitness":    float(best_fitness),
        "ga_config": {
            "pop_size": args.pop,
            "n_gen":    args.gen,
            "n_eval":   args.eval,
            "seed":     args.seed,
        },
        "env_config": "truck_selection_3door_50~70trucks",
        "history":    history,
        "benchmark":  {
            pol: {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in agg.items()}
            for pol, agg in bm.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    main()
