"""
GA 학습 스크립트 — 8-Door Cross-Docking 환경에서 GAPolicy 파라미터 최적화.

알고리즘:
  - 표현: 실수 벡터 (6개 유전자)
  - 선택: 토너먼트 선택 (k=3)
  - 교차: 균등 교차 (crossover_rate=0.8)
  - 변이: 가우시안 노이즈 (σ=0.2, 경계 클리핑)
  - 엘리트: 상위 3개 개체 무조건 보존

사용법:
    python ga/train_ga.py
    python ga/train_ga.py --pop 60 --gen 150 --eval 10
    python ga/train_ga.py --pop 30 --gen 10 --eval 3   # 빠른 테스트
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
from env.policies import HeuristicPriorityPolicy
from ga.ga_policy import (
    GAPolicy,
    GENE_BOUNDS,
    GENE_NAMES,
    HEURISTIC_GENES,
    N_GENES,
)


# ─────────────────────────────────────────────────────────────────
# 8-Door 환경 설정 (README 실험과 동일)
# ─────────────────────────────────────────────────────────────────

CONFIG_8DOOR: dict = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 8,
    "arrival_count_min": 133,
    "arrival_count_max": 187,
    # buffer_capacity = 60.0 (DEFAULT_CONFIG 유지 — README 실험과 동일 조건)
}

BOUNDS_LO: np.ndarray = GENE_BOUNDS[:, 0]
BOUNDS_HI: np.ndarray = GENE_BOUNDS[:, 1]

# ─────────────────────────────────────────────────────────────────
# GA 하이퍼파라미터
# ─────────────────────────────────────────────────────────────────

POP_SIZE   = 50
N_GEN      = 100
N_EVAL     = 8    # 적합도 평가 에피소드 수
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
    cfg: dict,
    env: CrossDockEnv | None = None,
) -> float:
    """N 에피소드 평균 처리량 (CBM) 반환."""
    num_lanes = cfg["num_lanes"]
    num_doors = cfg["num_inbound_doors"]
    buf_cap   = cfg["buffer_capacity"]

    if env is None:
        env = CrossDockEnv(config=cfg, seed=seeds[0])

    total = 0.0
    for seed in seeds:
        env._seed = seed
        obs = env.reset()
        policies = [GAPolicy(genes, buffer_capacity=buf_cap) for _ in range(num_lanes)]
        done = False
        while not done:
            actions = [policies[k].act(obs[k], num_doors) for k in range(num_lanes)]
            obs, _, done, _ = env.step(actions)
        total += env.metrics["total_throughput"]
    return total / len(seeds)


# ─────────────────────────────────────────────────────────────────
# GA 연산자
# ─────────────────────────────────────────────────────────────────

def tournament_select(
    pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=TOURN_SIZE)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
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
    cfg:      dict | None = None,
    verbose:  bool = True,
) -> tuple[np.ndarray, float, list[dict]]:
    """
    Returns
    -------
    best_genes : np.ndarray, shape (6,)
    best_fitness : float  (평균 처리량 CBM)
    history : list of {"gen", "best", "mean"}
    """
    cfg = cfg or CONFIG_8DOOR
    rng = np.random.default_rng(seed)
    eval_seeds = list(range(n_eval))

    # 초기 개체군 — 기존 Heuristic 유전자를 씨앗으로 포함
    pop = rng.uniform(BOUNDS_LO, BOUNDS_HI, size=(pop_size, N_GENES))
    pop[0] = HEURISTIC_GENES.copy()

    env = CrossDockEnv(config=cfg, seed=0)

    if verbose:
        print(f"[GA] 개체군 초기화 완료 (pop={pop_size}, gen={n_gen}, eval={n_eval})")
        print("[GA] 초기 적합도 계산 중...")

    t0 = time.perf_counter()
    fitness = np.array([evaluate(pop[i], eval_seeds, cfg, env) for i in range(pop_size)])
    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"[GA] 초기 평가 완료 ({elapsed:.1f}s) | "
            f"best={fitness.max():.1f}  mean={fitness.mean():.1f}\n"
        )

    best_fitness = float(fitness.max())
    best_genes   = pop[np.argmax(fitness)].copy()
    history: list[dict] = []

    for gen in range(n_gen):
        # 엘리트 보존
        elite_idx = np.argsort(fitness)[-ELITE_K:]
        new_pop   = [pop[i].copy() for i in elite_idx]

        # 나머지 개체 생성
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
        fitness = np.array([evaluate(pop[i], eval_seeds, cfg, env) for i in range(pop_size)])

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
                f"best={gen_best:.1f}  mean={gen_mean:.1f} | "
                f"all-time best={best_fitness:.1f}{marker}"
            )

    return best_genes, best_fitness, history


# ─────────────────────────────────────────────────────────────────
# 벤치마크: GA vs Heuristic
# ─────────────────────────────────────────────────────────────────

def benchmark(
    genes: np.ndarray,
    cfg: dict,
    n_episodes: int = 20,
    seed_offset: int = 200,
) -> dict[str, dict]:
    """
    GA 정책과 Heuristic을 동일한 시드 집합에서 비교.

    Returns
    -------
    {"GA": agg_dict, "Heuristic": agg_dict}
    각 agg_dict: {"metric_name": {"mean": ..., "std": ...}}
    """
    num_lanes = cfg["num_lanes"]
    num_doors = cfg["num_inbound_doors"]
    buf_cap   = cfg["buffer_capacity"]
    env = CrossDockEnv(config=cfg, seed=0)

    results: dict[str, list[dict]] = {"GA": [], "Heuristic": []}
    for ep in range(n_episodes):
        seed = seed_offset + ep
        for policy_name, make_policies in [
            ("GA",        lambda: [GAPolicy(genes, buf_cap)         for _ in range(num_lanes)]),
            ("Heuristic", lambda: [HeuristicPriorityPolicy()        for _ in range(num_lanes)]),
        ]:
            env._seed = seed
            obs = env.reset()
            policies = make_policies()
            done = False
            while not done:
                actions = [policies[k].act(obs[k], num_doors) for k in range(num_lanes)]
                obs, _, done, _ = env.step(actions)
            m = env.metrics.copy()
            m["door_utilization"] = env.door_utilization
            m["avg_dwell_time"]   = env.avg_dwell_time
            m["avg_fill_rate"]    = env.avg_fill_rate
            results[policy_name].append(m)

    def agg(lst: list[dict]) -> dict:
        keys = list(lst[0].keys())
        return {
            k: {"mean": float(np.mean([r[k] for r in lst])),
                "std":  float(np.std( [r[k] for r in lst]))}
            for k in keys
        }

    return {name: agg(lst) for name, lst in results.items()}


# ─────────────────────────────────────────────────────────────────
# 결과 출력 헬퍼
# ─────────────────────────────────────────────────────────────────

def _print_benchmark(bm: dict[str, dict]) -> None:
    cols = [
        ("total_throughput",     "처리량(CBM)"),
        ("avg_fill_rate",        "탑재율"),
        ("empty_departures",     "빈출발"),
        ("buffer_overflow_count","오버플로우"),
    ]
    header = f"{'정책':12s}" + "".join(f"{lbl:>16s}" for _, lbl in cols)
    print(header)
    print("-" * len(header))
    for policy_name, agg in bm.items():
        row = f"{policy_name:12s}"
        for key, _ in cols:
            mv = agg[key]
            if key == "avg_fill_rate":
                row += f"  {mv['mean']*100:>7.1f}%±{mv['std']*100:<4.1f}"
            else:
                row += f"  {mv['mean']:>8.1f}±{mv['std']:<4.1f}"
        print(row)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="8-Door 환경에서 GA로 GAPolicy 파라미터 최적화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pop",            type=int,   default=POP_SIZE,
                        help="개체군 크기")
    parser.add_argument("--gen",            type=int,   default=N_GEN,
                        help="세대 수")
    parser.add_argument("--eval",           type=int,   default=N_EVAL,
                        help="세대별 적합도 평가 에피소드 수")
    parser.add_argument("--seed",           type=int,   default=0,
                        help="GA 난수 시드")
    parser.add_argument("--bench-episodes", type=int,   default=20,
                        help="최종 벤치마크 에피소드 수")
    parser.add_argument("--output",         type=str,   default="ga/best_genes.json",
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    print("=" * 62)
    print("[GA] Cross-Docking 8-Door Policy Optimization")
    print(f"     pop={args.pop}  gen={args.gen}  eval={args.eval}  seed={args.seed}")
    print(f"     환경: {CONFIG_8DOOR['num_inbound_doors']}도어 "
          f"{CONFIG_8DOOR['arrival_count_min']}~{CONFIG_8DOOR['arrival_count_max']}대")
    print("=" * 62 + "\n")

    # ── GA 학습 ──────────────────────────────────────────────────
    best_genes, best_fitness, history = run_ga(
        pop_size=args.pop,
        n_gen=args.gen,
        n_eval=args.eval,
        seed=args.seed,
        cfg=CONFIG_8DOOR,
        verbose=True,
    )

    print("\n" + "=" * 62)
    print("[결과] 최적 유전자")
    print("=" * 62)
    for name, val in zip(GENE_NAMES, best_genes):
        print(f"  {name:15s} = {val:+.4f}")
    print(f"\n  학습 피트니스 = {best_fitness:.1f} CBM  (eval seeds 0~{args.eval-1})")

    # ── 벤치마크 ─────────────────────────────────────────────────
    print(f"\n[벤치마크] {args.bench_episodes} 에피소드 (GA vs Heuristic) ...")
    bm = benchmark(best_genes, CONFIG_8DOOR, n_episodes=args.bench_episodes)

    print()
    _print_benchmark(bm)

    # ── 저장 ─────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
        "env_config": "8door_133~187trucks",
        "history":    history,
        "benchmark":  {
            pol: {
                k: {"mean": float(v["mean"]), "std": float(v["std"])}
                for k, v in agg.items()
            }
            for pol, agg in bm.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] {args.output}")


if __name__ == "__main__":
    main()
