"""
train_ga_dr.py — Domain Randomization GA

적합도 평가 시 disruption_door_failure_prob를 [DR_MIN, DR_MAX]에서 균등 샘플링.
기존 best_genes_2stage.json 유전자를 초기 개체군 씨앗으로 사용.
결과: ga/best_genes_2stage_dr.json

실행:
    python ga/train_ga_dr.py
    python ga/train_ga_dr.py --pop 30 --gen 50
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from ga.ga_policy import GAPolicy, GENE_BOUNDS, GENE_NAMES, HEURISTIC_GENES, N_GENES

# ─────────────────────────────────────────────────────
# 환경 설정 (실험과 동일한 3-door 설정)
# ─────────────────────────────────────────────────────
BASE_CFG = {
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
    "disruption_door_failure_duration_min": 10,
    "disruption_door_failure_duration_max": 20,
}

DR_MIN = 0.02
DR_MAX = 0.25

BOUNDS_LO = GENE_BOUNDS[:, 0]
BOUNDS_HI = GENE_BOUNDS[:, 1]

# GA 하이퍼파라미터
POP_SIZE   = 30
N_GEN      = 50
N_EVAL     = 8
TOURN_SIZE = 3
CROSS_RATE = 0.8
MUT_SIGMA  = 0.2
ELITE_K    = 3

OUTPUT_PATH = os.path.join(ROOT, "ga", "best_genes_2stage_dr.json")


# ─────────────────────────────────────────────────────
# 적합도 함수 — 매 에피소드 disruption_prob 랜덤 샘플링
# ─────────────────────────────────────────────────────
def evaluate(genes: np.ndarray, n_eval: int, seed_offset: int, rng: np.random.Generator) -> float:
    total_ticks = 0
    for i in range(n_eval):
        d_prob = float(rng.uniform(DR_MIN, DR_MAX))
        cfg = {**BASE_CFG, "disruption_door_failure_prob": d_prob}
        env = CrossDockEnv(config=cfg, seed=seed_offset + i)
        obs = env.reset()
        n = env.num_lanes
        policies = [GAPolicy(genes) for _ in range(n)]
        done = False
        while not done:
            actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(n)]
            obs, _, done, _ = env.step(actions)
        total_ticks += env.t
    return -(total_ticks / n_eval)


# ─────────────────────────────────────────────────────
# GA 연산자
# ─────────────────────────────────────────────────────
def tournament_select(pop, fitness, rng):
    idx = rng.integers(0, len(pop), size=TOURN_SIZE)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def crossover(p1, p2, rng):
    if rng.random() < CROSS_RATE:
        mask = rng.random(N_GENES) < 0.5
        return np.where(mask, p1, p2)
    return p1.copy()


def mutate(genes, rng):
    return np.clip(genes + rng.normal(0.0, MUT_SIGMA, size=N_GENES), BOUNDS_LO, BOUNDS_HI)


# ─────────────────────────────────────────────────────
# 메인 GA 루프
# ─────────────────────────────────────────────────────
def run_ga_dr(pop_size=POP_SIZE, n_gen=N_GEN, n_eval=N_EVAL, seed=0):
    rng = np.random.default_rng(seed)

    # 기존 유전자(best_genes_2stage.json)를 씨앗으로 활용
    prior_path = os.path.join(ROOT, "ga", "best_genes_2stage.json")
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            prior_genes = np.array(json.load(f)["genes"])
        print(f"[GA-DR] 기존 유전자 로드: {prior_path}")
    else:
        prior_genes = HEURISTIC_GENES.copy()

    pop = rng.uniform(BOUNDS_LO, BOUNDS_HI, size=(pop_size, N_GENES))
    pop[0] = prior_genes
    pop[1] = HEURISTIC_GENES.copy()

    print(f"\nGA Domain Randomization 학습 시작")
    print(f"  disruption_prob 범위: [{DR_MIN}, {DR_MAX}]")
    print(f"  pop={pop_size}, gen={n_gen}, eval_per_gen={n_eval}")
    print(f"{'Gen':>5} {'Best Ticks':>12} {'Mean Ticks':>12} {'개선':>5}")
    print("-" * 38)

    t0 = time.perf_counter()
    fitness = np.array([evaluate(pop[i], n_eval, seed_offset=i*n_eval, rng=rng)
                        for i in range(pop_size)])

    best_fitness = float(fitness.max())
    best_genes   = pop[np.argmax(fitness)].copy()
    history = []

    for gen in range(n_gen):
        elite_idx = np.argsort(fitness)[-ELITE_K:]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < pop_size:
            child = mutate(crossover(
                tournament_select(pop, fitness, rng),
                tournament_select(pop, fitness, rng),
                rng), rng)
            new_pop.append(child)

        pop = np.array(new_pop[:pop_size])
        # 세대마다 seed_offset 달리해 평가 다양성 확보
        fitness = np.array([evaluate(pop[i], n_eval,
                                     seed_offset=1000 + gen*pop_size*n_eval + i*n_eval,
                                     rng=rng)
                            for i in range(pop_size)])

        gen_best = float(fitness.max())
        improved = gen_best > best_fitness
        if improved:
            best_fitness = gen_best
            best_genes   = pop[np.argmax(fitness)].copy()

        history.append({"gen": gen+1, "best": gen_best, "mean": float(fitness.mean())})
        marker = " ★" if improved else ""
        print(f"{gen+1:>5} {-gen_best:>12.1f} {-fitness.mean():>12.1f}{marker}")

    elapsed = time.perf_counter() - t0
    print(f"\n학습 완료 ({elapsed:.0f}s)  best ticks = {-best_fitness:.1f}")
    print("최적 유전자:")
    for name, val in zip(GENE_NAMES, best_genes):
        print(f"  {name:25s} = {val:+.4f}")

    # 저장
    result = {
        "genes": best_genes.tolist(),
        "gene_names": GENE_NAMES,
        "fitness": float(best_fitness),
        "dr_range": [DR_MIN, DR_MAX],
        "ga_config": {"pop_size": pop_size, "n_gen": n_gen, "n_eval": n_eval, "seed": seed},
        "history": history,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[저장] {OUTPUT_PATH}")
    return best_genes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop",  type=int, default=POP_SIZE)
    parser.add_argument("--gen",  type=int, default=N_GEN)
    parser.add_argument("--eval", type=int, default=N_EVAL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_ga_dr(args.pop, args.gen, args.eval, args.seed)
