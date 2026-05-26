#!/usr/bin/env python3
"""
run_all_experiments_2stage.py
8-Door 2-Stage 환경에서 모든 정책 실험 수행 후 viz/ 에 JSON 저장.

실행:
    python run_all_experiments_2stage.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import RandomPolicy, FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy

VIZ_DIR = os.path.join(ROOT, "viz")
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 8-Door 2-Stage 환경 설정
# ─────────────────────────────────────────────────────────────────
CFG_8D = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 8,
    "num_outbound_doors": 5,
    "arrival_count_min": 133,
    "arrival_count_max": 187,
}

N_BENCH = 20   # 벤치마크 에피소드 수
SEED_EVAL = 42  # viz JSON 생성용 단일 시드

# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────

def capture_frame(env, actions, rewards):
    return {
        "t": env.t,
        "buffer": float(env.buffer),
        "buffer_capacity": env.buffer_capacity,
        "waiting_trucks": [
            {"arrival_time": int(t.arrival_time),
             "shipments": {str(k): int(v) for k, v in t.shipments.items()},
             "total_volume": int(t.total_volume()),
             "is_rush": bool(getattr(t, "is_rush", False))}
            for t in env.waiting_trucks
        ],
        "scheduled_trucks": [
            {"arrival_time": int(t.arrival_time),
             "shipments": {str(k): int(v) for k, v in t.shipments.items()},
             "total_volume": int(t.total_volume())}
            for t in env.arrival_schedule
        ],
        "doors": [
            {"door_id": d.door_id, "is_busy": bool(d.is_busy),
             "is_failed": bool(d.is_failed), "failure_remaining": int(d.failure_remaining),
             "remaining_time": int(d.remaining_time), "assigned_lane": int(d.assigned_lane),
             "assigned_truck_volume": int(d.assigned_truck.total_volume()) if d.assigned_truck else 0,
             "assigned_truck_shipments": (
                 {str(k): int(v) for k, v in d.assigned_truck.shipments.items()}
                 if d.assigned_truck else {})}
            for d in env.doors
        ],
        "outbound_doors": [
            {"door_id": od.door_id, "is_busy": bool(od.is_busy),
             "assigned_dest": int(od.assigned_dest),
             "loaded": float(od.loaded), "fill_rate": float(od.fill_rate),
             "loading_timer": int(od.loading_timer), "capacity": float(od.capacity)}
            for od in env.outbound_doors
        ],
        "lanes": [
            {"lane_id": int(lane.lane_id),
             "queue_volume": float(lane.queue_volume),
             "congestion": float(lane.congestion),
             "outbound_door": next(
                 ({"door_id": od.door_id, "loaded": float(od.loaded),
                   "fill_rate": float(od.fill_rate), "loading_timer": int(od.loading_timer),
                   "capacity": float(od.capacity)}
                  for od in env.outbound_doors
                  if od.is_busy and od.assigned_dest == k), None)}
            for k, lane in enumerate(env.lanes)
        ],
        "actions": list(actions),
        "rewards": [float(r) for r in rewards],
        "disruptions": list(env.disruption_log),
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in env.metrics.items()},
    }


def run_episode_frames(policy_factory, cfg, seed=42):
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    policies = policy_factory(env)
    frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))
    return frames, env.metrics, env


def run_episode_metrics(policy_factory, cfg, seed):
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    policies = policy_factory(env)
    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
        obs, _, done, _ = env.step(actions)
    m = env.metrics.copy()
    m["door_utilization"]         = env.door_utilization
    m["outbound_door_utilization"] = env.outbound_door_utilization
    m["avg_dwell_time"]           = env.avg_dwell_time
    m["avg_fill_rate"]            = env.avg_fill_rate
    return m


def aggregate(results):
    keys = list(results[0].keys())
    return {k: {"mean": float(np.mean([r[k] for r in results])),
                "std":  float(np.std( [r[k] for r in results]))} for k in keys}


def save_viz_json(frames, metrics, env, policy_name, seed, path):
    avg_fill = metrics["total_fill_rate"] / max(metrics["outbound_departures"], 1)
    data = {
        "meta": {
            "policy": policy_name, "seed": seed,
            "num_steps": len(frames),
            "num_lanes": env.num_lanes,
            "num_inbound_doors": env.num_inbound_doors,
            "num_outbound_doors": env.num_outbound_doors,
            "stage": 2,
            "final_metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            },
        },
        "frames": frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  → {path}  throughput={metrics['total_throughput']:.1f}  "
          f"fill={avg_fill:.1%}  overflow={metrics['buffer_overflow_count']}  "
          f"empty={metrics['empty_departures']}")


# ─────────────────────────────────────────────────────────────────
# 1. 베이스라인 정책 비교 + viz JSON
# ─────────────────────────────────────────────────────────────────

BASELINE_POLICIES = {
    "random":    lambda env: [RandomPolicy(np.random.default_rng(i)) for i in range(env.num_lanes)],
    "fifo":      lambda env: [FIFOPolicy()                            for _ in range(env.num_lanes)],
    "greedy":    lambda env: [GreedyPolicy()                          for _ in range(env.num_lanes)],
    "heuristic": lambda env: [HeuristicPriorityPolicy()               for _ in range(env.num_lanes)],
}

print("=" * 62)
print("[1] 베이스라인 정책 비교 (8-Door 2-Stage, 20 에피소드)")
print("=" * 62)
bench_results = {}
for name, factory in BASELINE_POLICIES.items():
    results = [run_episode_metrics(factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
    bench_results[name] = aggregate(results)
    tp = bench_results[name]["total_throughput"]
    fr = bench_results[name]["avg_fill_rate"]
    print(f"  {name:12s}  throughput={tp['mean']:>7.1f}±{tp['std']:<5.1f}  "
          f"fill={fr['mean']*100:.1f}%")

    # viz JSON (seed=42 단일 에피소드)
    frames, metrics, env = run_episode_frames(factory, CFG_8D, seed=SEED_EVAL)
    save_viz_json(frames, metrics, env, name, SEED_EVAL,
                  os.path.join(VIZ_DIR, f"sim_2stage_{name}.json"))

# ─────────────────────────────────────────────────────────────────
# 2. MILP
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[2] MILP (8-Door 2-Stage)")
print("=" * 62)
try:
    from mip.solve_mip import run_episode_mip, solve_assignment

    def _run_mip_bench(seed):
        import time as _time
        cfg_with_count = {**CFG_8D}
        # run_episode_mip 는 별도 env를 만들므로 config 패치
        from env.crossdock_env import CrossDockEnv as _Env
        env = _Env(config=cfg_with_count, seed=seed)
        obs = env.reset()
        total_mip_t = 0.0; calls = 0; done = False
        while not done:
            idle_doors = [d for d in env.doors if not d.is_busy]
            waiting    = env.waiting_trucks
            if idle_doors and waiting:
                t0 = _time.perf_counter()
                assigned = solve_assignment(waiting, env.doors, env.outbound_trucks,
                                            max(env.buffer_capacity - env.buffer, 0.0))
                total_mip_t += _time.perf_counter() - t0; calls += 1
                assigned_set = set()
                front = []
                for idx in assigned:
                    if idx is not None and idx not in assigned_set:
                        front.append(waiting[idx]); assigned_set.add(idx)
                rest = [t for i, t in enumerate(waiting) if i not in assigned_set]
                env.waiting_trucks = front + rest
                req = set()
                for idx in assigned:
                    if idx is not None and idx < len(waiting) and waiting[idx].shipments:
                        req.add(min(waiting[idx].shipments,
                                    key=lambda k: env.outbound_trucks[k].departure_timer))
                if len(req) < len(idle_doors):
                    for k in sorted(range(env.num_lanes),
                                    key=lambda k: env.outbound_trucks[k].departure_timer):
                        if len(req) >= len(idle_doors): break
                        req.add(k)
                actions = [1 if k in req else 0 for k in range(env.num_lanes)]
            else:
                actions = [0] * env.num_lanes
            obs, _, done, _ = env.step(actions)
        m = env.metrics.copy()
        m["door_utilization"] = env.door_utilization
        m["outbound_door_utilization"] = env.outbound_door_utilization
        m["avg_dwell_time"] = env.avg_dwell_time
        m["avg_fill_rate"]  = env.avg_fill_rate
        m["avg_mip_ms"] = total_mip_t / max(calls, 1) * 1000
        return m, env

    mip_bench = [_run_mip_bench(ep)[0] for ep in range(N_BENCH)]
    bench_results["mip"] = aggregate(mip_bench)
    tp = bench_results["mip"]["total_throughput"]
    fr = bench_results["mip"]["avg_fill_rate"]
    print(f"  {'mip':12s}  throughput={tp['mean']:>7.1f}±{tp['std']:<5.1f}  "
          f"fill={fr['mean']*100:.1f}%")

    # viz JSON
    frames_mip, metrics_mip, env_mip = None, None, None
    from mip.solve_mip import capture_frame as mip_cap
    env_viz = CrossDockEnv(config=CFG_8D, seed=SEED_EVAL)
    obs_viz = env_viz.reset()
    import time as _t
    frames_mip = [mip_cap(env_viz, [0]*env_viz.num_lanes, [0.0]*env_viz.num_lanes)]
    total_mt = 0.0; calls_viz = 0; done_viz = False
    while not done_viz:
        idle_doors = [d for d in env_viz.doors if not d.is_busy]
        waiting = env_viz.waiting_trucks
        if idle_doors and waiting:
            t0 = _t.perf_counter()
            assigned = solve_assignment(waiting, env_viz.doors, env_viz.outbound_trucks,
                                        max(env_viz.buffer_capacity - env_viz.buffer, 0.0))
            total_mt += _t.perf_counter() - t0; calls_viz += 1
            assigned_set = set()
            front = []
            for idx in assigned:
                if idx is not None and idx not in assigned_set:
                    front.append(waiting[idx]); assigned_set.add(idx)
            rest = [t for i, t in enumerate(waiting) if i not in assigned_set]
            env_viz.waiting_trucks = front + rest
            req = set()
            for idx in assigned:
                if idx is not None and idx < len(waiting) and waiting[idx].shipments:
                    req.add(min(waiting[idx].shipments,
                                key=lambda k: env_viz.outbound_trucks[k].departure_timer))
            if len(req) < len(idle_doors):
                for k in sorted(range(env_viz.num_lanes),
                                key=lambda k: env_viz.outbound_trucks[k].departure_timer):
                    if len(req) >= len(idle_doors): break
                    req.add(k)
            actions_viz = [1 if k in req else 0 for k in range(env_viz.num_lanes)]
        else:
            actions_viz = [0] * env_viz.num_lanes
        obs_viz, rewards_viz, done_viz, _ = env_viz.step(actions_viz)
        frames_mip.append(mip_cap(env_viz, actions_viz, rewards_viz))
    save_viz_json(frames_mip, env_viz.metrics, env_viz, "mip", SEED_EVAL,
                  os.path.join(VIZ_DIR, "sim_2stage_mip.json"))
except ImportError as e:
    print(f"  MILP 건너뜀 (pulp 미설치): {e}")

# ─────────────────────────────────────────────────────────────────
# 3. RL 학습 (8-Door 2-Stage, 2000 에피소드)
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[3] RL 학습 (8-Door 2-Stage, 2000 에피소드)")
print("=" * 62)
from rl.train_rl import train as rl_train
from rl.networks import NumpyMLP
from rl.rl_policy import QLearningPolicy, normalize_obs

rl_result = rl_train(
    num_episodes=2000,
    lr=5e-4,
    seed=42,
    save_dir="checkpoints_2stage_8door",
    env_config=CFG_8D,
)
rl_net = rl_result["net"]

# RL 벤치마크
def rl_factory(env, net=rl_net):
    p = QLearningPolicy(net=net, epsilon=0.0)
    return [p] * env.num_lanes

rl_bench = [run_episode_metrics(rl_factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
bench_results["rl"] = aggregate(rl_bench)
tp = bench_results["rl"]["total_throughput"]
fr = bench_results["rl"]["avg_fill_rate"]
print(f"  {'rl':12s}  throughput={tp['mean']:>7.1f}±{tp['std']:<5.1f}  "
      f"fill={fr['mean']*100:.1f}%")

# RL viz JSON
frames_rl, metrics_rl, env_rl = run_episode_frames(rl_factory, CFG_8D, seed=SEED_EVAL)
save_viz_json(frames_rl, metrics_rl, env_rl, "rl_2stage", SEED_EVAL,
              os.path.join(VIZ_DIR, "sim_2stage_rl.json"))

# ─────────────────────────────────────────────────────────────────
# 4. GA 학습 (8-Door 2-Stage)
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[4] GA 학습 (8-Door 2-Stage, pop=50 gen=100)")
print("=" * 62)
from ga.train_ga import run_ga, CONFIG_8DOOR as GA_CFG_8D
from ga.ga_policy import GAPolicy

# GA_CFG_8D 는 기존 3도어 기준이므로 8도어 config 로 덮어씀
GA_CFG_8D_ACTUAL = {**GA_CFG_8D, **CFG_8D}

best_genes, best_fitness, ga_history = run_ga(
    pop_size=50, n_gen=100, n_eval=8, seed=0,
    cfg=GA_CFG_8D_ACTUAL, verbose=True,
)

buf_cap = GA_CFG_8D_ACTUAL["buffer_capacity"]
def ga_factory(env, genes=best_genes, buf_cap=buf_cap):
    return [GAPolicy(genes, buffer_capacity=buf_cap) for _ in range(env.num_lanes)]

ga_bench = [run_episode_metrics(ga_factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
bench_results["ga"] = aggregate(ga_bench)
tp = bench_results["ga"]["total_throughput"]
fr = bench_results["ga"]["avg_fill_rate"]
print(f"  {'ga':12s}  throughput={tp['mean']:>7.1f}±{tp['std']:<5.1f}  "
      f"fill={fr['mean']*100:.1f}%")

# GA viz JSON
frames_ga, metrics_ga, env_ga = run_episode_frames(ga_factory, CFG_8D, seed=SEED_EVAL)
save_viz_json(frames_ga, metrics_ga, env_ga, "ga_2stage", SEED_EVAL,
              os.path.join(VIZ_DIR, "sim_2stage_ga.json"))

# GA 유전자 저장
ga_genes_path = os.path.join(ROOT, "ga", "best_genes_2stage.json")
with open(ga_genes_path, "w") as f:
    from ga.ga_policy import GENE_NAMES
    json.dump({"genes": best_genes.tolist(), "gene_names": GENE_NAMES,
               "fitness": float(best_fitness), "history": ga_history,
               "config": "8door_2stage"}, f, indent=2)
print(f"  GA 유전자 저장: {ga_genes_path}")

# ─────────────────────────────────────────────────────────────────
# 5. 최종 벤치마크 요약 출력 + 저장
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[결과] 8-Door 2-Stage 정책 비교 (20 에피소드)")
print("=" * 62)
cols = ["total_throughput", "avg_fill_rate", "empty_departures",
        "buffer_overflow_count", "door_utilization", "outbound_door_utilization"]
labels = ["처리량(CBM)", "탑재율", "빈출발", "오버플로우", "In-DoorUtil", "Out-DoorUtil"]

header = f"{'정책':12s}" + "".join(f"{lbl:>14s}" for lbl in labels)
print(header)
print("-" * len(header))
for name, agg in bench_results.items():
    row = f"{name:12s}"
    for col in cols:
        if col not in agg:
            row += f"{'N/A':>14s}"; continue
        mv = agg[col]
        if col == "avg_fill_rate":
            row += f"  {mv['mean']*100:>6.1f}%±{mv['std']*100:<4.1f}"
        elif col in ("door_utilization", "outbound_door_utilization"):
            row += f"  {mv['mean']*100:>6.1f}%±{mv['std']*100:<4.1f}"
        else:
            row += f"  {mv['mean']:>8.1f}±{mv['std']:<4.1f}"
    print(row)

# 요약 JSON 저장
summary_path = os.path.join(VIZ_DIR, "benchmark_2stage_8door.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        name: {k: {"mean": float(v["mean"]), "std": float(v["std"])}
               for k, v in agg.items()}
        for name, agg in bench_results.items()
    }, f, indent=2, ensure_ascii=False)
print(f"\n[저장] 벤치마크 요약: {summary_path}")
print("[저장] viz JSON 파일: viz/sim_2stage_*.json")
