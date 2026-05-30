#!/usr/bin/env python3
"""
run_all_experiments_2stage.py
Lane-mode (3-action: 0=skip / 1=request_inbound / 2=boost_outbound) 실험.
  - num_outbound_doors=3 (< num_lanes=5) → 아웃바운드 도크 희소성 생성
  - buffer_capacity=80 → 버퍼 포화 시 overflow 패널티
  - RL이 인바운드 수용 타이밍 + 아웃바운드 우선순위를 동시에 최적화
결과는 viz/<YYYYMMDD_NNN>/ 하위 폴더에 저장.
"""
from __future__ import annotations
import json, os, sys, time
from datetime import date
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import RandomPolicy, FIFOPolicy, GreedyPolicy, HeuristicPriorityPolicy

VIZ_DIR = os.path.join(ROOT, "viz")
os.makedirs(VIZ_DIR, exist_ok=True)

# ── dated run 디렉토리 생성 ────────────────────────────────────────
today = date.today().strftime("%Y%m%d")
serial = 1
while True:
    run_dirname = f"{today}_{serial:03d}"
    RUN_DIR = os.path.join(VIZ_DIR, run_dirname)
    if not os.path.exists(RUN_DIR):
        break
    serial += 1
os.makedirs(RUN_DIR)
print(f"[저장 경로] {RUN_DIR}")

# ─────────────────────────────────────────────────────────────────
# Lane-mode 3-action 환경 설정
# ─────────────────────────────────────────────────────────────────
CFG_8D = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 3,           # 인바운드 병목
    "num_outbound_doors": 3,          # 아웃바운드 희소 (< num_lanes=5)
    "buffer_capacity": 80.0,          # 유한 버퍼 → overflow 패널티 활성화
    "arrival_count_min": 50,
    "arrival_count_max": 70,
    "all_trucks_at_start": False,
    "arrival_pattern": "clustered",
    "arrival_cluster_count": 4,
    "arrival_time_window": 300,
    "compound_trucks": False,
    "use_truck_selection": False,     # Lane mode (3-action)
    # 작업자 부족 이벤트
    "enable_disruptions": True,
    "disruption_door_failure": True,
    "disruption_door_failure_prob": 0.02,
    "disruption_door_failure_duration_min": 10,
    "disruption_door_failure_duration_max": 20,
}

N_BENCH   = 20
SEED_EVAL = 42

# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────

def _n_agents(env: CrossDockEnv) -> int:
    return env.top_k_trucks if env.use_truck_selection else env.num_lanes


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
        "outbound_waiting_count": len(getattr(env, "outbound_waiting", [])),
        "actions": list(actions),
        "rewards": [float(r) for r in rewards],
        "disruptions": list(env.disruption_log),
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in env.metrics.items()},
    }


def run_episode_frames(policy_factory, cfg, seed=42):
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    n = _n_agents(env)
    policies = policy_factory(env)
    frames = [capture_frame(env, [0]*n, [0.0]*n)]
    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(n)]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))
    return frames, env.metrics, env


def run_episode_metrics(policy_factory, cfg, seed):
    env = CrossDockEnv(config=cfg, seed=seed)
    obs = env.reset()
    n = _n_agents(env)
    policies = policy_factory(env)
    done = False
    while not done:
        actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(n)]
        obs, _, done, _ = env.step(actions)
    m = env.metrics.copy()
    m["total_ticks"]               = float(env.t)
    m["door_utilization"]          = env.door_utilization
    m["outbound_door_utilization"] = env.outbound_door_utilization
    m["avg_dwell_time"]            = env.avg_dwell_time
    m["avg_fill_rate"]             = env.avg_fill_rate
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
    print(f"  → {path}  ticks={env.t}  throughput={metrics['total_throughput']:.1f}  "
          f"fill={avg_fill:.1%}  empty={metrics['empty_departures']}")


# ─────────────────────────────────────────────────────────────────
# 1. 베이스라인 정책 비교 + viz JSON
# ─────────────────────────────────────────────────────────────────

BASELINE_POLICIES = {
    "random":    lambda env: [RandomPolicy(np.random.default_rng(i))
                              for i in range(_n_agents(env))],
    "fifo":      lambda env: [FIFOPolicy()               for _ in range(_n_agents(env))],
    "greedy":    lambda env: [GreedyPolicy()             for _ in range(_n_agents(env))],
    "heuristic": lambda env: [HeuristicPriorityPolicy()  for _ in range(_n_agents(env))],
}

print("=" * 62)
print("[1] 베이스라인 정책 비교 (Lane-mode 3-action, 20 에피소드)")
print("=" * 62)
bench_results = {}
for name, factory in BASELINE_POLICIES.items():
    results = [run_episode_metrics(factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
    bench_results[name] = aggregate(results)
    tk = bench_results[name]["total_ticks"]
    print(f"  {name:12s}  ticks={tk['mean']:>7.1f}±{tk['std']:<5.1f}")

    frames, metrics, env = run_episode_frames(factory, CFG_8D, seed=SEED_EVAL)
    save_viz_json(frames, metrics, env, name, SEED_EVAL,
                  os.path.join(RUN_DIR, f"sim_2stage_{name}.json"))

# ─────────────────────────────────────────────────────────────────
# 2. MILP
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[2] MILP (Lane-mode: 인바운드 재정렬 + 아웃바운드 긴급 우선)")
print("=" * 62)
try:
    from mip.solve_mip import solve_assignment
    from mip.solve_mip import capture_frame as mip_cap

    MILP_URGENT_TIMER = 10  # 이 tick 이하이면 outbound 긴급 → action=2

    def _run_mip_one(seed, capture=False):
        env = CrossDockEnv(config=CFG_8D, seed=seed)
        obs_list = env.reset()
        n = _n_agents(env)
        total_mip_t = 0.0; calls = 0
        frames = [mip_cap(env, [0]*n, [0.0]*n)] if capture else None
        done = False
        while not done:
            idle_doors = [d for d in env.doors if not d.is_busy]
            waiting    = env.waiting_trucks
            if idle_doors and waiting:
                t0 = time.perf_counter()
                assigned = solve_assignment(waiting, env.doors, env.outbound_trucks,
                                            max(env.buffer_capacity - env.buffer, 0.0))
                total_mip_t += time.perf_counter() - t0; calls += 1
                assigned_set = set()
                front = []
                for idx in assigned:
                    if idx is not None and idx not in assigned_set:
                        front.append(waiting[idx]); assigned_set.add(idx)
                rest = [t for i, t in enumerate(waiting) if i not in assigned_set]
                env.waiting_trucks = front + rest

            # Lane-mode actions: action=2 when outbound is urgent, else 1 or 0
            obs_list = env.get_obs()
            actions = []
            for k in range(n):
                o = obs_list[k]
                timer      = float(o[3])
                lane_queue = float(o[0])
                idle       = float(o[5])
                wait       = float(o[6])
                if lane_queue > 0 and timer < MILP_URGENT_TIMER:
                    actions.append(2)
                elif wait > 0 and idle > 0:
                    actions.append(1)
                else:
                    actions.append(0)

            obs_list, rewards, done, _ = env.step(actions)
            if capture:
                frames.append(mip_cap(env, actions, rewards))
        m = env.metrics.copy()
        m["total_ticks"]               = float(env.t)
        m["door_utilization"]          = env.door_utilization
        m["outbound_door_utilization"] = env.outbound_door_utilization
        m["avg_dwell_time"]            = env.avg_dwell_time
        m["avg_fill_rate"]             = env.avg_fill_rate
        m["avg_mip_ms"]                = total_mip_t / max(calls, 1) * 1000
        return m, env, frames

    mip_bench = [_run_mip_one(ep)[0] for ep in range(N_BENCH)]
    bench_results["mip"] = aggregate(mip_bench)
    tk = bench_results["mip"]["total_ticks"]
    print(f"  {'mip':12s}  ticks={tk['mean']:>7.1f}±{tk['std']:<5.1f}")

    m_viz, env_viz, frames_viz = _run_mip_one(SEED_EVAL, capture=True)
    save_viz_json(frames_viz, m_viz, env_viz, "mip", SEED_EVAL,
                  os.path.join(RUN_DIR, "sim_2stage_mip.json"))
except ImportError as e:
    print(f"  MILP 건너뜀 (pulp 미설치): {e}")

# ─────────────────────────────────────────────────────────────────
# 3. RL 학습 (Truck-Selection, 2000 에피소드)
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[3] RL 학습 (Lane-mode 3-action, 2000 에피소드)")
print("=" * 62)
from rl.train_rl import train as rl_train
from rl.rl_policy import QLearningPolicy

rl_result = rl_train(
    num_episodes=2000,
    lr=5e-4,
    seed=42,
    save_dir="checkpoints_2stage_8door",
    env_config=CFG_8D,
)
rl_net = rl_result["net"]

def rl_factory(env, net=rl_net):
    p = QLearningPolicy(net=net, epsilon=0.0)
    return [p] * _n_agents(env)

rl_bench = [run_episode_metrics(rl_factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
bench_results["rl"] = aggregate(rl_bench)
tk = bench_results["rl"]["total_ticks"]
print(f"  {'rl':12s}  ticks={tk['mean']:>7.1f}±{tk['std']:<5.1f}")

frames_rl, metrics_rl, env_rl = run_episode_frames(rl_factory, CFG_8D, seed=SEED_EVAL)
save_viz_json(frames_rl, metrics_rl, env_rl, "rl", SEED_EVAL,
              os.path.join(RUN_DIR, "sim_2stage_rl.json"))

# ─────────────────────────────────────────────────────────────────
# 4. GA 학습 (Truck-Selection, pop=50 gen=100)
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("[4] GA 학습 (Lane-mode 3-action, pop=50 gen=100)")
print("=" * 62)
from ga.train_ga import run_ga
from ga.ga_policy import GAPolicy, GENE_NAMES

GA_CFG = {**DEFAULT_CONFIG, **CFG_8D}

best_genes, best_fitness, ga_history = run_ga(
    pop_size=50, n_gen=100, n_eval=8, seed=0,
    cfg=GA_CFG, verbose=True,
)

def ga_factory(env, genes=best_genes):
    return [GAPolicy(genes) for _ in range(_n_agents(env))]

ga_bench = [run_episode_metrics(ga_factory, CFG_8D, seed=ep) for ep in range(N_BENCH)]
bench_results["ga"] = aggregate(ga_bench)
tk = bench_results["ga"]["total_ticks"]
print(f"  {'ga':12s}  ticks={tk['mean']:>7.1f}±{tk['std']:<5.1f}")

frames_ga, metrics_ga, env_ga = run_episode_frames(ga_factory, CFG_8D, seed=SEED_EVAL)
save_viz_json(frames_ga, metrics_ga, env_ga, "ga", SEED_EVAL,
              os.path.join(RUN_DIR, "sim_2stage_ga.json"))

ga_genes_path = os.path.join(ROOT, "ga", "best_genes_2stage.json")
with open(ga_genes_path, "w") as f:
    json.dump({"genes": best_genes.tolist(), "gene_names": GENE_NAMES,
               "fitness": float(best_fitness), "history": ga_history,
               "config": "lane_mode_3action"}, f, indent=2)
print(f"  GA 유전자 저장: {ga_genes_path}")

# ─────────────────────────────────────────────────────────────────
# 5. 최종 벤치마크 요약
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("[결과] Lane-mode 3-action 정책 비교 (20 에피소드)  ★ 목표: total_ticks 최소화")
print("=" * 72)
cols   = ["total_ticks", "total_throughput", "avg_fill_rate",
          "empty_departures", "door_utilization", "outbound_door_utilization"]
labels = ["Ticks(↓best)", "처리량(CBM)", "탑재율", "빈출발", "In-DoorUtil", "Out-DoorUtil"]

header = f"{'정책':12s}" + "".join(f"{lbl:>14s}" for lbl in labels)
print(header)
print("-" * len(header))
for name, agg in bench_results.items():
    row = f"{name:12s}"
    for col in cols:
        if col not in agg:
            row += f"{'N/A':>14s}"; continue
        mv = agg[col]
        if col in ("avg_fill_rate", "door_utilization", "outbound_door_utilization"):
            row += f"  {mv['mean']*100:>6.1f}%±{mv['std']*100:<4.1f}"
        else:
            row += f"  {mv['mean']:>8.1f}±{mv['std']:<4.1f}"
    print(row)

summary_path = os.path.join(RUN_DIR, "benchmark_2stage_8door.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        name: {k: {"mean": float(v["mean"]), "std": float(v["std"])}
               for k, v in agg.items()}
        for name, agg in bench_results.items()
    }, f, indent=2, ensure_ascii=False)
print(f"\n[저장] 벤치마크 요약: {summary_path}")
print(f"[저장] viz JSON: {RUN_DIR}/sim_2stage_*.json")
print(f"[run_dir] {run_dirname}")
