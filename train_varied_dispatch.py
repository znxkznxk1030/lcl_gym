#!/usr/bin/env python3
"""
레인별 다른 출발 주기 시나리오 RL 학습 + MIP 비교 JSON 생성.

Lane 0: 8스텝  (매우 빠름)
Lane 1: 14스텝 (빠름)
Lane 2: 22스텝 (중간)
Lane 3: 32스텝 (느림)
Lane 4: 45스텝 (매우 느림)

Usage:
    python train_varied_dispatch.py           # 학습 + JSON 생성
    python train_varied_dispatch.py --gen-only  # 기존 가중치로 JSON만 재생성
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from mip.solve_mip import solve_assignment
import numpy as np

INTERVALS  = [8, 14, 22, 32, 45]
SAVE_DIR   = "checkpoints_varied_dispatch"
VIZ_DIR    = "viz"
SEED       = 42

ENV_CONFIG = {
    "lane_dispatch_intervals": INTERVALS,
    "num_inbound_doors": 3,
}


# ── capture_frame (공통) ─────────────────────────────────────────────────────

def capture_frame(env, actions, rewards):
    return {
        "t": env.t,
        "buffer": float(env.buffer),
        "buffer_capacity": env.buffer_capacity,
        "waiting_trucks": [
            {"arrival_time": int(t.arrival_time),
             "shipments": {str(k): float(v) for k, v in t.shipments.items()},
             "total_volume": float(t.total_volume()),
             "is_rush": bool(getattr(t, "is_rush", False))}
            for t in env.waiting_trucks],
        "scheduled_trucks": [
            {"arrival_time": int(t.arrival_time),
             "shipments": {str(k): float(v) for k, v in t.shipments.items()},
             "total_volume": float(t.total_volume())}
            for t in env.arrival_schedule],
        "doors": [
            {"door_id": d.door_id,
             "is_busy": bool(d.is_busy),
             "is_failed": bool(d.is_failed),
             "failure_remaining": int(d.failure_remaining),
             "remaining_time": int(d.remaining_time),
             "assigned_lane": int(d.assigned_lane),
             "assigned_truck_volume": float(d.assigned_truck.total_volume()) if d.assigned_truck else 0,
             "assigned_truck_shipments": ({str(k): float(v) for k, v in d.assigned_truck.shipments.items()}
                                          if d.assigned_truck else {})}
            for d in env.doors],
        "lanes": [
            {"lane_id": int(lane.lane_id),
             "queue_volume": float(lane.queue_volume),
             "congestion": float(lane.congestion),
             "outbound_loaded": float(env.outbound_trucks[k].loaded),
             "outbound_fill_rate": float(env.outbound_trucks[k].fill_rate),
             "outbound_departure_timer": int(env.outbound_trucks[k].departure_timer),
             "outbound_capacity": float(env.outbound_trucks[k].capacity),
             "lane_dispatch_interval": INTERVALS[k]}
            for k, lane in enumerate(env.lanes)],
        "actions": list(actions),
        "rewards": [float(r) for r in rewards],
        "disruptions": list(env.disruption_log),
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in env.metrics.items()},
    }


def make_meta(policy, seed, frames, env):
    return {
        "policy": policy,
        "seed": seed,
        "num_steps": len(frames),
        "num_doors": env.num_inbound_doors,
        "num_lanes": env.num_lanes,
        "dispatch_interval_max": max(INTERVALS),
        "lane_dispatch_intervals": INTERVALS,
        "scenario": "varied_dispatch",
        "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in env.metrics.items()},
    }


# ── MIP 에피소드 ─────────────────────────────────────────────────────────────

def run_mip(seed=SEED):
    config = {**DEFAULT_CONFIG, **ENV_CONFIG}
    env = CrossDockEnv(seed=seed, config=config)
    obs = env.reset()
    frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
    done = False
    t0_total = 0.0
    calls = 0

    while not done:
        idle = [d for d in env.doors if not d.is_busy]
        waiting = env.waiting_trucks
        if idle and waiting:
            t0 = time.perf_counter()
            assigned = solve_assignment(
                waiting_trucks=waiting, doors=env.doors,
                outbound_trucks=env.outbound_trucks,
                buffer_remaining=max(env.buffer_capacity - env.buffer, 0.0),
            )
            t0_total += time.perf_counter() - t0
            calls += 1

            front, seen = [], set()
            for idx in assigned:
                if idx is not None and idx not in seen:
                    front.append(waiting[idx]); seen.add(idx)
            env.waiting_trucks = front + [t for i,t in enumerate(waiting) if i not in seen]

            req = set()
            for idx in assigned:
                if idx is not None and idx < len(waiting) and waiting[idx].shipments:
                    req.add(min(waiting[idx].shipments.keys(),
                                key=lambda k: env.outbound_trucks[k].departure_timer))
            if len(req) < len(idle):
                for k in sorted(range(env.num_lanes), key=lambda k: env.outbound_trucks[k].departure_timer):
                    if len(req) >= len(idle): break
                    req.add(k)
            actions = [1 if k in req else 0 for k in range(env.num_lanes)]
        else:
            actions = [0]*env.num_lanes

        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    return frames, env, {"mip_calls": calls, "total_sec": round(t0_total,4)}


# ── RL 에피소드 ──────────────────────────────────────────────────────────────

def run_rl(seed=SEED):
    from rl.networks import NumpyMLP
    from rl.rl_policy import QLearningPolicy, normalize_obs

    ckpt = os.path.join(SAVE_DIR, "weights_final.npz")
    w = np.load(ckpt)
    obs_sz = w["W1"].shape[0]

    config = {**DEFAULT_CONFIG, **ENV_CONFIG}
    env = CrossDockEnv(seed=seed, config=config)
    obs = env.reset()

    net = NumpyMLP(obs_size=obs_sz, n_actions=2)
    net.load(ckpt)
    policy = QLearningPolicy(net=net, epsilon=0.0)

    frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
    done = False
    while not done:
        actions = [policy.act(obs[k][:obs_sz], env.num_inbound_doors)
                   for k in range(env.num_lanes)]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    return frames, env


# ── 학습 ────────────────────────────────────────────────────────────────────

def train():
    from rl.train_rl import train as rl_train
    print(f"레인별 출발 주기: {dict(enumerate(INTERVALS))}")
    print(f"저장 디렉토리: {SAVE_DIR}\n")
    rl_train(
        num_episodes=2000,
        lr=5e-4,
        seed=SEED,
        save_dir=SAVE_DIR,
        env_config=ENV_CONFIG,
        log_interval=200,
    )


# ── JSON 저장 ────────────────────────────────────────────────────────────────

def save_json(data, fname):
    path = os.path.join(VIZ_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    m = data["meta"]["final_metrics"]
    avg_fill = m["total_fill_rate"] / m["outbound_departures"] if m["outbound_departures"] > 0 else 0
    print(f"  → {fname}  throughput={m['total_throughput']:.1f}  fill={avg_fill:.1%}  overflow={m['buffer_overflow_count']}")


def gen_jsons():
    print("\n=== MIP (varied dispatch) ===")
    frames, env, stats = run_mip()
    save_json({"meta": {**make_meta("mip", SEED, frames, env), "solver_stats": stats},
               "frames": frames}, "sim_mip_varied_dispatch.json")

    print("\n=== RL (varied dispatch) ===")
    frames, env = run_rl()
    save_json({"meta": make_meta("rl_varied", SEED, frames, env),
               "frames": frames}, "sim_rl_varied_dispatch.json")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen_only = "--gen-only" in sys.argv
    if not gen_only:
        train()
    gen_jsons()
    print("\n완료. viz/sim_mip_varied_dispatch.json, sim_rl_varied_dispatch.json")
