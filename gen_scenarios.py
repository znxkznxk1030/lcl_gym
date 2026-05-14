#!/usr/bin/env python3
"""
모든 시나리오 JSON 재생성 스크립트.

Usage:
    python gen_scenarios.py          # 전체 재생성
    python gen_scenarios.py --mip    # MIP만
    python gen_scenarios.py --rl     # RL만
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
import numpy as np

VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz")

CONFIG_8DOOR = {
    "num_inbound_doors": 8,
    "arrival_count_min": 133,
    "arrival_count_max": 187,
}

SCENARIOS = {
    "baseline":     {},
    "door_failure": {"enable_disruptions": True, "disruption_door_failure": True},
    "rush_truck":   {"enable_disruptions": True, "disruption_rush_truck":   True},
    "timer_shock":  {"enable_disruptions": True, "disruption_timer_shock":  True},
    "all":          {"enable_disruptions": True, "disruption_door_failure": True,
                     "disruption_rush_truck": True, "disruption_timer_shock": True},
}


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
            for t in env.waiting_trucks
        ],
        "scheduled_trucks": [
            {"arrival_time": int(t.arrival_time),
             "shipments": {str(k): float(v) for k, v in t.shipments.items()},
             "total_volume": float(t.total_volume())}
            for t in env.arrival_schedule
        ],
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
            for d in env.doors
        ],
        "lanes": [
            {"lane_id": int(lane.lane_id),
             "queue_volume": float(lane.queue_volume),
             "congestion": float(lane.congestion),
             "outbound_loaded": float(env.outbound_trucks[k].loaded),
             "outbound_fill_rate": float(env.outbound_trucks[k].fill_rate),
             "outbound_departure_timer": int(env.outbound_trucks[k].departure_timer),
             "outbound_capacity": float(env.outbound_trucks[k].capacity)}
            for k, lane in enumerate(env.lanes)
        ],
        "actions": list(actions),
        "rewards": [float(r) for r in rewards],
        "disruptions": list(env.disruption_log),
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in env.metrics.items()},
    }


# ── MIP ─────────────────────────────────────────────────────────────────────

def gen_mip_scenario(scenario_name, disruption_cfg, seed=42):
    from mip.solve_mip import solve_assignment

    config = {**DEFAULT_CONFIG, **CONFIG_8DOOR, **disruption_cfg}
    env = CrossDockEnv(seed=seed, config=config)
    obs = env.reset()

    frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
    done = False
    total_mip_time = 0.0
    mip_calls = 0

    while not done:
        idle_doors = [d for d in env.doors if not d.is_busy]
        waiting = env.waiting_trucks

        if idle_doors and waiting:
            t0 = time.perf_counter()
            assigned_indices = solve_assignment(
                waiting_trucks=waiting,
                doors=env.doors,
                outbound_trucks=env.outbound_trucks,
                buffer_remaining=max(env.buffer_capacity - env.buffer, 0.0),
            )
            total_mip_time += time.perf_counter() - t0
            mip_calls += 1

            reordered_front, assigned_set = [], set()
            for idx in assigned_indices:
                if idx is not None and idx not in assigned_set:
                    reordered_front.append(waiting[idx])
                    assigned_set.add(idx)
            env.waiting_trucks = reordered_front + [t for i, t in enumerate(waiting) if i not in assigned_set]

            requesting_lanes = set()
            for idx in assigned_indices:
                if idx is not None and idx < len(waiting) and waiting[idx].shipments:
                    urgent_lane = min(waiting[idx].shipments.keys(),
                                      key=lambda k: env.outbound_trucks[k].departure_timer)
                    requesting_lanes.add(urgent_lane)

            if len(requesting_lanes) < len(idle_doors):
                for k in sorted(range(env.num_lanes), key=lambda k: env.outbound_trucks[k].departure_timer):
                    if len(requesting_lanes) >= len(idle_doors):
                        break
                    requesting_lanes.add(k)

            actions = [1 if k in requesting_lanes else 0 for k in range(env.num_lanes)]
        else:
            actions = [0] * env.num_lanes

        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    avg_ms = total_mip_time / max(mip_calls, 1) * 1000
    data = {
        "meta": {
            "policy": "mip",
            "seed": seed,
            "num_steps": len(frames),
            "num_doors": config["num_inbound_doors"],
            "num_lanes": config["num_lanes"],
            "dispatch_interval_max": config["dispatch_interval_max"],
            "scenario": scenario_name,
            "solver": "PULP CBC (MILP)",
            "solver_stats": {
                "mip_calls": mip_calls,
                "total_mip_time_sec": round(total_mip_time, 4),
                "avg_mip_time_ms": round(avg_ms, 2),
            },
            "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in env.metrics.items()},
        },
        "frames": frames,
    }
    out = os.path.join(VIZ_DIR, f"sim_mip_8door_{scenario_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    m = env.metrics
    avg_fill = m["total_fill_rate"] / m["outbound_departures"] if m["outbound_departures"] > 0 else 0
    print(f"  MIP 8door/{scenario_name}: throughput={m['total_throughput']:.1f}  "
          f"fill={avg_fill:.1%}  overflow={m['buffer_overflow_count']}  → {os.path.basename(out)}")


# ── RL ──────────────────────────────────────────────────────────────────────

def gen_rl_scenario(scenario_name, disruption_cfg, seed=42,
                    checkpoint="checkpoints_8door_disruption/weights_final.npz"):
    from rl.networks import NumpyMLP
    from rl.rl_policy import QLearningPolicy, normalize_obs

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint)
    w_data = np.load(ckpt_path)
    ckpt_obs_size = w_data["W1"].shape[0]

    config = {**DEFAULT_CONFIG, **CONFIG_8DOOR, **disruption_cfg}
    env = CrossDockEnv(seed=seed, config=config)
    obs = env.reset()

    net = NumpyMLP(obs_size=ckpt_obs_size, n_actions=2)
    net.load(ckpt_path)
    policy = QLearningPolicy(net=net, epsilon=0.0)

    frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
    done = False

    while not done:
        actions = [policy.act(obs[k][:ckpt_obs_size], env.num_inbound_doors)
                   for k in range(env.num_lanes)]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    data = {
        "meta": {
            "policy": "rl",
            "seed": seed,
            "num_steps": len(frames),
            "num_doors": config["num_inbound_doors"],
            "num_lanes": config["num_lanes"],
            "dispatch_interval_max": config["dispatch_interval_max"],
            "scenario": scenario_name,
            "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in env.metrics.items()},
        },
        "frames": frames,
    }
    out = os.path.join(VIZ_DIR, f"sim_rl_8door_disruption_{scenario_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    m = env.metrics
    avg_fill = m["total_fill_rate"] / m["outbound_departures"] if m["outbound_departures"] > 0 else 0
    print(f"  RL  8door/{scenario_name}: throughput={m['total_throughput']:.1f}  "
          f"fill={avg_fill:.1%}  overflow={m['buffer_overflow_count']}  → {os.path.basename(out)}")


# ── 3-door 기본 JSONs ────────────────────────────────────────────────────────

def gen_3door_baselines(seed=42):
    from viz.export_simulation import run_episode_baseline, run_episode_rl, capture_frame as cf
    from env.policies import GreedyPolicy, HeuristicPriorityPolicy, FIFOPolicy

    for policy_cls, name in [(GreedyPolicy, "greedy"), (HeuristicPriorityPolicy, "heuristic"), (FIFOPolicy, "fifo")]:
        frames, metrics = run_episode_baseline(policy_cls, seed=seed)
        data = {
            "meta": {
                "policy": name, "seed": seed, "num_steps": len(frames),
                "num_lanes": DEFAULT_CONFIG["num_lanes"],
                "num_doors": DEFAULT_CONFIG["num_inbound_doors"],
                "dispatch_interval_max": DEFAULT_CONFIG["dispatch_interval_max"],
                "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in metrics.items()},
            },
            "frames": frames,
        }
        out = os.path.join(VIZ_DIR, f"sim_{name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        avg_fill = metrics["total_fill_rate"] / metrics["outbound_departures"] if metrics["outbound_departures"] > 0 else 0
        print(f"  {name}: throughput={metrics['total_throughput']:.1f}  fill={avg_fill:.1%}  → {os.path.basename(out)}")


def gen_3door_mip(seed=42):
    from mip.solve_mip import run_episode_mip
    frames, metrics, solver_stats = run_episode_mip(seed=seed, verbose=False)
    data = {
        "meta": {
            "policy": "mip", "seed": seed, "num_steps": len(frames),
            "num_lanes": DEFAULT_CONFIG["num_lanes"],
            "num_doors": DEFAULT_CONFIG["num_inbound_doors"],
            "dispatch_interval_max": DEFAULT_CONFIG["dispatch_interval_max"],
            "solver": "PULP CBC (MILP)", "solver_stats": solver_stats,
            "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in metrics.items()},
        },
        "frames": frames,
    }
    out = os.path.join(VIZ_DIR, "sim_mip.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    avg_fill = metrics["total_fill_rate"] / metrics["outbound_departures"] if metrics["outbound_departures"] > 0 else 0
    print(f"  mip: throughput={metrics['total_throughput']:.1f}  fill={avg_fill:.1%}  → sim_mip.json")


def gen_3door_rl(seed=42, checkpoint="checkpoints/weights_final.npz"):
    from viz.export_simulation import run_episode_rl
    frames, metrics = run_episode_rl(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint), seed=seed
    )
    data = {
        "meta": {
            "policy": "rl", "seed": seed, "num_steps": len(frames),
            "num_lanes": DEFAULT_CONFIG["num_lanes"],
            "num_doors": DEFAULT_CONFIG["num_inbound_doors"],
            "dispatch_interval_max": DEFAULT_CONFIG["dispatch_interval_max"],
            "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in metrics.items()},
        },
        "frames": frames,
    }
    out = os.path.join(VIZ_DIR, "sim_rl.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    avg_fill = metrics["total_fill_rate"] / metrics["outbound_departures"] if metrics["outbound_departures"] > 0 else 0
    print(f"  rl: throughput={metrics['total_throughput']:.1f}  fill={avg_fill:.1%}  → sim_rl.json")


def gen_3door_mip_disruptions(seed=42):
    from mip.solve_mip import run_episode_mip
    name_map = {
        "baseline":     {},
        "door_failure": {"enable_disruptions": True, "disruption_door_failure": True},
        "rush_truck":   {"enable_disruptions": True, "disruption_rush_truck":   True},
        "timer_shock":  {"enable_disruptions": True, "disruption_timer_shock":  True},
        "all_disruptions": {"enable_disruptions": True, "disruption_door_failure": True,
                            "disruption_rush_truck": True, "disruption_timer_shock": True},
    }
    for scenario_name, disruption_cfg in name_map.items():
        frames, metrics, solver_stats = run_episode_mip(seed=seed, verbose=False,
                                                        disruption_config=disruption_cfg)
        data = {
            "meta": {
                "policy": "mip", "seed": seed, "num_steps": len(frames),
                "num_lanes": DEFAULT_CONFIG["num_lanes"],
                "num_doors": DEFAULT_CONFIG["num_inbound_doors"],
                "dispatch_interval_max": DEFAULT_CONFIG["dispatch_interval_max"],
                "scenario": scenario_name,
                "solver": "PULP CBC (MILP)", "solver_stats": solver_stats,
                "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in metrics.items()},
            },
            "frames": frames,
        }
        out = os.path.join(VIZ_DIR, f"sim_mip_{scenario_name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        avg_fill = metrics["total_fill_rate"] / metrics["outbound_departures"] if metrics["outbound_departures"] > 0 else 0
        print(f"  MIP 3door/{scenario_name}: throughput={metrics['total_throughput']:.1f}  "
              f"fill={avg_fill:.1%}  → {os.path.basename(out)}")


def gen_8door_scaled_mip(seed=42):
    from mip.solve_mip import solve_assignment
    name_map = {
        "baseline":     {},
        "door_failure": {"enable_disruptions": True, "disruption_door_failure": True},
        "rush_truck":   {"enable_disruptions": True, "disruption_rush_truck":   True},
        "timer_shock":  {"enable_disruptions": True, "disruption_timer_shock":  True},
    }
    for scenario_name, disruption_cfg in name_map.items():
        config = {**DEFAULT_CONFIG, **CONFIG_8DOOR, **disruption_cfg}
        env = CrossDockEnv(seed=seed, config=config)
        obs = env.reset()
        frames = [capture_frame(env, [0]*env.num_lanes, [0.0]*env.num_lanes)]
        done = False
        total_mip_time = 0.0
        mip_calls = 0

        while not done:
            idle_doors = [d for d in env.doors if not d.is_busy]
            waiting = env.waiting_trucks
            if idle_doors and waiting:
                t0 = time.perf_counter()
                assigned_indices = solve_assignment(
                    waiting_trucks=waiting, doors=env.doors,
                    outbound_trucks=env.outbound_trucks,
                    buffer_remaining=max(env.buffer_capacity - env.buffer, 0.0),
                )
                total_mip_time += time.perf_counter() - t0
                mip_calls += 1
                reordered_front, assigned_set = [], set()
                for idx in assigned_indices:
                    if idx is not None and idx not in assigned_set:
                        reordered_front.append(waiting[idx])
                        assigned_set.add(idx)
                env.waiting_trucks = reordered_front + [t for i, t in enumerate(waiting) if i not in assigned_set]
                requesting_lanes = set()
                for idx in assigned_indices:
                    if idx is not None and idx < len(waiting) and waiting[idx].shipments:
                        urgent_lane = min(waiting[idx].shipments.keys(),
                                          key=lambda k: env.outbound_trucks[k].departure_timer)
                        requesting_lanes.add(urgent_lane)
                if len(requesting_lanes) < len(idle_doors):
                    for k in sorted(range(env.num_lanes), key=lambda k: env.outbound_trucks[k].departure_timer):
                        if len(requesting_lanes) >= len(idle_doors):
                            break
                        requesting_lanes.add(k)
                actions = [1 if k in requesting_lanes else 0 for k in range(env.num_lanes)]
            else:
                actions = [0] * env.num_lanes
            obs, rewards, done, _ = env.step(actions)
            frames.append(capture_frame(env, actions, rewards))

        data = {
            "meta": {
                "policy": f"mip_8door_scaled_{scenario_name}",
                "seed": seed, "num_steps": len(frames),
                "num_lanes": config["num_lanes"],
                "num_doors": config["num_inbound_doors"],
                "note": "truck arrivals scaled 8/3x",
                "scenario": scenario_name,
                "solver": "PULP CBC (MILP)",
                "solver_stats": {
                    "mip_calls": mip_calls,
                    "total_mip_time_sec": round(total_mip_time, 4),
                    "avg_mip_time_ms": round(total_mip_time / max(mip_calls, 1) * 1000, 2),
                },
                "final_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in env.metrics.items()},
            },
            "frames": frames,
        }
        out = os.path.join(VIZ_DIR, f"sim_mip_8door_scaled_{scenario_name}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        avg_fill = env.metrics["total_fill_rate"] / env.metrics["outbound_departures"] if env.metrics["outbound_departures"] > 0 else 0
        print(f"  MIP 8door_scaled/{scenario_name}: throughput={env.metrics['total_throughput']:.1f}  "
              f"fill={avg_fill:.1%}  → {os.path.basename(out)}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    do_mip = "--mip" in args or not args
    do_rl  = "--rl"  in args or not args

    print("=== 3-door 기본 JSON 재생성 ===")
    gen_3door_baselines()
    if do_mip:
        gen_3door_mip()
    if do_rl:
        gen_3door_rl()

    print("\n=== 8-door MIP 시나리오 재생성 ===")
    if do_mip:
        for name, cfg in SCENARIOS.items():
            gen_mip_scenario(name, cfg)

    print("\n=== 8-door RL 시나리오 재생성 ===")
    if do_rl:
        for name, cfg in SCENARIOS.items():
            gen_rl_scenario(name, cfg)

    print("\n=== 3-door MIP 돌발 시나리오 재생성 ===")
    if do_mip:
        gen_3door_mip_disruptions()

    print("\n=== 8-door scaled MIP 시나리오 재생성 (legacy) ===")
    if do_mip:
        gen_8door_scaled_mip()

    print("\n완료.")
