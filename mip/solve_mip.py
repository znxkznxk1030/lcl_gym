#!/usr/bin/env python3
"""
MILP 기반 Cross-Docking 최적화 솔버.

[문제 정의]
인바운드 트럭(~60대)이 스케줄에 따라 도착. 각 트럭은 2~3개 목적지 레인의 화물 혼재.
유휴 도어에 트럭을 배정(처리 시간 1~10 스텝)하면 화물이 레인 큐로 이동.
아웃바운드 트럭은 레인별 정해진 주기로 출발 → fill_rate 최대화가 목표.

[MILP 역할]
매 스텝마다 "대기 트럭 × 유휴 도어" 배정 문제를 MILP로 풀어,
urgency(잔여 출발 시간)·화물량을 가중한 최적 배정 순서를 결정한다.

목적함수: maximize Σ_{j,i} x_{j,i} · score_j
  score_j = Σ_k  v_{j,k} / (departure_timer_k + 1)

제약조건:
  (1) 트럭은 최대 1개 도어에 배정
  (2) 도어는 최대 1개 트럭 처리
  (3) 버퍼 여유량 초과 배정 금지
  (4) x_{j,i} ∈ {0,1}

Usage:
    python mip/solve_mip.py
    python mip/solve_mip.py --seed 42 --output viz/sim_mip.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.entities import Truck, Door, OutboundTruck


# ──────────────────────────────────────────────────────────────
#  Per-step MILP 솔버
# ──────────────────────────────────────────────────────────────

def solve_assignment(
    waiting_trucks: List[Truck],
    doors: List[Door],
    outbound_trucks: List[OutboundTruck],
    buffer_remaining: float,
    max_trucks: int = 15,
) -> List[int]:
    """
    현재 스텝에서 대기 트럭을 유휴 도어에 최적 배정.

    Returns:
        assigned_truck_indices: 각 유휴 도어에 배정할 waiting_trucks 인덱스 리스트.
                                비어있으면 배정 없음.
    """
    # 유휴 + 정상(비고장) 도어만 대상
    idle_door_indices = [i for i, d in enumerate(doors) if not d.is_busy and not d.is_failed]
    if not idle_door_indices or not waiting_trucks:
        return []

    # 고려할 트럭 수 제한 (MIP 규모 관리)
    trucks = waiting_trucks[:max_trucks]
    n_t = len(trucks)
    n_d = len(idle_door_indices)

    # urgency 가중 점수 계산 (긴급 트럭은 3배 가중)
    RUSH_MULTIPLIER = 3.0
    scores = []
    for truck in trucks:
        s = sum(
            vol / (outbound_trucks[lane_id].departure_timer + 1)
            for lane_id, vol in truck.shipments.items()
        )
        if getattr(truck, "is_rush", False):
            s *= RUSH_MULTIPLIER
        scores.append(s)

    # ── MILP 정의 ──
    prob = pulp.LpProblem("DoorAssignment", pulp.LpMaximize)

    # x[j][d] = 트럭 j를 유휴도어 d에 배정
    x = [[pulp.LpVariable(f"x_{j}_{d}", cat="Binary") for d in range(n_d)] for j in range(n_t)]

    # 목적함수
    prob += pulp.lpSum(x[j][d] * scores[j] for j in range(n_t) for d in range(n_d))

    # (1) 트럭 하나는 최대 하나의 도어
    for j in range(n_t):
        prob += pulp.lpSum(x[j][d] for d in range(n_d)) <= 1

    # (2) 도어 하나는 최대 하나의 트럭
    for d in range(n_d):
        prob += pulp.lpSum(x[j][d] for j in range(n_t)) <= 1

    # (3) 버퍼 여유량 제약: 배정된 트럭들의 총 화물량 ≤ buffer_remaining
    if buffer_remaining < sum(t.total_volume() for t in trucks):
        prob += pulp.lpSum(
            x[j][d] * trucks[j].total_volume()
            for j in range(n_t)
            for d in range(n_d)
        ) <= buffer_remaining

    # 풀기 (로그 억제)
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=1.0)
    prob.solve(solver)

    # 결과 추출: 도어별 배정 트럭 인덱스
    assigned = []
    for d in range(n_d):
        for j in range(n_t):
            if pulp.value(x[j][d]) is not None and pulp.value(x[j][d]) > 0.5:
                assigned.append(j)
                break
        else:
            assigned.append(None)

    return assigned  # 길이 = n_d (유휴 도어 수)


# ──────────────────────────────────────────────────────────────
#  MIP 정책 실행
# ──────────────────────────────────────────────────────────────

def capture_frame(env: CrossDockEnv, actions: list, rewards: list) -> dict:
    return {
        "t": env.t,
        "buffer": float(env.buffer),
        "buffer_capacity": env.buffer_capacity,
        "waiting_trucks": [
            {
                "arrival_time": int(t.arrival_time),
                "shipments": {str(k): float(v) for k, v in t.shipments.items()},
                "total_volume": float(t.total_volume()),
                "is_rush": bool(getattr(t, "is_rush", False)),
            }
            for t in env.waiting_trucks
        ],
        "scheduled_trucks": [
            {
                "arrival_time": int(t.arrival_time),
                "shipments": {str(k): float(v) for k, v in t.shipments.items()},
                "total_volume": float(t.total_volume()),
            }
            for t in env.arrival_schedule
        ],
        "doors": [
            {
                "door_id": d.door_id,
                "is_busy": bool(d.is_busy),
                "is_failed": bool(d.is_failed),
                "failure_remaining": int(d.failure_remaining),
                "remaining_time": int(d.remaining_time),
                "assigned_lane": int(d.assigned_lane),
                "assigned_truck_volume": (
                    float(d.assigned_truck.total_volume()) if d.assigned_truck else 0
                ),
                "assigned_truck_shipments": (
                    {str(k): float(v) for k, v in d.assigned_truck.shipments.items()}
                    if d.assigned_truck else {}
                ),
            }
            for d in env.doors
        ],
        "lanes": [
            {
                "lane_id": int(lane.lane_id),
                "queue_volume": float(lane.queue_volume),
                "congestion": float(lane.congestion),
                "outbound_loaded": float(env.outbound_trucks[k].loaded),
                "outbound_fill_rate": float(env.outbound_trucks[k].fill_rate),
                "outbound_departure_timer": int(env.outbound_trucks[k].departure_timer),
                "outbound_capacity": float(env.outbound_trucks[k].capacity),
            }
            for k, lane in enumerate(env.lanes)
        ],
        "actions": list(actions),
        "rewards": [float(r) for r in rewards],
        "disruptions": list(env.disruption_log),
        "metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in env.metrics.items()
        },
    }


def run_episode_mip(
    seed: int = 42,
    verbose: bool = True,
    disruption_config: dict = None,
) -> tuple[list[dict], dict, dict]:
    """
    MILP 정책으로 에피소드 실행.

    전략:
      1. 매 스텝 MILP로 최적 트럭-도어 배정 순서 결정
      2. MILP 결과로 waiting_trucks 순서 재정렬 (env는 항상 큐 앞에서 FIFO 배정)
      3. 배정된 트럭이 있는 레인은 action=1 (요청)
    """
    env = CrossDockEnv(seed=seed, config=disruption_config or {})
    obs = env.reset()

    initial_actions = [0] * env.num_lanes
    initial_rewards = [0.0] * env.num_lanes
    frames = [capture_frame(env, initial_actions, initial_rewards)]

    total_mip_time = 0.0
    mip_call_count = 0
    done = False

    while not done:
        idle_doors = [d for d in env.doors if not d.is_busy]
        waiting = env.waiting_trucks

        # ── MILP 배정 결정 ──
        if idle_doors and waiting:
            t0 = time.perf_counter()
            assigned_indices = solve_assignment(
                waiting_trucks=waiting,
                doors=env.doors,
                outbound_trucks=env.outbound_trucks,
                buffer_remaining=max(env.buffer_capacity - env.buffer, 0.0),
            )
            total_mip_time += time.perf_counter() - t0
            mip_call_count += 1

            # MILP 결과로 큐 앞부분 재정렬:
            # assigned_indices[i] = 유휴도어 i에 배정할 트럭의 waiting 인덱스
            reordered_front = []
            assigned_set = set()
            for idx in assigned_indices:
                if idx is not None and idx not in assigned_set:
                    reordered_front.append(waiting[idx])
                    assigned_set.add(idx)

            # 나머지 트럭은 순서 유지
            rest = [t for i, t in enumerate(waiting) if i not in assigned_set]
            env.waiting_trucks = reordered_front + rest

            # 배정된 트럭의 레인들이 모두 요청
            requesting_lanes = set()
            for idx in assigned_indices:
                if idx is not None and idx < len(waiting):
                    truck = waiting[idx]
                    # 가장 긴급한 레인 하나만 요청 (도어 배정 트리거용)
                    if truck.shipments:
                        urgent_lane = min(
                            truck.shipments.keys(),
                            key=lambda k: env.outbound_trucks[k].departure_timer,
                        )
                        requesting_lanes.add(urgent_lane)

            # 유휴 도어 수만큼 레인이 요청해야 도어가 활성화됨
            # 부족하면 urgency 순으로 채움
            if len(requesting_lanes) < len(idle_doors):
                all_lanes_by_urgency = sorted(
                    range(env.num_lanes),
                    key=lambda k: env.outbound_trucks[k].departure_timer,
                )
                for k in all_lanes_by_urgency:
                    if len(requesting_lanes) >= len(idle_doors):
                        break
                    requesting_lanes.add(k)

            actions = [1 if k in requesting_lanes else 0 for k in range(env.num_lanes)]
        else:
            actions = [0] * env.num_lanes

        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    if verbose:
        print(f"[MIP] MILP 호출 횟수: {mip_call_count}회, 총 풀이 시간: {total_mip_time:.2f}s")
        print(f"[MIP] 평균 풀이 시간: {total_mip_time/max(mip_call_count,1)*1000:.1f}ms/스텝")

    solver_stats = {
        "mip_calls": mip_call_count,
        "total_mip_time_sec": round(total_mip_time, 4),
        "avg_mip_time_ms": round(total_mip_time / max(mip_call_count, 1) * 1000, 2),
        "disruption_door_failures": env.metrics.get("disruption_door_failures", 0),
        "disruption_interrupted_trucks": env.metrics.get("disruption_interrupted_trucks", 0),
        "disruption_rush_trucks": env.metrics.get("disruption_rush_trucks", 0),
        "disruption_timer_shocks": env.metrics.get("disruption_timer_shocks", 0),
    }
    return frames, env.metrics, solver_stats


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MILP 기반 Cross-Docking 시뮬레이션을 실행하고 JSON으로 저장합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="출력 JSON 경로 (기본값: viz/sim_mip.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(root, "viz", "sim_mip.json")

    print(f"[MIP] seed={args.seed} | solver=PULP/CBC")
    print("[MIP] 에피소드 실행 중...")

    frames, metrics, solver_stats = run_episode_mip(seed=args.seed, verbose=True)

    avg_fill = (
        metrics["total_fill_rate"] / metrics["outbound_departures"]
        if metrics["outbound_departures"] > 0 else 0.0
    )

    data = {
        "meta": {
            "policy": "mip",
            "seed": args.seed,
            "num_steps": len(frames),
            "num_lanes": DEFAULT_CONFIG["num_lanes"],
            "num_doors": DEFAULT_CONFIG["num_inbound_doors"],
            "dispatch_interval_max": DEFAULT_CONFIG["dispatch_interval_max"],
            "solver": "PULP CBC (MILP)",
            "solver_stats": solver_stats,
            "final_metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            },
        },
        "frames": frames,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[결과] {len(frames)}개 프레임 → {args.output}")
    print(
        f"[결과] 처리량={metrics['total_throughput']:.1f} CBM  "
        f"평균탑재율={avg_fill:.1%}  "
        f"오버플로우={metrics['buffer_overflow_count']}  "
        f"빈출발={metrics['empty_departures']}"
    )
    print(f"\n[뷰어] viz/index.html 을 열고 sim_mip.json 을 불러오세요.")


if __name__ == "__main__":
    main()
