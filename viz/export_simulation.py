#!/usr/bin/env python3
"""
Cross-Docking 시뮬레이션 에피소드를 JSON으로 익스포트.
Three.js 3D 뷰어(index.html)에서 로드하여 시각화합니다.

Usage:
    python viz/export_simulation.py
    python viz/export_simulation.py --policy heuristic --seed 7 --output viz/sim.json
    python viz/export_simulation.py --policy rl   # 저장된 가중치로 RL 정책 실행
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.crossdock_env import CrossDockEnv
from env.policies import GreedyPolicy, RandomPolicy, FIFOPolicy, HeuristicPriorityPolicy

POLICY_MAP = {
    "greedy":    GreedyPolicy,
    "random":    RandomPolicy,
    "fifo":      FIFOPolicy,
    "heuristic": HeuristicPriorityPolicy,
}


# ──────────────────────────────────────────────
#  상태 캡처
# ──────────────────────────────────────────────

def capture_frame(env: CrossDockEnv, actions: list[int], rewards: list[float]) -> dict:
    """현재 환경 상태를 JSON-직렬화 가능한 dict로 반환."""
    return {
        "t": env.t,
        "buffer": float(env.buffer),
        "buffer_capacity": env.buffer_capacity,
        "waiting_trucks": [
            {
                "arrival_time": int(t.arrival_time),
                "shipments": {str(k): int(v) for k, v in t.shipments.items()},
                "total_volume": int(t.total_volume()),
            }
            for t in env.waiting_trucks
        ],
        "scheduled_trucks": [
            {
                "arrival_time": int(t.arrival_time),
                "shipments": {str(k): int(v) for k, v in t.shipments.items()},
                "total_volume": int(t.total_volume()),
            }
            for t in env.arrival_schedule
        ],
        "doors": [
            {
                "door_id": d.door_id,
                "is_busy": bool(d.is_busy),
                "remaining_time": int(d.remaining_time),
                "assigned_lane": int(d.assigned_lane),
                "assigned_truck_volume": (
                    int(d.assigned_truck.total_volume()) if d.assigned_truck else 0
                ),
                "assigned_truck_shipments": (
                    {str(k): int(v) for k, v in d.assigned_truck.shipments.items()}
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
        "metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in env.metrics.items()
        },
    }


# ──────────────────────────────────────────────
#  에피소드 실행
# ──────────────────────────────────────────────

def run_episode_baseline(policy_cls, seed: int = 42) -> tuple[list[dict], dict]:
    env = CrossDockEnv(seed=seed)
    obs = env.reset()
    policies = [policy_cls() for _ in range(env.num_lanes)]

    initial_actions = [0] * env.num_lanes
    initial_rewards = [0.0] * env.num_lanes
    frames = [capture_frame(env, initial_actions, initial_rewards)]

    done = False
    while not done:
        actions = [
            p.act(obs[k], env.num_inbound_doors)
            for k, p in enumerate(policies)
        ]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    return frames, env.metrics


def run_episode_rl(checkpoint_path: str, seed: int = 42) -> tuple[list[dict], dict]:
    """저장된 DQN 가중치로 에피소드 실행."""
    import numpy as np
    from rl.networks import NumpyMLP
    from rl.rl_policy import QLearningPolicy

    env = CrossDockEnv(seed=seed)
    obs = env.reset()

    # 체크포인트 가중치에서 obs_size 자동 감지 (학습 당시 설정과 다를 수 있음)
    w_data = np.load(checkpoint_path)
    ckpt_obs_size = w_data['W1'].shape[0]
    net = NumpyMLP(obs_size=ckpt_obs_size, n_actions=env.num_inbound_doors + 1)
    net.load(checkpoint_path)
    policy = QLearningPolicy(net=net, epsilon=0.0)  # 평가 모드: 탐험 없음

    initial_actions = [0] * env.num_lanes
    initial_rewards = [0.0] * env.num_lanes
    frames = [capture_frame(env, initial_actions, initial_rewards)]

    done = False
    while not done:
        actions = [
            policy.act(obs[k][:ckpt_obs_size], env.num_inbound_doors)
            for k in range(env.num_lanes)
        ]
        obs, rewards, done, _ = env.step(actions)
        frames.append(capture_frame(env, actions, rewards))

    return frames, env.metrics


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="시뮬레이션 에피소드를 JSON으로 익스포트합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        default="greedy",
        choices=[*POLICY_MAP.keys(), "rl"],
        help="실행할 정책",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/weights_final.npz",
        help="RL 정책 체크포인트 경로 (--policy rl 전용)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="출력 JSON 경로 (기본값: viz/simulation_data.json)",
    )
    args = parser.parse_args()

    # 출력 경로 결정
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "simulation_data.json")

    print(f"[export] policy={args.policy}  seed={args.seed}")

    if args.policy == "rl":
        checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.checkpoint,
        )
        if not os.path.exists(checkpoint):
            print(f"[error] 체크포인트를 찾을 수 없습니다: {checkpoint}")
            sys.exit(1)
        frames, metrics = run_episode_rl(checkpoint, seed=args.seed)
    else:
        frames, metrics = run_episode_baseline(POLICY_MAP[args.policy], seed=args.seed)

    data = {
        "meta": {
            "policy": args.policy,
            "seed": args.seed,
            "num_steps": len(frames),
            "num_lanes": 5,
            "num_doors": 3,
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

    avg_fill = (
        metrics["total_fill_rate"] / metrics["outbound_departures"]
        if metrics["outbound_departures"] > 0 else 0.0
    )
    print(f"[export] {len(frames)}개 프레임 저장 → {args.output}")
    print(
        f"[export] 최종 지표: "
        f"처리량={metrics['total_throughput']:.0f}  "
        f"평균탑재율={avg_fill:.1%}  "
        f"오버플로우={metrics['buffer_overflow_count']}"
    )
    print(f"\n[뷰어] viz/index.html 을 브라우저에서 열고 JSON 파일을 불러오세요.")


if __name__ == "__main__":
    main()
