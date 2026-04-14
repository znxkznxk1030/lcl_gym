from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class LaneConfig:
    agent_id: str
    destination: str
    cutoff_interval: int
    cutoff_phase: int
    dispatch_capacity: float
    lane_buffer_capacity: float
    dispatch_cost: float
    base_arrival_rate: float
    demand_pattern: Sequence[float]
    deadline_mean: int
    deadline_jitter: int
    volume_range: tuple[float, float]
    urgency_bias: float = 1.0
    holding_cost: float = 0.15
    missed_deadline_penalty: float = 3.0


@dataclass(frozen=True)
class ArrivalConfig:
    flexible_ratio: float = 0.0
    flexible_lane_count_weights: tuple[float, ...] = (0.8, 0.2)
    volume_noise_scale: float = 0.25


@dataclass(frozen=True)
class ObservationConfig:
    top_k_shipments: int = 5
    include_other_lane_summary: bool = True
    include_action_mask: bool = True
    other_lane_feature_count: int = 4


@dataclass(frozen=True)
class RewardConfig:
    team_weight: float = 0.4
    local_weight: float = 0.6
    accept_reward: float = 0.3
    dispatch_volume_reward: float = 1.0
    utilization_reward: float = 1.0
    staging_pressure_penalty: float = 0.8
    overflow_penalty: float = 3.5
    invalid_action_penalty: float = 0.75
    missed_deadline_penalty: float = 2.5
    waiting_penalty: float = 0.05


@dataclass(frozen=True)
class ConflictConfig:
    claim_priority_weight: float = 1.0
    cutoff_priority_weight: float = 0.35
    backlog_priority_weight: float = 0.25


@dataclass(frozen=True)
class EnvironmentConfig:
    horizon: int = 48
    seed: int = 7
    staging_capacity: float = 30.0
    dispatch_slots_per_step: int = 2
    num_agents: int = 5
    lanes: tuple[LaneConfig, ...] = field(default_factory=tuple)
    arrivals: ArrivalConfig = field(default_factory=ArrivalConfig)
    observations: ObservationConfig = field(default_factory=ObservationConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    conflicts: ConflictConfig = field(default_factory=ConflictConfig)
    reject_invalid_actions: bool = True
    phase_name: str = "phase_1"

    def lane_ids(self) -> tuple[str, ...]:
        return tuple(l.agent_id for l in self.lanes)


def _default_lanes(heterogeneous: bool) -> tuple[LaneConfig, ...]:
    if heterogeneous:
        return (
            LaneConfig(
                agent_id="lane_tokyo",
                destination="Tokyo",
                cutoff_interval=4,
                cutoff_phase=0,
                dispatch_capacity=7.5,
                lane_buffer_capacity=8.5,
                dispatch_cost=5.8,
                base_arrival_rate=0.9,
                demand_pattern=(0.8, 1.0, 1.25, 0.95),
                deadline_mean=7,
                deadline_jitter=2,
                volume_range=(0.7, 2.0),
                urgency_bias=1.3,
            ),
            LaneConfig(
                agent_id="lane_singapore",
                destination="Singapore",
                cutoff_interval=5,
                cutoff_phase=1,
                dispatch_capacity=9.0,
                lane_buffer_capacity=10.0,
                dispatch_cost=6.2,
                base_arrival_rate=0.8,
                demand_pattern=(1.1, 1.0, 0.85, 1.15, 1.05),
                deadline_mean=9,
                deadline_jitter=2,
                volume_range=(0.9, 2.2),
                urgency_bias=1.0,
            ),
            LaneConfig(
                agent_id="lane_dubai",
                destination="Dubai",
                cutoff_interval=6,
                cutoff_phase=2,
                dispatch_capacity=8.5,
                lane_buffer_capacity=9.0,
                dispatch_cost=6.8,
                base_arrival_rate=0.7,
                demand_pattern=(1.2, 0.9, 0.8, 1.1, 1.0, 1.15),
                deadline_mean=10,
                deadline_jitter=3,
                volume_range=(1.0, 2.8),
                urgency_bias=0.85,
            ),
            LaneConfig(
                agent_id="lane_hamburg",
                destination="Hamburg",
                cutoff_interval=7,
                cutoff_phase=3,
                dispatch_capacity=10.0,
                lane_buffer_capacity=11.0,
                dispatch_cost=7.3,
                base_arrival_rate=0.65,
                demand_pattern=(0.9, 0.95, 1.0, 1.1, 1.2, 0.85, 1.05),
                deadline_mean=12,
                deadline_jitter=3,
                volume_range=(1.1, 3.0),
                urgency_bias=0.8,
            ),
            LaneConfig(
                agent_id="lane_los_angeles",
                destination="Los Angeles",
                cutoff_interval=5,
                cutoff_phase=0,
                dispatch_capacity=8.0,
                lane_buffer_capacity=9.5,
                dispatch_cost=6.0,
                base_arrival_rate=1.0,
                demand_pattern=(1.0, 1.2, 0.9, 1.1, 0.85),
                deadline_mean=8,
                deadline_jitter=2,
                volume_range=(0.8, 2.4),
                urgency_bias=1.15,
            ),
        )
    return tuple(
        LaneConfig(
            agent_id=f"lane_{idx}",
            destination=f"Lane-{idx}",
            cutoff_interval=5,
            cutoff_phase=0,
            dispatch_capacity=8.0,
            lane_buffer_capacity=10.0,
            dispatch_cost=6.0,
            base_arrival_rate=0.8,
            demand_pattern=(1.0,),
            deadline_mean=9,
            deadline_jitter=1,
            volume_range=(0.9, 2.2),
            urgency_bias=1.0,
        )
        for idx in range(1, 6)
    )


def build_phase_config(phase: int, seed: int = 7, horizon: int = 48) -> EnvironmentConfig:
    if phase == 1:
        return EnvironmentConfig(
            horizon=horizon,
            seed=seed,
            staging_capacity=100.0,
            dispatch_slots_per_step=5,
            lanes=_default_lanes(heterogeneous=False),
            arrivals=ArrivalConfig(flexible_ratio=0.0),
            observations=ObservationConfig(top_k_shipments=4, include_other_lane_summary=False),
            rewards=RewardConfig(),
            phase_name="phase_1_fixed_lane",
        )
    if phase == 2:
        return EnvironmentConfig(
            horizon=horizon,
            seed=seed,
            staging_capacity=24.0,
            dispatch_slots_per_step=2,
            lanes=_default_lanes(heterogeneous=False),
            arrivals=ArrivalConfig(flexible_ratio=0.35),
            observations=ObservationConfig(top_k_shipments=5, include_other_lane_summary=True),
            rewards=RewardConfig(),
            phase_name="phase_2_flexible_bottlenecks",
        )
    if phase == 3:
        return EnvironmentConfig(
            horizon=horizon,
            seed=seed,
            staging_capacity=22.0,
            dispatch_slots_per_step=2,
            lanes=_default_lanes(heterogeneous=True),
            arrivals=ArrivalConfig(flexible_ratio=0.45, flexible_lane_count_weights=(0.75, 0.25)),
            observations=ObservationConfig(top_k_shipments=5, include_other_lane_summary=True),
            rewards=RewardConfig(),
            phase_name="phase_3_heterogeneous_full",
        )
    raise ValueError(f"Unsupported phase: {phase}")
