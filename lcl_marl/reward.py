from __future__ import annotations

from dataclasses import dataclass, field

from .config import EnvironmentConfig, LaneConfig


@dataclass
class RewardBreakdown:
    team_component: float
    local_components: dict[str, float] = field(default_factory=dict)


def compute_rewards(env: "LCLConsolidationEnv", step_events: dict) -> RewardBreakdown:
    reward_cfg = env.config.rewards
    staging_pressure = env.staging_volume / max(1.0, env.config.staging_capacity)
    team_component = (
        reward_cfg.dispatch_volume_reward * step_events["total_dispatched_volume"]
        - reward_cfg.overflow_penalty * step_events["overflow_shipments"]
        - reward_cfg.missed_deadline_penalty * step_events["new_missed_deadlines"]
        - reward_cfg.staging_pressure_penalty * staging_pressure
    )

    local_components: dict[str, float] = {}
    for lane in env.config.lanes:
        lane_events = step_events["lane_events"][lane.agent_id]
        local = _lane_reward(lane, lane_events, env.config)
        local_components[lane.agent_id] = (
            env.config.rewards.team_weight * team_component
            + env.config.rewards.local_weight * local
        )
    return RewardBreakdown(team_component=team_component, local_components=local_components)


def _lane_reward(lane_config: LaneConfig, lane_events: dict, env_config: EnvironmentConfig) -> float:
    reward_cfg = env_config.rewards
    utilization = 0.0
    if lane_events["dispatch_granted"]:
        utilization = lane_events["dispatch_volume"] / max(1.0, lane_config.dispatch_capacity)
    return (
        reward_cfg.accept_reward * lane_events["accepted_count"]
        + reward_cfg.dispatch_volume_reward * lane_events["dispatch_volume"]
        + reward_cfg.utilization_reward * utilization
        - lane_config.dispatch_cost * lane_events["dispatch_granted"]
        - reward_cfg.invalid_action_penalty * lane_events["invalid_actions"]
        - lane_config.missed_deadline_penalty * lane_events["new_missed_deadlines"]
        - lane_config.holding_cost * lane_events["queued_volume"]
        - reward_cfg.waiting_penalty * lane_events["waiting_shipments"]
    )
