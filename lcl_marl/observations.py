from __future__ import annotations

from typing import Optional

from .config import EnvironmentConfig, LaneConfig
from .conflict_resolution import time_to_cutoff
from .models import LaneState, Shipment


def build_observations(env: "LCLConsolidationEnv") -> tuple[dict[str, dict], dict[str, list[Optional[str]]], dict[str, dict]]:
    observations: dict[str, dict] = {}
    visibility: dict[str, list[Optional[str]]] = {}
    masks: dict[str, dict] = {}
    for lane in env.config.lanes:
        visible_shipments = _visible_shipments(env, lane)
        visibility[lane.agent_id] = visible_shipments
        masks[lane.agent_id] = _build_action_mask(env, lane, visible_shipments)
        observations[lane.agent_id] = {
            "agent_id": lane.agent_id,
            "step": env.current_step,
            "phase": env.config.phase_name,
            "local_state": _local_state(env, lane),
            "visible_shipments": [
                _shipment_view(env.shipments[shipment_id], env.current_step)
                for shipment_id in visible_shipments
                if shipment_id is not None
            ],
            "global_summary": _global_summary(env),
            "other_lanes": _other_lane_summary(env, lane) if env.config.observations.include_other_lane_summary else [],
            "action_mask": masks[lane.agent_id] if env.config.observations.include_action_mask else None,
        }
    return observations, visibility, masks


def build_centralized_state(env: "LCLConsolidationEnv") -> dict:
    return {
        "step": env.current_step,
        "horizon": env.config.horizon,
        "phase": env.config.phase_name,
        "staging": {
            "shipment_count": len(env.staging_queue),
            "total_volume": round(env.staging_volume, 2),
            "capacity": env.config.staging_capacity,
        },
        "dispatch_slots_per_step": env.config.dispatch_slots_per_step,
        "lanes": [
            {
                "lane_id": lane.agent_id,
                "queued_shipments": len(env.lane_states[lane.agent_id].queued_shipments),
                "queued_volume": round(env.lane_states[lane.agent_id].queued_volume, 2),
                "dispatch_capacity": lane.dispatch_capacity,
                "time_to_cutoff": time_to_cutoff(env.current_step, lane),
            }
            for lane in env.config.lanes
        ],
        "metrics": env.metrics.snapshot(),
    }


def _visible_shipments(env: "LCLConsolidationEnv", lane: LaneConfig) -> list[Optional[str]]:
    candidates = []
    lane_state = env.lane_states[lane.agent_id]
    for shipment_id in env.staging_queue:
        shipment = env.shipments[shipment_id]
        if lane.agent_id not in shipment.compatible_lanes:
            continue
        remaining_capacity = lane.lane_buffer_capacity - lane_state.queued_volume
        if shipment.volume > remaining_capacity:
            continue
        score = shipment.urgency + (1.0 / max(1, shipment.slack(env.current_step))) + (0.25 if shipment.fixed_lane else 0.0)
        candidates.append((score, shipment_id))
    candidates.sort(reverse=True)
    top_ids = [shipment_id for _, shipment_id in candidates[: env.config.observations.top_k_shipments]]
    while len(top_ids) < env.config.observations.top_k_shipments:
        top_ids.append(None)
    return top_ids


def _build_action_mask(env: "LCLConsolidationEnv", lane: LaneConfig, visible_shipments: list[Optional[str]]) -> dict:
    accept_mask = [1 if shipment_id is not None else 0 for shipment_id in visible_shipments]
    accept_mask.append(1)
    dispatch_allowed = int(
        env.lane_states[lane.agent_id].queued_volume > 0
        and time_to_cutoff(env.current_step, lane) == 0
    )
    dispatch_mask = [1, dispatch_allowed]
    joint_mask: list[int] = []
    for accept_index, accept_valid in enumerate(accept_mask):
        for request_dispatch, dispatch_valid in enumerate(dispatch_mask):
            if accept_valid == 0:
                joint_mask.append(0)
            elif request_dispatch == 1 and dispatch_valid == 0:
                joint_mask.append(0)
            else:
                joint_mask.append(1)
    return {
        "accept_mask": accept_mask,
        "dispatch_mask": dispatch_mask,
        "joint_mask": joint_mask,
    }


def _local_state(env: "LCLConsolidationEnv", lane: LaneConfig) -> dict:
    lane_state = env.lane_states[lane.agent_id]
    queue_shipments = [env.shipments[shipment_id] for shipment_id in lane_state.queued_shipments]
    overdue = sum(1 for shipment in queue_shipments if shipment.deadline_step <= env.current_step)
    return {
        "destination": lane.destination,
        "queued_shipments": len(queue_shipments),
        "queued_volume": round(lane_state.queued_volume, 2),
        "buffer_capacity": lane.lane_buffer_capacity,
        "dispatch_capacity": lane.dispatch_capacity,
        "dispatch_cost": lane.dispatch_cost,
        "time_to_cutoff": time_to_cutoff(env.current_step, lane),
        "overdue_shipments": overdue,
    }


def _global_summary(env: "LCLConsolidationEnv") -> dict:
    utilization = env.staging_volume / max(1.0, env.config.staging_capacity)
    return {
        "remaining_steps": env.config.horizon - env.current_step,
        "staging_shipments": len(env.staging_queue),
        "staging_volume": round(env.staging_volume, 2),
        "staging_utilization": round(utilization, 4),
        "dispatch_slots_per_step": env.config.dispatch_slots_per_step,
        "overflow_shipments": env.metrics.overflow_shipments,
        "missed_deadlines": env.metrics.missed_deadlines,
    }


def _other_lane_summary(env: "LCLConsolidationEnv", lane: LaneConfig) -> list[dict]:
    summaries = []
    for other in env.config.lanes:
        if other.agent_id == lane.agent_id:
            continue
        other_state = env.lane_states[other.agent_id]
        summaries.append(
            {
                "lane_id": other.agent_id,
                "queued_shipments": len(other_state.queued_shipments),
                "queued_volume_ratio": round(other_state.queued_volume / max(1.0, other.lane_buffer_capacity), 4),
                "time_to_cutoff": time_to_cutoff(env.current_step, other),
                "dispatch_count": other_state.dispatch_count,
            }
        )
    return summaries


def _shipment_view(shipment: Shipment, current_step: int) -> dict:
    return {
        "shipment_id": shipment.shipment_id,
        "volume": shipment.volume,
        "compatible_lanes": shipment.compatible_lanes,
        "deadline_step": shipment.deadline_step,
        "slack": shipment.slack(current_step),
        "urgency": shipment.urgency,
        "fixed_lane": shipment.fixed_lane,
        "anchor_lane": shipment.anchor_lane,
    }
