from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .config import ConflictConfig, LaneConfig
from .models import LaneState, Shipment


@dataclass
class ClaimResolution:
    winners: dict[str, str]
    rejected: dict[str, list[str]]


def resolve_claims(
    claims: dict[str, str],
    shipments: dict[str, Shipment],
    lane_states: dict[str, LaneState],
    lane_configs: dict[str, LaneConfig],
    current_step: int,
    config: ConflictConfig,
) -> ClaimResolution:
    grouped: dict[str, list[str]] = {}
    for lane_id, shipment_id in claims.items():
        grouped.setdefault(shipment_id, []).append(lane_id)

    winners: dict[str, str] = {}
    rejected: dict[str, list[str]] = {}
    for shipment_id, candidates in grouped.items():
        if len(candidates) == 1:
            winners[candidates[0]] = shipment_id
            continue
        shipment = shipments[shipment_id]
        ranked = sorted(
            candidates,
            key=lambda lane_id: _claim_priority(
                shipment=shipment,
                lane_state=lane_states[lane_id],
                lane_config=lane_configs[lane_id],
                current_step=current_step,
                config=config,
            ),
            reverse=True,
        )
        winner = ranked[0]
        winners[winner] = shipment_id
        for lane_id in ranked[1:]:
            rejected.setdefault(lane_id, []).append(shipment_id)
    return ClaimResolution(winners=winners, rejected=rejected)


def resolve_dispatch_requests(
    requested_lanes: Iterable[str],
    lane_states: dict[str, LaneState],
    lane_configs: dict[str, LaneConfig],
    current_step: int,
    available_slots: int,
) -> tuple[list[str], list[str]]:
    requested = list(requested_lanes)
    ranked = sorted(
        requested,
        key=lambda lane_id: _dispatch_priority(lane_states[lane_id], lane_configs[lane_id], current_step),
        reverse=True,
    )
    approved = ranked[:available_slots]
    denied = ranked[available_slots:]
    return approved, denied


def _claim_priority(
    shipment: Shipment,
    lane_state: LaneState,
    lane_config: LaneConfig,
    current_step: int,
    config: ConflictConfig,
) -> float:
    slack = max(1, shipment.deadline_step - current_step)
    cutoff_pressure = 1.0 / max(1, time_to_cutoff(current_step, lane_config))
    backlog_ratio = lane_state.queued_volume / max(1.0, lane_config.lane_buffer_capacity)
    return (
        config.claim_priority_weight * (shipment.urgency + 1.0 / slack)
        + config.cutoff_priority_weight * cutoff_pressure
        - config.backlog_priority_weight * backlog_ratio
    )


def _dispatch_priority(lane_state: LaneState, lane_config: LaneConfig, current_step: int) -> float:
    utilization = lane_state.queued_volume / max(1.0, lane_config.dispatch_capacity)
    cutoff_pressure = 1.0 / max(1, time_to_cutoff(current_step, lane_config))
    overdue_pressure = lane_state.missed_deadlines
    return 2.0 * utilization + cutoff_pressure + 0.5 * overdue_pressure


def time_to_cutoff(current_step: int, lane_config: LaneConfig) -> int:
    remainder = current_step % lane_config.cutoff_interval
    offset = (lane_config.cutoff_phase - remainder) % lane_config.cutoff_interval
    return offset
