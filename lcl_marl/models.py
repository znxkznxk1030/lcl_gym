from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Shipment:
    shipment_id: str
    arrival_step: int
    compatible_lanes: Tuple[str, ...]
    volume: float
    deadline_step: int
    urgency: float
    anchor_lane: str
    fixed_lane: bool
    metadata: dict = field(default_factory=dict)
    accepted_by: Optional[str] = None
    accepted_step: Optional[int] = None
    dispatched_by: Optional[str] = None
    dispatch_step: Optional[int] = None
    overflowed: bool = False
    missed_deadline: bool = False

    def slack(self, current_step: int) -> int:
        return self.deadline_step - current_step

    def waiting_time(self, current_step: int) -> int:
        end_step = self.dispatch_step if self.dispatch_step is not None else current_step
        return max(0, end_step - self.arrival_step)

    def is_compatible(self, lane_id: str) -> bool:
        return lane_id in self.compatible_lanes


@dataclass(frozen=True)
class LaneAction:
    accept_index: int
    request_dispatch: bool = False


@dataclass
class LaneState:
    lane_id: str
    queued_shipments: list[str] = field(default_factory=list)
    queued_volume: float = 0.0
    dispatched_volume: float = 0.0
    dispatch_count: int = 0
    accepted_count: int = 0
    missed_deadlines: int = 0
    invalid_actions: int = 0
    last_dispatch_step: Optional[int] = None


@dataclass
class DispatchRecord:
    lane_id: str
    step: int
    shipment_ids: list[str]
    total_volume: float
    utilization: float
    cost: float
