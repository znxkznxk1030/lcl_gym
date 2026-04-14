from __future__ import annotations

from dataclasses import dataclass, field

from .models import DispatchRecord


@dataclass
class LaneMetrics:
    accepted_shipments: int = 0
    dispatched_shipments: int = 0
    dispatched_volume: float = 0.0
    dispatch_count: int = 0
    total_delay: int = 0
    missed_deadlines: int = 0
    invalid_actions: int = 0
    rejected_claims: int = 0
    overflow_losses: int = 0


@dataclass
class MetricsTracker:
    total_arrivals: int = 0
    completed_shipments: int = 0
    overflow_shipments: int = 0
    missed_deadlines: int = 0
    dispatch_count: int = 0
    total_dispatched_volume: float = 0.0
    cumulative_staging_utilization: float = 0.0
    steps_recorded: int = 0
    dispatch_records: list[DispatchRecord] = field(default_factory=list)
    lane_metrics: dict[str, LaneMetrics] = field(default_factory=dict)

    def ensure_lane(self, lane_id: str) -> LaneMetrics:
        if lane_id not in self.lane_metrics:
            self.lane_metrics[lane_id] = LaneMetrics()
        return self.lane_metrics[lane_id]

    def record_staging_utilization(self, utilization: float) -> None:
        self.cumulative_staging_utilization += utilization
        self.steps_recorded += 1

    def snapshot(self) -> dict:
        avg_staging = self.cumulative_staging_utilization / self.steps_recorded if self.steps_recorded else 0.0
        avg_dispatch_util = 0.0
        if self.dispatch_records:
            avg_dispatch_util = sum(record.utilization for record in self.dispatch_records) / len(self.dispatch_records)
        return {
            "total_arrivals": self.total_arrivals,
            "completed_shipments": self.completed_shipments,
            "overflow_shipments": self.overflow_shipments,
            "missed_deadlines": self.missed_deadlines,
            "dispatch_count": self.dispatch_count,
            "total_dispatched_volume": round(self.total_dispatched_volume, 2),
            "avg_staging_utilization": round(avg_staging, 4),
            "avg_dispatch_utilization": round(avg_dispatch_util, 4),
            "lane_metrics": {
                lane_id: {
                    "accepted_shipments": lane.accepted_shipments,
                    "dispatched_shipments": lane.dispatched_shipments,
                    "dispatched_volume": round(lane.dispatched_volume, 2),
                    "dispatch_count": lane.dispatch_count,
                    "avg_delay": round(lane.total_delay / lane.dispatched_shipments, 4) if lane.dispatched_shipments else 0.0,
                    "missed_deadlines": lane.missed_deadlines,
                    "invalid_actions": lane.invalid_actions,
                    "rejected_claims": lane.rejected_claims,
                    "overflow_losses": lane.overflow_losses,
                }
                for lane_id, lane in self.lane_metrics.items()
            },
        }
