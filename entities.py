from dataclasses import dataclass, field
from typing import List


@dataclass
class Truck:
    arrival_time: int
    shipments: dict  # {lane_id: volume}

    def total_volume(self) -> int:
        return sum(self.shipments.values())

    def volume_for_lane(self, lane_id: int) -> int:
        return self.shipments.get(lane_id, 0)


@dataclass
class Door:
    door_id: int
    is_busy: bool = False
    remaining_time: int = 0
    assigned_truck: Truck = None
    assigned_lane: int = -1

    def tick(self):
        if self.is_busy:
            self.remaining_time -= 1
            if self.remaining_time <= 0:
                self.is_busy = False
                self.remaining_time = 0
                truck = self.assigned_truck
                self.assigned_truck = None
                self.assigned_lane = -1
                return truck  # return finished truck
        return None

    def assign(self, truck: Truck, lane_id: int, processing_time: int):
        self.is_busy = True
        self.remaining_time = processing_time
        self.assigned_truck = truck
        self.assigned_lane = lane_id


@dataclass
class Lane:
    lane_id: int
    dispatch_interval: int   # fixed interval between dispatches
    queue_volume: float = 0.0
    dispatch_timer: int = 0
    total_dispatched: float = 0.0
    late_dispatches: int = 0
    on_time_dispatches: int = 0
    dwell_times: List[float] = field(default_factory=list)

    @property
    def congestion(self) -> float:
        """Normalized congestion [0, 1] relative to a soft cap of 50 units."""
        return min(self.queue_volume / 50.0, 1.0)

    def add_volume(self, volume: float):
        self.queue_volume += volume

    def tick_dispatch(self) -> float:
        """Decrement timer; dispatch when it hits 0. Return dispatched volume."""
        self.dispatch_timer -= 1
        dispatched = 0.0
        if self.dispatch_timer <= 0:
            dispatched = self.queue_volume
            self.total_dispatched += dispatched
            self.queue_volume = 0.0
            self.dispatch_timer = self.dispatch_interval  # reset
        return dispatched
