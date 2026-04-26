from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Truck:
    """인바운드 트럭 — 2~3개 목적지 화물 혼재"""
    arrival_time: int
    shipments: dict  # {lane_id: volume}

    def total_volume(self) -> float:
        return sum(self.shipments.values())

    def volume_for_lane(self, lane_id: int) -> float:
        return self.shipments.get(lane_id, 0.0)

    @property
    def num_destinations(self) -> int:
        return len(self.shipments)


@dataclass
class OutboundTruck:
    """아웃바운드 트럭 — 목적지 1개 전용"""
    lane_id: int
    capacity: float
    departure_timer: int
    loaded: float = 0.0

    @property
    def fill_rate(self) -> float:
        return self.loaded / (self.capacity + 1e-8)

    @property
    def space_remaining(self) -> float:
        return max(self.capacity - self.loaded, 0.0)

    def load(self, volume: float) -> float:
        """화물 탑재. 실제 탑재된 양 반환."""
        actual = min(volume, self.space_remaining)
        self.loaded += actual
        return actual

    def tick(self) -> bool:
        """타이머 감소. 출발 여부 반환 (True = 출발)."""
        self.departure_timer -= 1
        return self.departure_timer <= 0


@dataclass
class Door:
    door_id: int
    is_busy: bool = False
    remaining_time: int = 0
    assigned_truck: Optional[Truck] = None
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
                return truck
        return None

    def assign(self, truck: Truck, lane_id: int, processing_time: int):
        self.is_busy = True
        self.remaining_time = processing_time
        self.assigned_truck = truck
        self.assigned_lane = lane_id


@dataclass
class Lane:
    lane_id: int
    queue_volume: float = 0.0    # 인바운드에서 분류된 화물 대기량

    @property
    def congestion(self) -> float:
        """정규화된 혼잡도 [0, 1]. soft cap = 15.0 CBM (아웃바운드 트럭 1대 용량 기준)."""
        return min(self.queue_volume / 15.0, 1.0)

    def add_volume(self, volume: float):
        self.queue_volume += volume

    def take_volume(self, max_volume: float) -> float:
        """아웃바운드 트럭이 화물 가져갈 때. 실제 가져간 양 반환."""
        taken = min(self.queue_volume, max_volume)
        self.queue_volume -= taken
        return taken
