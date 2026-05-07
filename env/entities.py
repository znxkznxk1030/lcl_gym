from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Truck:
    """인바운드 트럭 — 2~3개 목적지 화물 혼재"""
    arrival_time: int
    shipments: dict  # {lane_id: volume}
    is_rush: bool = False  # 긴급 트럭 여부 (돌발 발생)

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
    is_failed: bool = False       # 고장 여부
    failure_remaining: int = 0    # 고장 잔여 스텝
    remaining_time: int = 0
    assigned_truck: Optional[Truck] = None
    assigned_lane: int = -1

    def tick(self):
        # 고장 중: 카운트다운만 진행, 화물 처리 없음
        if self.is_failed:
            self.failure_remaining -= 1
            if self.failure_remaining <= 0:
                self.is_failed = False
                self.failure_remaining = 0
            return None
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

    def fail(self, duration: int) -> Optional[Truck]:
        """도어 고장 처리. 처리 중이던 트럭은 대기열로 반환."""
        self.is_failed = True
        self.failure_remaining = duration
        interrupted = None
        if self.is_busy:
            interrupted = self.assigned_truck
            self.is_busy = False
            self.remaining_time = 0
            self.assigned_truck = None
            self.assigned_lane = -1
        return interrupted


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
