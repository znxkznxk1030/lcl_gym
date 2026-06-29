from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Truck:
    """인바운드 트럭 — 2~3개 목적지 화물 혼재

    Compound 모드(논문 Shahmardan & Sajadieh 2020):
      - truck_type="compound": 여러 목적지 화물(f_idk)을 싣고 와서, 배정된 목적지(assigned_dest)
        화물은 보유한 채(kept_volume) 나머지만 하차(unloaded_volume)한 뒤 자신의 목적지로 출발.
      - truck_type="outbound": 단일 목적지 전용 출고 차량 (Stage-1 하차 없음).
    """
    arrival_time: int
    shipments: dict  # {dest_id: volume}  (compound: f_idk for single product type)
    is_rush: bool = False  # 긴급 트럭 여부 (돌발 발생)
    routed_lane: int = -1  # (레거시) compound 모드에서 정책이 지정한 아웃바운드 레인
    # --- compound 모드 필드 (기본값으로 non-compound 무영향) ---
    truck_type: str = "inbound"   # "compound" | "outbound" | "inbound"(레거시)
    assigned_dest: int = -1       # compound 트럭이 보유·운반하는 단일 목적지
    kept_volume: float = 0.0      # assigned_dest 화물 중 트럭에 남겨둔 양 (partial 하차)
    unloaded_volume: float = 0.0  # 하차되는(=도어 점유시간 산정) 총량
    DE: int = 0                   # entering time (도어 위치 도달, 논문 DE_i)
    DL: int = 0                   # exiting time  (도어 이탈, 논문 DL_i)

    def total_volume(self) -> float:
        return sum(self.shipments.values())

    def volume_for_lane(self, lane_id: int) -> float:
        return self.shipments.get(lane_id, 0.0)

    def dest_volume(self, dest: int) -> float:
        return self.shipments.get(dest, 0.0)

    def argmax_dest(self) -> int:
        """화물량이 가장 많은 목적지 — compound 트럭이 보유할 목적지(논문: 목적지 1개 배정)."""
        if not self.shipments:
            return -1
        return max(self.shipments, key=self.shipments.get)

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
        truck.routed_lane = lane_id  # 화물이 도착할 레인을 트럭에 기록

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
class OutboundDoor:
    """
    Stage 2 아웃바운드 로딩 도크.

    목적지(assigned_dest)는 트럭이 출발할 때마다 동적으로 재할당된다.
    loading_timer 가 0이 되면 현재 트럭이 출발하고 도크는 idle 상태로 전환.
    """

    door_id: int
    capacity: float = 15.0
    is_busy: bool = False
    assigned_dest: int = -1   # 현재 로딩 중인 목적지 (-1 = idle)
    loaded: float = 0.0
    loading_timer: int = 0    # 잔여 로딩 타임스텝
    # --- compound 모드 ---
    serving_truck: Optional["Truck"] = None  # 이 도크를 점유한 compound/outbound 트럭 (None=레거시)
    preloaded: float = 0.0                    # 도킹 시점에 이미 트럭에 실려있던 양 (kept_volume)

    @property
    def fill_rate(self) -> float:
        return self.loaded / (self.capacity + 1e-8)

    @property
    def space_remaining(self) -> float:
        return max(self.capacity - self.loaded, 0.0)

    def start_loading(self, dest: int, loading_time: int,
                      truck: Optional["Truck"] = None, preloaded: float = 0.0) -> None:
        """새 아웃바운드 트럭 수령 + 목적지 할당.

        compound 모드: truck/preloaded 전달 시 kept_volume(preloaded)을 적재 상태로 시작.
        """
        self.is_busy = True
        self.assigned_dest = dest
        self.loaded = float(preloaded)
        self.loading_timer = loading_time
        self.serving_truck = truck
        self.preloaded = float(preloaded)

    def add_load(self, volume: float) -> float:
        """소팅 레인에서 화물 탑재. 실제 탑재량 반환."""
        actual = min(volume, self.space_remaining)
        self.loaded += actual
        return actual

    def tick(self) -> Tuple[bool, Optional[dict]]:
        """
        타이머 1 감소.
        Returns
        -------
        (departed, depart_info) — departed=True 이면 트럭이 출발.
        """
        if not self.is_busy:
            return False, None
        self.loading_timer -= 1
        if self.loading_timer <= 0:
            info = {
                "door_id":     self.door_id,
                "dest":        self.assigned_dest,
                "loaded":      self.loaded,
                "fill_rate":   self.fill_rate,
                "empty":       self.fill_rate < 0.1,
                "is_compound": self.serving_truck is not None,
                "preloaded":   self.preloaded,
            }
            self.is_busy = False
            self.assigned_dest = -1
            self.loaded = 0.0
            self.serving_truck = None
            self.preloaded = 0.0
            return True, info
        return False, None


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
