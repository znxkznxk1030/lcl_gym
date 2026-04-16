"""
CrossDockEnv — Cross-Docking Multi-Agent RL Environment

구조:
  Inbound Truck  (2~3 목적지 혼재)
      → 인바운드 도어 처리
      → 버퍼 → 레인 큐
      → Outbound Truck (목적지 1개 전용) 탑재 후 출발

에이전트 = 목적지 레인 (num_lanes개)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any

from .entities import Lane, Door, Truck, OutboundTruck
from .conflict_resolver import ConflictResolver


DEFAULT_CONFIG = {
    "num_lanes": 5,
    "num_inbound_doors": 3,
    "buffer_capacity": 150,       # 인바운드 처리량 증가에 맞춰 상향
    "episode_length": 100,
    "truck_arrival_prob": 0.4,    # 도착 확률 상향 (목적지 분산 보정)
    "max_door_processing": 10,
    "inbound_min_dest": 2,        # 인바운드 트럭 최소 목적지 수
    "inbound_max_dest": 3,        # 인바운드 트럭 최대 목적지 수 (inclusive)
    "inbound_vol_min": 5,         # 목적지당 최소 화물량
    "inbound_vol_max": 20,        # 목적지당 최대 화물량
    "outbound_capacity": 50,      # 아웃바운드 트럭 1대 최대 적재량
    "dispatch_interval": 20,      # 아웃바운드 출발 주기 (스텝)
    # reward weights
    "reward_alpha": 0.7,          # 팀 보상 비중
    "reward_beta": 0.3,           # 개인 보상 비중
    # conflict resolver weights
    "cr_alpha": 1.0,
    "cr_beta": 1.0,
    "cr_gamma": 1.0,
}


class CrossDockEnv:
    # ------------------------------------------------------------------
    # Construction / Reset
    # ------------------------------------------------------------------

    def __init__(self, config: dict = None, seed: int = 42):
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.num_lanes: int             = cfg["num_lanes"]
        self.num_inbound_doors: int     = cfg["num_inbound_doors"]
        self.buffer_capacity: int       = cfg["buffer_capacity"]
        self.episode_length: int        = cfg["episode_length"]
        self.truck_arrival_prob: float  = cfg["truck_arrival_prob"]
        self.max_door_processing: int   = cfg["max_door_processing"]
        self.inbound_min_dest: int      = cfg["inbound_min_dest"]
        self.inbound_max_dest: int      = cfg["inbound_max_dest"]
        self.inbound_vol_min: int       = cfg["inbound_vol_min"]
        self.inbound_vol_max: int       = cfg["inbound_vol_max"]
        self.outbound_capacity: float   = cfg["outbound_capacity"]
        self.dispatch_interval: int     = cfg["dispatch_interval"]
        self.reward_alpha: float        = cfg["reward_alpha"]
        self.reward_beta: float         = cfg["reward_beta"]

        self.resolver = ConflictResolver(
            alpha=cfg["cr_alpha"],
            beta=cfg["cr_beta"],
            gamma=cfg["cr_gamma"],
        )

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # obs: [queue, congestion, fill_rate, departure_in,
        #        buffer_remaining, idle_doors, waiting_trucks,
        #        door_match_0 .. door_match_{D-1}]
        self.obs_size: int = 7 + self.num_inbound_doors

        self.lanes: List[Lane] = []
        self.doors: List[Door] = []
        self.outbound_trucks: List[OutboundTruck] = []
        self.buffer: float = 0.0
        self.waiting_trucks: List[Truck] = []
        self.t: int = 0
        self.metrics: Dict[str, Any] = {}

        self.reset()

    def reset(self) -> List[np.ndarray]:
        self.rng = np.random.default_rng(self._seed)
        self.t = 0
        self.buffer = 0.0
        self.waiting_trucks = []

        self.lanes = [Lane(lane_id=k) for k in range(self.num_lanes)]
        self.doors = [Door(door_id=i) for i in range(self.num_inbound_doors)]

        # 각 레인마다 아웃바운드 트럭 1대 대기
        self.outbound_trucks = [
            OutboundTruck(
                lane_id=k,
                capacity=self.outbound_capacity,
                departure_timer=self.dispatch_interval,
            )
            for k in range(self.num_lanes)
        ]

        self.metrics = {
            "total_throughput": 0.0,        # 아웃바운드 탑재된 총 화물량
            "total_fill_rate": 0.0,         # 아웃바운드 출발 탑재율 합산
            "outbound_departures": 0,       # 총 아웃바운드 출발 횟수
            "empty_departures": 0,          # 빈 채로 출발한 횟수 (fill_rate < 0.1)
            "buffer_overflow_count": 0,
            "door_busy_steps": 0,
            "total_steps": 0,
            "dwell_time_sum": 0.0,
            "dwell_count": 0,
        }

        return self.get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        assert len(actions) == self.num_lanes

        # 1. 인바운드 트럭 도착 생성
        new_trucks = self._generate_arrivals()

        # 2. 도어 틱 — 처리 완료된 트럭 방출
        released_trucks = self._tick_doors()

        # 3. 방출된 트럭 화물: 버퍼 → 레인 큐
        overflow = self._process_released(released_trucks)

        # 4. 대기열에 새 트럭 추가
        self.waiting_trucks.extend(new_trucks)

        # 5. 행동 수집 → 충돌 해결 → 도어 배정
        requests = {k: actions[k] for k in range(self.num_lanes)}
        allocation = self.resolver.resolve(
            requests, self.lanes, self.doors,
            self.waiting_trucks, self.outbound_trucks,
        )
        self._assign_doors(allocation)

        # 6. 아웃바운드 트럭 출발 처리
        depart_info = self._depart_outbound()

        # 7. 보상 계산
        rewards = self._compute_rewards(
            depart_info=depart_info,
            overflow=overflow,
        )

        # 8. 메트릭 업데이트
        self._update_metrics(depart_info, overflow)

        self.t += 1
        done = self.t >= self.episode_length
        obs = self.get_obs()
        info = {"t": self.t, "metrics": self.metrics.copy()} if done else {"t": self.t}
        return obs, rewards, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self) -> List[np.ndarray]:
        idle_doors      = sum(1 for d in self.doors if not d.is_busy)
        buffer_remaining = max(self.buffer_capacity - self.buffer, 0.0)
        waiting         = len(self.waiting_trucks)

        obs_list = []
        for k, lane in enumerate(self.lanes):
            ob_truck = self.outbound_trucks[k]

            # 도어별 화물 매칭도: 대기 트럭 중 이 레인 행 화물 비율 최대값
            door_matches = np.zeros(self.num_inbound_doors, dtype=np.float32)
            if self.waiting_trucks:
                best_match = max(
                    t.volume_for_lane(lane.lane_id) / (t.total_volume() + 1e-6)
                    for t in self.waiting_trucks
                )
                for i, door in enumerate(self.doors):
                    if not door.is_busy:
                        door_matches[i] = best_match

            obs = np.array(
                [
                    lane.queue_volume,                # 0: 레인 적재량
                    lane.congestion,                  # 1: 혼잡도 (0~1)
                    ob_truck.fill_rate,               # 2: 아웃바운드 탑재율 (0~1)  ★NEW
                    float(ob_truck.departure_timer),  # 3: 아웃바운드 출발까지 스텝  ★NEW
                    float(buffer_remaining),          # 4: 버퍼 여유량
                    float(idle_doors),                # 5: 유휴 도어 수
                    float(waiting),                   # 6: 대기 인바운드 트럭 수
                ]
                + door_matches.tolist(),              # 7..7+D-1: 도어 매칭도
                dtype=np.float32,
            )
            obs_list.append(obs)
        return obs_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_arrivals(self) -> List[Truck]:
        """인바운드 트럭 생성 — 2~3개 목적지 혼재, 목적지당 5~20 화물량"""
        trucks = []
        if self.rng.random() < self.truck_arrival_prob:
            n_dest = int(self.rng.integers(
                self.inbound_min_dest, self.inbound_max_dest + 1
            ))
            dest_lanes = self.rng.choice(
                self.num_lanes, size=min(n_dest, self.num_lanes), replace=False
            )
            volumes = self.rng.integers(
                self.inbound_vol_min, self.inbound_vol_max + 1, size=len(dest_lanes)
            )
            shipments = {int(k): int(v) for k, v in zip(dest_lanes, volumes)}
            trucks.append(Truck(arrival_time=self.t, shipments=shipments))
        return trucks

    def _tick_doors(self) -> List[Truck]:
        released = []
        for door in self.doors:
            truck = door.tick()
            if truck is not None:
                released.append(truck)
        return released

    def _process_released(self, trucks: List[Truck]) -> int:
        """인바운드 트럭 화물: 버퍼 → 레인 큐"""
        overflow = 0
        for truck in trucks:
            for lane_id, volume in truck.shipments.items():
                space = self.buffer_capacity - self.buffer
                if space <= 0:
                    overflow += 1
                    continue
                actual = min(volume, space)
                self.buffer += actual
                self.lanes[lane_id].add_volume(actual)
                dwell = self.t - truck.arrival_time
                self.metrics["dwell_time_sum"] += dwell
                self.metrics["dwell_count"] += 1
        return overflow

    def _assign_doors(self, allocation: Dict[int, int]):
        for door_idx, lane_id in allocation.items():
            if not self.waiting_trucks:
                break
            truck = self.waiting_trucks.pop(0)
            processing_time = int(
                self.rng.integers(1, self.max_door_processing + 1)
            )
            self.doors[door_idx].assign(truck, lane_id, processing_time)

    def _depart_outbound(self) -> List[dict]:
        """
        아웃바운드 트럭 출발 처리.
        타이머가 0이 된 레인의 아웃바운드 트럭:
          1) 레인 큐에서 잔여 capacity만큼 탑재
          2) fill_rate 기록
          3) 새 아웃바운드 트럭으로 교체
        반환: 레인별 출발 정보 리스트 (출발 없으면 None)
        """
        depart_info = []
        for k in range(self.num_lanes):
            ob = self.outbound_trucks[k]
            if ob.tick():
                # 레인 큐에서 화물 탑재 (용량 범위 내)
                loaded = ob.load(self.lanes[k].take_volume(ob.space_remaining))
                fill = ob.fill_rate

                # 버퍼에서 차감
                self.buffer = max(self.buffer - ob.loaded, 0.0)

                depart_info.append({
                    "lane_id":   k,
                    "loaded":    ob.loaded,
                    "fill_rate": fill,
                    "empty":     fill < 0.1,
                })

                # 새 아웃바운드 트럭 생성
                self.outbound_trucks[k] = OutboundTruck(
                    lane_id=k,
                    capacity=self.outbound_capacity,
                    departure_timer=self.dispatch_interval,
                )
            else:
                depart_info.append(None)

        return depart_info

    def _compute_rewards(
        self,
        depart_info: List,
        overflow: int,
    ) -> List[float]:
        # 이번 스텝에 출발한 트럭 정보 집계
        departures = [d for d in depart_info if d is not None]
        total_loaded     = sum(d["loaded"]    for d in departures)
        empty_departures = sum(1              for d in departures if d["empty"])

        r_team = total_loaded - 0.5 * overflow - 2.0 * empty_departures

        rewards = []
        for k, lane in enumerate(self.lanes):
            d = depart_info[k]
            local_loaded = d["loaded"] if d is not None else 0.0
            r_local = local_loaded - 0.1 * lane.congestion
            r_final = self.reward_alpha * r_team + self.reward_beta * r_local
            rewards.append(float(r_final))
        return rewards

    def _update_metrics(self, depart_info: List, overflow: int):
        for d in depart_info:
            if d is not None:
                self.metrics["total_throughput"]  += d["loaded"]
                self.metrics["total_fill_rate"]   += d["fill_rate"]
                self.metrics["outbound_departures"] += 1
                if d["empty"]:
                    self.metrics["empty_departures"] += 1

        self.metrics["buffer_overflow_count"] += overflow
        self.metrics["door_busy_steps"] += sum(1 for d in self.doors if d.is_busy)
        self.metrics["total_steps"] += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def door_utilization(self) -> float:
        steps = self.metrics["total_steps"]
        if steps == 0:
            return 0.0
        return self.metrics["door_busy_steps"] / (steps * self.num_inbound_doors)

    @property
    def avg_dwell_time(self) -> float:
        if self.metrics["dwell_count"] == 0:
            return 0.0
        return self.metrics["dwell_time_sum"] / self.metrics["dwell_count"]

    @property
    def avg_fill_rate(self) -> float:
        n = self.metrics["outbound_departures"]
        if n == 0:
            return 0.0
        return self.metrics["total_fill_rate"] / n
