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


DEFAULT_CONFIG = {
    "num_lanes": 5,
    "num_inbound_doors": 3,
    "buffer_capacity": 60.0,      # CBM — 트럭 4~5대분 스테이징 공간
    "episode_length": 100,
    "truck_arrival_prob": 0.4,    # use_scheduled_arrivals=False 일 때만 사용
    "max_door_processing": 10,
    "inbound_min_dest": 2,        # 인바운드 트럭 최소 목적지 수
    "inbound_max_dest": 3,        # 인바운드 트럭 최대 목적지 수 (inclusive)
    "inbound_vol_min": 0.5,       # CBM — 목적지당 최소 화물량
    "inbound_vol_max": 5.0,       # CBM — 목적지당 최대 화물량
    "outbound_capacity": 15.0,    # CBM — 아웃바운드 트럭 1대 최대 적재량 (소형 트럭)
    "dispatch_interval": 20,      # use_staggered_dispatch=False 일 때만 사용
    # 스케줄 기반 출고
    "use_staggered_dispatch": True,
    "dispatch_interval_min": 12,  # 레인별 출발 주기 최솟값 (스텝)
    "dispatch_interval_max": 28,  # 레인별 출발 주기 최댓값 (스텝)
    # 레인별 고정 출발 주기 (list 설정 시 use_staggered_dispatch 무시)
    # 예: [8, 14, 22, 32, 45] → Lane 0이 가장 빠르고 Lane 4가 가장 느림
    "lane_dispatch_intervals": None,
    # ── Truck-Selection RL 모드 ──────────────────────────────────────────────
    # True: 에이전트 = 트럭슬롯 K개, 액션 = 이 트럭 선택/건너뜀
    # False: 기존 레인 에이전트 모드
    "use_truck_selection": False,
    "top_k_trucks": 15,
    # 스케줄 기반 입고
    "use_scheduled_arrivals": True,
    "arrival_count_min": 50,      # 에피소드당 최소 인바운드 트럭 수
    "arrival_count_max": 70,      # 에피소드당 최대 인바운드 트럭 수
    "arrival_pattern": "clustered", # "uniform" | "clustered"
    "arrival_cluster_count": 4,   # clustered 패턴에서 클러스터(배송 배치) 수
    # reward weights
    "reward_alpha": 0.7,          # 팀 보상 비중
    "reward_beta": 0.3,           # 개인 보상 비중
    # conflict resolver weights
    "cr_alpha": 1.0,
    "cr_beta": 1.0,
    "cr_gamma": 1.0,
    # ── 돌발사항 (Disruptions) ──────────────────────────────────────────
    "enable_disruptions": False,
    # 1) 도어 고장: 랜덤 도어 1개가 N스텝간 마비 (처리 중 트럭도 대기열로 복귀)
    "disruption_door_failure": False,
    "disruption_door_failure_prob": 0.015,       # 스텝당 발생 확률
    "disruption_door_failure_duration_min": 5,
    "disruption_door_failure_duration_max": 12,
    # 2) 긴급 트럭: 미스케줄 고용량 트럭이 돌발 도착 (전 레인 화물 혼재, 대기열 앞 삽입)
    "disruption_rush_truck": False,
    "disruption_rush_truck_prob": 0.025,         # 스텝당 발생 확률
    "disruption_rush_volume_min": 6.0,           # 레인당 화물량 CBM
    "disruption_rush_volume_max": 12.0,
    # 3) 아웃바운드 타이머 쇼크: 특정 레인 출발 타이머가 강제로 단축
    "disruption_timer_shock": False,
    "disruption_timer_shock_prob": 0.025,        # 스텝·레인당 발생 확률
    "disruption_timer_shock_min": 2,             # 강제 잔여 스텝 범위
    "disruption_timer_shock_max": 4,
}


class CrossDockEnv:
    # ------------------------------------------------------------------
    # Construction / Reset
    # ------------------------------------------------------------------

    def __init__(self, config: dict = None, seed: int = 42):
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.num_lanes: int             = cfg["num_lanes"]
        self.num_inbound_doors: int     = cfg["num_inbound_doors"]
        self.buffer_capacity: float     = cfg["buffer_capacity"]
        self.episode_length: int        = cfg["episode_length"]
        self.truck_arrival_prob: float  = cfg["truck_arrival_prob"]
        self.max_door_processing: int   = cfg["max_door_processing"]
        self.inbound_min_dest: int      = cfg["inbound_min_dest"]
        self.inbound_max_dest: int      = cfg["inbound_max_dest"]
        self.inbound_vol_min: float     = cfg["inbound_vol_min"]
        self.inbound_vol_max: float     = cfg["inbound_vol_max"]
        self.outbound_capacity: float     = cfg["outbound_capacity"]
        self.dispatch_interval: int       = cfg["dispatch_interval"]
        self.use_staggered_dispatch: bool = cfg["use_staggered_dispatch"]
        self.dispatch_interval_min: int   = cfg["dispatch_interval_min"]
        self.dispatch_interval_max: int   = cfg["dispatch_interval_max"]
        self.lane_dispatch_intervals: list  = cfg["lane_dispatch_intervals"]  # None or list
        self.use_truck_selection: bool      = cfg["use_truck_selection"]
        self.top_k_trucks: int              = cfg["top_k_trucks"]
        self.reward_alpha: float          = cfg["reward_alpha"]
        self.reward_beta: float         = cfg["reward_beta"]
        self.use_scheduled_arrivals: bool = cfg["use_scheduled_arrivals"]
        self.arrival_count_min: int     = cfg["arrival_count_min"]
        self.arrival_count_max: int     = cfg["arrival_count_max"]
        self.arrival_pattern: str       = cfg["arrival_pattern"]
        self.arrival_cluster_count: int = cfg["arrival_cluster_count"]

        # 돌발사항 설정
        self.enable_disruptions: bool            = cfg["enable_disruptions"]
        self.disruption_door_failure: bool       = cfg["disruption_door_failure"]
        self.disruption_door_failure_prob: float = cfg["disruption_door_failure_prob"]
        self.disruption_door_failure_dur_min: int = cfg["disruption_door_failure_duration_min"]
        self.disruption_door_failure_dur_max: int = cfg["disruption_door_failure_duration_max"]
        self.disruption_rush_truck: bool         = cfg["disruption_rush_truck"]
        self.disruption_rush_truck_prob: float   = cfg["disruption_rush_truck_prob"]
        self.disruption_rush_vol_min: float      = cfg["disruption_rush_volume_min"]
        self.disruption_rush_vol_max: float      = cfg["disruption_rush_volume_max"]
        self.disruption_timer_shock: bool        = cfg["disruption_timer_shock"]
        self.disruption_timer_shock_prob: float  = cfg["disruption_timer_shock_prob"]
        self.disruption_timer_shock_min: int     = cfg["disruption_timer_shock_min"]
        self.disruption_timer_shock_max: int     = cfg["disruption_timer_shock_max"]

        self.disruption_log: List[dict] = []  # 현재 스텝 돌발 기록


        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # obs 크기
        # - 기존 레인 모드:  [queue, congestion, fill_rate, timer,
        #                     buffer_remaining, idle_doors, waiting, scheduled,
        #                     door_match_0..D-1]  = 8 + D
        # - 트럭선택 모드:   [timer×nL, queue×nL, buffer_remaining, idle_doors,
        #                     waiting,              ← 글로벌 13개
        #                     vol_L0..L4, total_vol, is_rush]  ← 트럭 7개
        #                   = 13 + 7 = 20
        if self.use_truck_selection:
            self.obs_size: int = self.num_lanes * 2 + 3 + 7   # 13 + 7 = 20
        else:
            self.obs_size: int = 8 + self.num_inbound_doors

        self.lanes: List[Lane] = []
        self.doors: List[Door] = []
        self.outbound_trucks: List[OutboundTruck] = []
        self.buffer: float = 0.0
        self.waiting_trucks: List[Truck] = []
        self.arrival_schedule: List[Truck] = []  # 스케줄 기반 입고 대기열
        self.t: int = 0
        self.metrics: Dict[str, Any] = {}

        self.reset()

    def reset(self) -> List[np.ndarray]:
        self.rng = np.random.default_rng(self._seed)
        self.t = 0
        self.buffer = 0.0
        self.waiting_trucks = []
        self.arrival_schedule = self._build_arrival_schedule() if self.use_scheduled_arrivals else []

        self.lanes = [Lane(lane_id=k) for k in range(self.num_lanes)]
        self.doors = [Door(door_id=i) for i in range(self.num_inbound_doors)]

        # 각 레인마다 아웃바운드 트럭 1대 대기 — 초기 타이머를 랜덤하게 배분해 출발 시각 분산
        self.outbound_trucks = [
            OutboundTruck(
                lane_id=k,
                capacity=self.outbound_capacity,
                departure_timer=self._sample_dispatch_timer(initial=True, lane_id=k),
            )
            for k in range(self.num_lanes)
        ]

        self.metrics = {
            "total_throughput": 0.0,
            "total_fill_rate": 0.0,
            "outbound_departures": 0,
            "empty_departures": 0,
            "buffer_overflow_count": 0,
            "door_busy_steps": 0,
            "total_steps": 0,
            "dwell_time_sum": 0.0,
            "dwell_count": 0,
            # 돌발사항 카운터
            "disruption_door_failures": 0,
            "disruption_interrupted_trucks": 0,
            "disruption_rush_trucks": 0,
            "disruption_timer_shocks": 0,
        }

        return self.get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        expected = self.top_k_trucks if self.use_truck_selection else self.num_lanes
        assert len(actions) == expected, f"Expected {expected} actions, got {len(actions)}"

        # 0. 돌발사항 적용
        self.disruption_log = []
        if self.enable_disruptions:
            self._apply_disruptions()

        # 1. 인바운드 트럭 도착 생성
        new_trucks = self._generate_arrivals()

        # 2. 도어 틱 — 처리 완료된 트럭 방출
        released_trucks = self._tick_doors()

        # 3. 방출된 트럭 화물: 버퍼 → 레인 큐
        overflow = self._process_released(released_trucks)

        # 4. 대기열에 새 트럭 추가
        self.waiting_trucks.extend(new_trucks)

        # 5. 행동 수집 → 요청 레인을 우선순위 순으로 유휴 도어에 배정
        self._assign_doors(actions)

        # 5.5. 레인 큐 → 아웃바운드 트럭 지속 적재 (매 스텝 점진 탑재)
        self._progressive_load()

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
        if self.use_truck_selection:
            return self._get_obs_truck_selection()
        return self._get_obs_lane()

    def _get_obs_lane(self) -> List[np.ndarray]:
        """기존 레인 에이전트 모드 obs."""
        idle_doors       = sum(1 for d in self.doors if not d.is_busy)
        buffer_remaining = max(self.buffer_capacity - self.buffer, 0.0)
        waiting          = len(self.waiting_trucks)
        scheduled        = len(self.arrival_schedule)

        obs_list = []
        for k, lane in enumerate(self.lanes):
            ob_truck = self.outbound_trucks[k]

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
                    lane.queue_volume,
                    lane.congestion,
                    ob_truck.fill_rate,
                    float(ob_truck.departure_timer),
                    float(buffer_remaining),
                    float(idle_doors),
                    float(waiting),
                    float(scheduled),
                ]
                + door_matches.tolist(),
                dtype=np.float32,
            )
            obs_list.append(obs)
        return obs_list

    def _get_obs_truck_selection(self) -> List[np.ndarray]:
        """트럭선택 모드 obs.

        에이전트 = 트럭슬롯 K개.
        각 에이전트 obs (20개):
          글로벌(13): departure_timer×nL, queue_volume×nL,
                      buffer_remaining, idle_doors, waiting_count
          트럭(7):    vol_L0..L{nL-1}, total_vol, is_rush
        빈 슬롯(대기 트럭 부족)은 트럭 7개 값을 0으로 패딩.
        """
        K  = self.top_k_trucks
        nL = self.num_lanes

        timers   = np.array([ob.departure_timer for ob in self.outbound_trucks], dtype=np.float32)
        queues   = np.array([lane.queue_volume   for lane in self.lanes],        dtype=np.float32)
        buf_rem  = float(max(self.buffer_capacity - self.buffer, 0.0))
        idle     = float(sum(1 for d in self.doors if not d.is_busy))
        n_wait   = float(len(self.waiting_trucks))

        global_ctx = np.concatenate([timers, queues, [buf_rem, idle, n_wait]])  # (13,)

        trucks = self.waiting_trucks[:K]
        obs_list = []
        for i in range(K):
            if i < len(trucks):
                t = trucks[i]
                vols = np.array([float(t.shipments.get(k, 0)) for k in range(nL)], dtype=np.float32)
                truck_feat = np.array(
                    [*vols, float(t.total_volume()), float(getattr(t, "is_rush", False))],
                    dtype=np.float32,
                )
            else:
                truck_feat = np.zeros(nL + 2, dtype=np.float32)  # 패딩

            obs_list.append(np.concatenate([global_ctx, truck_feat]))
        return obs_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_dispatch_timer(self, initial: bool = False, lane_id: int = 0) -> int:
        # 레인별 고정 주기가 설정된 경우
        if self.lane_dispatch_intervals is not None:
            interval = int(self.lane_dispatch_intervals[lane_id % len(self.lane_dispatch_intervals)])
            if initial:
                return int(self.rng.integers(1, interval + 1))
            return interval
        if not self.use_staggered_dispatch:
            return self.dispatch_interval
        if initial:
            # 첫 출발을 레인마다 분산시키기 위해 [1, max] 전 구간에서 샘플
            return int(self.rng.integers(1, self.dispatch_interval_max + 1))
        return int(self.rng.integers(self.dispatch_interval_min, self.dispatch_interval_max + 1))

    def _build_arrival_schedule(self) -> List[Truck]:
        """에피소드 시작 시 전체 입고 스케줄을 미리 생성."""
        n = int(self.rng.integers(self.arrival_count_min, self.arrival_count_max + 1))

        if self.arrival_pattern == "clustered":
            # 클러스터 중심을 에피소드 전반에 균등 배치 + 소폭 랜덤 오프셋
            base = np.linspace(0.1, 0.9, self.arrival_cluster_count)
            jitter = self.rng.uniform(-0.05, 0.05, size=self.arrival_cluster_count)
            centers = np.clip(base + jitter, 0.05, 0.95) * self.episode_length
            cluster_ids = self.rng.integers(0, self.arrival_cluster_count, size=n)
            spread = self.episode_length * 0.08  # 클러스터 폭 (에피소드의 8%)
            raw_times = centers[cluster_ids] + self.rng.normal(0, spread, size=n)
            arrival_times = sorted(int(np.clip(t, 0, self.episode_length - 1)) for t in raw_times)
        else:  # uniform
            arrival_times = sorted(
                int(t) for t in self.rng.integers(0, self.episode_length, size=n)
            )

        schedule = []
        for t in arrival_times:
            n_dest = int(self.rng.integers(self.inbound_min_dest, self.inbound_max_dest + 1))
            dest_lanes = self.rng.choice(
                self.num_lanes, size=min(n_dest, self.num_lanes), replace=False
            )
            volumes = self.rng.integers(
                max(1, int(self.inbound_vol_min)),
                int(self.inbound_vol_max) + 1,
                size=len(dest_lanes),
            )
            shipments = {int(k): int(v) for k, v in zip(dest_lanes, volumes)}
            schedule.append(Truck(arrival_time=t, shipments=shipments))
        return schedule

    def _generate_arrivals(self) -> List[Truck]:
        """인바운드 트럭 도착 — 스케줄 모드 또는 확률 모드."""
        if self.use_scheduled_arrivals:
            trucks = []
            while self.arrival_schedule and self.arrival_schedule[0].arrival_time <= self.t:
                trucks.append(self.arrival_schedule.pop(0))
            return trucks

        # 기존 확률 기반 모드
        if self.rng.random() < self.truck_arrival_prob:
            n_dest = int(self.rng.integers(
                self.inbound_min_dest, self.inbound_max_dest + 1
            ))
            dest_lanes = self.rng.choice(
                self.num_lanes, size=min(n_dest, self.num_lanes), replace=False
            )
            volumes = self.rng.integers(
                max(1, int(self.inbound_vol_min)),
                int(self.inbound_vol_max) + 1,
                size=len(dest_lanes),
            )
            shipments = {int(k): int(v) for k, v in zip(dest_lanes, volumes)}
            return [Truck(arrival_time=self.t, shipments=shipments)]
        return []

    def _tick_doors(self) -> List[Truck]:
        released = []
        for door in self.doors:
            truck = door.tick()
            if truck is not None:
                released.append(truck)
        return released

    def _process_released(self, trucks: List[Truck]) -> int:
        """인바운드 트럭 화물: 버퍼 → 레인 큐
        - 버퍼 완전 포화: 트럭 전체 반환
        - 버퍼 부분 여유: 들어가는 만큼 적재, 나머지 화물만 담은 부분 트럭을 반환
          → 화물 소실 없음 (보존 보장)"""
        overflow = 0
        returned: List[Truck] = []

        for truck in trucks:
            if self.buffer_capacity - self.buffer <= 0:
                returned.append(truck)
                continue

            leftover: dict = {}
            for lane_id, volume in truck.shipments.items():
                space  = int(self.buffer_capacity - self.buffer)  # floor → 항상 정수
                actual = min(int(volume), space)                   # 화물량도 정수 보장
                if actual > 0:
                    self.buffer += actual
                    self.lanes[lane_id].add_volume(actual)
                    dwell = self.t - truck.arrival_time
                    self.metrics["dwell_time_sum"] += dwell
                    self.metrics["dwell_count"] += 1
                remaining = volume - actual
                if remaining > 0:
                    leftover[lane_id] = remaining

            if leftover:
                # 나머지 화물을 부분 트럭으로 만들어 큐 앞에 반환
                partial = Truck(
                    arrival_time=truck.arrival_time,
                    shipments=leftover,
                    is_rush=truck.is_rush,
                )
                returned.append(partial)
                overflow += 1

        for truck in reversed(returned):
            self.waiting_trucks.insert(0, truck)

        return overflow

    def _assign_doors(self, actions: List[int]):
        if self.use_truck_selection:
            self._assign_doors_truck_selection(actions)
        else:
            self._assign_doors_lane(actions)

    def _assign_doors_lane(self, actions: List[int]):
        """기존 레인 모드: 요청 레인 긴급도 순 → FIFO 트럭 배정."""
        idle_doors = [d for d in self.doors if not d.is_busy and not d.is_failed]
        if not idle_doors or not self.waiting_trucks:
            return

        requesting = [k for k, a in enumerate(actions) if a == 1]
        if not requesting:
            return

        requesting.sort(key=lambda k: self.outbound_trucks[k].departure_timer)

        for door, lane_id in zip(idle_doors, requesting):
            if not self.waiting_trucks:
                break
            truck = self.waiting_trucks.pop(0)
            processing_time = int(self.rng.integers(1, self.max_door_processing + 1))
            door.assign(truck, lane_id, processing_time)

    def _assign_doors_truck_selection(self, actions: List[int]):
        """트럭선택 모드: 선택된 슬롯(action=1) 트럭을 큐 앞으로 이동 → 유휴 도어 FIFO 배정.

        actions 길이 = top_k_trucks.
        actions[i]=1 → i번 슬롯 트럭을 선택.
        선택된 트럭들이 큐 앞으로 이동되고, 유휴 도어 수만큼 순서대로 배정.
        아무도 선택 안 하면 아무 도어도 열지 않음(RL이 선택을 강제로 배움).
        """
        idle_doors = [d for d in self.doors if not d.is_busy and not d.is_failed]
        if not idle_doors or not self.waiting_trucks:
            return

        n_avail = len(self.waiting_trucks)
        selected = [i for i, a in enumerate(actions) if a == 1 and i < n_avail]
        if not selected:
            return

        # 선택된 트럭을 큐 앞으로, 나머지는 뒤로
        sel_set = set(selected)
        front = [self.waiting_trucks[i] for i in selected]
        rest  = [t for i, t in enumerate(self.waiting_trucks) if i not in sel_set]
        self.waiting_trucks = front + rest

        # 유휴 도어에 FIFO 배정
        for door in idle_doors:
            if not self.waiting_trucks:
                break
            truck = self.waiting_trucks.pop(0)
            # 이 트럭에서 가장 긴급한 레인을 assigned_lane으로
            if truck.shipments:
                lane_id = min(
                    truck.shipments.keys(),
                    key=lambda k: self.outbound_trucks[k].departure_timer,
                )
            else:
                lane_id = 0
            processing_time = int(self.rng.integers(1, self.max_door_processing + 1))
            door.assign(truck, lane_id, processing_time)

    def _progressive_load(self):
        """레인 큐 → 아웃바운드 트럭으로 매 스텝 점진 적재.
        화물이 레인에 도착하는 즉시 대기 트럭에 탑재해 fill_rate가 실시간으로 증가."""
        for k, lane in enumerate(self.lanes):
            ob = self.outbound_trucks[k]
            if ob.space_remaining > 0 and lane.queue_volume > 0:
                transferable = int(min(lane.queue_volume, ob.space_remaining))  # 정수 보장
                ob.load(transferable)
                lane.take_volume(transferable)
                self.buffer = max(self.buffer - transferable, 0.0)

    def _depart_outbound(self) -> List[dict]:
        """아웃바운드 트럭 출발 처리.
        타이머가 0이 되면 현재 적재된 화물로 출발 후 새 트럭 교체.
        (화물 탑재는 _progressive_load에서 이미 처리됨)
        """
        depart_info = []
        for k in range(self.num_lanes):
            ob = self.outbound_trucks[k]
            if ob.tick():
                fill = ob.fill_rate
                depart_info.append({
                    "lane_id":   k,
                    "loaded":    ob.loaded,
                    "fill_rate": fill,
                    "empty":     fill < 0.1,
                })
                # 새 아웃바운드 트럭 생성 — 출발 주기 랜덤 배분
                self.outbound_trucks[k] = OutboundTruck(
                    lane_id=k,
                    capacity=self.outbound_capacity,
                    departure_timer=self._sample_dispatch_timer(lane_id=k),
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

    def _apply_disruptions(self):
        """세 가지 돌발사항을 독립적으로 적용. disruption_log에 기록."""
        # 1) 도어 고장
        if self.disruption_door_failure and self.rng.random() < self.disruption_door_failure_prob:
            healthy = [d for d in self.doors if not d.is_failed]
            if healthy:
                door = healthy[int(self.rng.integers(0, len(healthy)))]
                duration = int(self.rng.integers(
                    self.disruption_door_failure_dur_min,
                    self.disruption_door_failure_dur_max + 1,
                ))
                interrupted = door.fail(duration)
                self.disruption_log.append({
                    "type": "door_failure",
                    "door_id": door.door_id,
                    "duration": duration,
                    "interrupted_truck": bool(interrupted),
                })
                self.metrics["disruption_door_failures"] += 1
                if interrupted:
                    self.waiting_trucks.insert(0, interrupted)
                    self.metrics["disruption_interrupted_trucks"] += 1

        # 2) 긴급 트럭 돌발 도착
        if self.disruption_rush_truck and self.rng.random() < self.disruption_rush_truck_prob:
            volumes = self.rng.uniform(
                self.disruption_rush_vol_min,
                self.disruption_rush_vol_max,
                size=self.num_lanes,
            ).round(1)
            shipments = {k: float(v) for k, v in enumerate(volumes)}
            rush = Truck(arrival_time=self.t, shipments=shipments, is_rush=True)
            self.waiting_trucks.insert(0, rush)  # 대기열 맨 앞 삽입
            self.disruption_log.append({
                "type": "rush_truck",
                "total_volume": float(rush.total_volume()),
            })
            self.metrics["disruption_rush_trucks"] += 1

        # 3) 아웃바운드 타이머 쇼크
        if self.disruption_timer_shock:
            for k, ob in enumerate(self.outbound_trucks):
                if self.rng.random() < self.disruption_timer_shock_prob:
                    shock_val = int(self.rng.integers(
                        self.disruption_timer_shock_min,
                        self.disruption_timer_shock_max + 1,
                    ))
                    if ob.departure_timer > shock_val:
                        ob.departure_timer = shock_val
                        self.disruption_log.append({
                            "type": "timer_shock",
                            "lane_id": k,
                            "forced_timer": shock_val,
                        })
                        self.metrics["disruption_timer_shocks"] += 1

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
