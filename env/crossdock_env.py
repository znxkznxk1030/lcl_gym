"""
CrossDockEnv — Cross-Docking MARL Environment (2-Stage)

Stage 1 (Inbound):  Truck → InboundDoor (1~10 steps) → SortingBuffer → SortingLane
Stage 2 (Outbound): SortingLane → OutboundDoor (dynamic dest, 12~28 steps) → Departure

변경 사항 (2-stage 리팩토링):
  - outbound_trucks (fixed per-lane) → outbound_doors (dynamic destination)
  - 아웃바운드 트럭 목적지를 매 출발마다 동적으로 재할당 (greedy by queue volume)
  - 관측 벡터에 idle_outbound_doors(obs[8]) 추가 → obs_size = 9 + num_inbound_doors
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from .entities import Lane, Door, Truck, OutboundTruck, OutboundDoor


DEFAULT_CONFIG = {
    "num_lanes": 5,
    "num_inbound_doors": 3,
    "num_outbound_doors": 8,           # Stage 2 아웃바운드 도크 수 (compound_trucks=False 시만 사용)
    "buffer_capacity": 1e9,            # 사실상 무한 버퍼 (제약 없음)
    "compound_trucks": False,          # True: 인바운드 트럭이 아웃바운드 트럭으로 전환 (동일 차량)
    "episode_length": 10000,            # 안전 상한 (실제 종료는 all_dispatched 조건)
    "truck_arrival_prob": 0.4,
    "max_door_processing": 10,
    "inbound_min_dest": 2,
    "inbound_max_dest": 3,
    "inbound_vol_min": 0.5,
    "inbound_vol_max": 5.0,
    "outbound_capacity": 15.0,
    # Stage 2 로딩 타임 (구 dispatch_interval 대체)
    "outbound_loading_time_min": 12,
    "outbound_loading_time_max": 28,
    # 하위 호환 — 사용 안 함 (레거시)
    "dispatch_interval": 20,
    "use_staggered_dispatch": True,
    "dispatch_interval_min": 12,
    "dispatch_interval_max": 28,
    "lane_dispatch_intervals": None,
    # Truck-Selection RL 모드
    "use_truck_selection": False,
    "top_k_trucks": 15,
    # 스케줄 기반 입고
    "use_scheduled_arrivals": True,
    "all_trucks_at_start": False,       # True: 모든 트럭을 t=0에 waiting_trucks에 일괄 배치
    "arrival_count_min": 50,
    "arrival_count_max": 70,
    "arrival_pattern": "clustered",
    "arrival_cluster_count": 4,
    "arrival_time_window": None,       # None = episode_length 사용. 정수 설정 시 트럭 도착을 해당 tick 내로 압축
    # reward weights
    "reward_alpha": 0.7,
    "reward_beta": 0.3,
    # conflict resolver weights
    "cr_alpha": 1.0,
    "cr_beta": 1.0,
    "cr_gamma": 1.0,
    # 돌발사항
    "enable_disruptions": False,
    "disruption_door_failure": False,
    "disruption_door_failure_prob": 0.015,
    "disruption_door_failure_duration_min": 5,
    "disruption_door_failure_duration_max": 12,
    "disruption_rush_truck": False,
    "disruption_rush_truck_prob": 0.025,
    "disruption_rush_volume_min": 6.0,
    "disruption_rush_volume_max": 12.0,
    "disruption_timer_shock": False,
    "disruption_timer_shock_prob": 0.025,
    "disruption_timer_shock_min": 2,
    "disruption_timer_shock_max": 4,
}


class CrossDockEnv:
    # ------------------------------------------------------------------
    # Construction / Reset
    # ------------------------------------------------------------------

    def __init__(self, config: dict = None, seed: int = 42):
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.num_lanes: int              = cfg["num_lanes"]
        self.num_inbound_doors: int      = cfg["num_inbound_doors"]
        self.num_outbound_doors: int     = cfg["num_outbound_doors"]
        self.buffer_capacity: float      = cfg["buffer_capacity"]
        self.episode_length: int         = cfg["episode_length"]
        self.truck_arrival_prob: float   = cfg["truck_arrival_prob"]
        self.max_door_processing: int    = cfg["max_door_processing"]
        self.inbound_min_dest: int       = cfg["inbound_min_dest"]
        self.inbound_max_dest: int       = cfg["inbound_max_dest"]
        self.inbound_vol_min: float      = cfg["inbound_vol_min"]
        self.inbound_vol_max: float      = cfg["inbound_vol_max"]
        self.outbound_capacity: float    = cfg["outbound_capacity"]
        self.outbound_loading_time_min: int = cfg["outbound_loading_time_min"]
        self.outbound_loading_time_max: int = cfg["outbound_loading_time_max"]

        # 하위 호환 속성 (MILP 등에서 접근)
        self.dispatch_interval_min: int  = cfg["outbound_loading_time_min"]
        self.dispatch_interval_max: int  = cfg["outbound_loading_time_max"]

        self.use_truck_selection: bool   = cfg["use_truck_selection"]
        self.top_k_trucks: int           = cfg["top_k_trucks"]
        self.reward_alpha: float         = cfg["reward_alpha"]
        self.reward_beta: float          = cfg["reward_beta"]
        self.use_scheduled_arrivals: bool = cfg["use_scheduled_arrivals"]
        self.all_trucks_at_start: bool   = cfg["all_trucks_at_start"]
        self.compound_trucks: bool       = cfg["compound_trucks"]
        self.arrival_count_min: int      = cfg["arrival_count_min"]
        self.arrival_count_max: int      = cfg["arrival_count_max"]
        self.arrival_pattern: str        = cfg["arrival_pattern"]
        self.arrival_cluster_count: int  = cfg["arrival_cluster_count"]
        self.arrival_time_window: Optional[int] = cfg.get("arrival_time_window", None)

        # 돌발사항
        self.enable_disruptions: bool             = cfg["enable_disruptions"]
        self.disruption_door_failure: bool        = cfg["disruption_door_failure"]
        self.disruption_door_failure_prob: float  = cfg["disruption_door_failure_prob"]
        self.disruption_door_failure_dur_min: int = cfg["disruption_door_failure_duration_min"]
        self.disruption_door_failure_dur_max: int = cfg["disruption_door_failure_duration_max"]
        self.disruption_rush_truck: bool          = cfg["disruption_rush_truck"]
        self.disruption_rush_truck_prob: float    = cfg["disruption_rush_truck_prob"]
        self.disruption_rush_vol_min: float       = cfg["disruption_rush_volume_min"]
        self.disruption_rush_vol_max: float       = cfg["disruption_rush_volume_max"]
        self.disruption_timer_shock: bool         = cfg["disruption_timer_shock"]
        self.disruption_timer_shock_prob: float   = cfg["disruption_timer_shock_prob"]
        self.disruption_timer_shock_min: int      = cfg["disruption_timer_shock_min"]
        self.disruption_timer_shock_max: int      = cfg["disruption_timer_shock_max"]

        self.disruption_log: List[dict] = []

        self._seed = seed
        # SeedSequence로 독립적인 두 스트림 생성
        # rng        : 환경 고유 난수 (아웃바운드 타이머, 도착 스케줄, 돌발)
        # assign_rng : 인바운드 도어 처리시간 전용 (정책 액션으로 트리거)
        _ss = np.random.SeedSequence(seed)
        _env_seed, _assign_seed = _ss.spawn(2)
        self.rng        = np.random.default_rng(_env_seed)
        self.assign_rng = np.random.default_rng(_assign_seed)

        # obs_size:
        #   레인 모드:       9 + num_inbound_doors  (기존 8+D에 idle_outbound_doors 추가)
        #   트럭선택 모드:   13 + 7 = 20 (변경 없음)
        if self.use_truck_selection:
            self.obs_size: int = self.num_lanes * 2 + 3 + 7
        else:
            self.obs_size: int = 9 + self.num_inbound_doors

        self.lanes: List[Lane] = []
        self.doors: List[Door] = []
        self.outbound_doors: List[OutboundDoor] = []
        # 하위 호환용 (MILP 코드가 outbound_trucks 를 직접 참조)
        self.outbound_trucks: List[OutboundTruck] = []
        self.buffer: float = 0.0
        self.waiting_trucks: List[Truck] = []
        self.arrival_schedule: List[Truck] = []
        self.t: int = 0
        self.metrics: Dict[str, Any] = {}

        self.reset()

    def reset(self) -> List[np.ndarray]:
        _ss = np.random.SeedSequence(self._seed)
        _env_seed, _assign_seed = _ss.spawn(2)
        self.rng        = np.random.default_rng(_env_seed)
        self.assign_rng = np.random.default_rng(_assign_seed)
        self.t = 0
        self.buffer = 0.0
        self.waiting_trucks = []
        if self.all_trucks_at_start:
            # 모든 트럭을 t=0에 waiting_trucks에 일괄 배치 (도착 스케줄 없음)
            trucks = self._build_arrival_schedule()
            for truck in trucks:
                truck.arrival_time = 0
            self.waiting_trucks = trucks
            self.arrival_schedule = []
        else:
            self.arrival_schedule = (
                self._build_arrival_schedule() if self.use_scheduled_arrivals else []
            )

        self.lanes = [Lane(lane_id=k) for k in range(self.num_lanes)]
        self.doors = [Door(door_id=i) for i in range(self.num_inbound_doors)]

        if self.compound_trucks:
            # Compound mode: 고정 수의 물리 도크, 초기에는 모두 유휴
            # 인바운드 처리 완료 트럭은 outbound_waiting 대기열에 줄을 섬
            self.outbound_doors = [
                OutboundDoor(door_id=i, capacity=self.outbound_capacity)
                for i in range(self.num_outbound_doors)
            ]
            self.outbound_waiting: List = []   # 아웃바운드 도크를 기다리는 트럭 토큰
        else:
            # Stage 2: 아웃바운드 도크 초기화 — 목적지를 round-robin으로 초기 할당 후 동적 전환
            self.outbound_doors = []
            for i in range(self.num_outbound_doors):
                od = OutboundDoor(door_id=i, capacity=self.outbound_capacity)
                dest = i % self.num_lanes
                init_timer = int(
                    self.rng.integers(1, self.outbound_loading_time_max + 1)
                )
                od.start_loading(dest, init_timer)
                self.outbound_doors.append(od)
            self.outbound_waiting = []

        # 하위 호환: MILP가 outbound_trucks[k].departure_timer 를 읽을 수 있도록 동기화
        self._sync_outbound_trucks_compat()

        self.metrics = {
            "total_throughput": 0.0,
            "total_fill_rate": 0.0,
            "outbound_departures": 0,
            "empty_departures": 0,
            "buffer_overflow_count": 0,
            "door_busy_steps": 0,
            "outbound_door_busy_steps": 0,
            "outbound_door_steps_total": 0,
            "total_steps": 0,
            "dwell_time_sum": 0.0,
            "dwell_count": 0,
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
        assert len(actions) == expected

        self.disruption_log = []
        if self.enable_disruptions:
            self._apply_disruptions()

        # 1. 인바운드 트럭 도착
        new_trucks = self._generate_arrivals()

        # 2. Stage 1: 인바운드 도어 틱 → 방출
        released_trucks = self._tick_doors()

        # 3. 방출 화물 → 버퍼 → 소팅 레인
        overflow = self._process_released(released_trucks)

        # 4. 대기열 추가
        self.waiting_trucks.extend(new_trucks)

        # 5. Stage 1 액션: 인바운드 트럭 → 유휴 인바운드 도어 배정
        self._assign_doors(actions)

        # 6. Stage 2: 유휴 아웃바운드 도크에 목적지 동적 할당 (action=2 레인 우선)
        self._assign_outbound_destinations(actions)

        # 6.5: action=2 레인이 있으면 빈 레인을 서비스하는 도크를 mid-trip 재배정
        self._reassign_empty_serving_docks(actions)

        # 7. Stage 2: 소팅 레인 → 아웃바운드 도크 점진 적재
        self._progressive_load()

        # 8. Stage 2: 아웃바운드 도크 틱 → 출발 처리
        depart_info = self._depart_outbound()

        # 9. 보상
        rewards = self._compute_rewards(depart_info=depart_info, overflow=overflow)

        # 10. 메트릭
        self._update_metrics(depart_info, overflow)

        # 하위 호환 동기화
        self._sync_outbound_trucks_compat()

        self.t += 1
        base_idle = (
            not self.waiting_trucks
            and not self.arrival_schedule
            and not any(d.is_busy for d in self.doors)
            and all(lane.queue_volume == 0 for lane in self.lanes)
        )
        if self.compound_trucks:
            # Compound mode: 레인 비고 + 현재 도킹 중인 트럭 출발까지 대기
            # (레인 빈 후 대기열 트럭은 더 이상 배정 안 되므로 자연히 수렴)
            all_dispatched = base_idle and not any(od.is_busy for od in self.outbound_doors)
        else:
            all_dispatched = base_idle and not any(od.is_busy for od in self.outbound_doors)
        done = all_dispatched or self.t >= self.episode_length
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
        """
        레인 에이전트 모드 obs (size = 9 + num_inbound_doors):
          0: lane_queue       1: congestion        2: outbound_fill_rate
          3: outbound_timer   4: buffer_remaining  5: idle_inbound_doors
          6: waiting_trucks   7: scheduled_trucks  8: idle_outbound_doors  ← NEW
          9..9+D-1: door_match_i
        """
        idle_inbound   = sum(1 for d in self.doors if not d.is_busy)
        idle_outbound  = sum(1 for od in self.outbound_doors if not od.is_busy)
        # buffer fill ratio [0, 2]: 0=empty, 1=at_capacity, >1=overflow zone
        buf_cap_eff    = float(self.buffer_capacity) if self.buffer_capacity < 1e8 else 500.0
        buffer_fill    = min(float(self.buffer) / max(buf_cap_eff, 1.0), 2.0)
        waiting        = len(self.waiting_trucks)
        scheduled      = len(self.arrival_schedule)

        # 목적지별로 현재 로딩 중인 아웃바운드 도크 조회
        serving: Dict[int, OutboundDoor] = {}
        for od in self.outbound_doors:
            if od.is_busy and od.assigned_dest >= 0:
                # 같은 목적지에 여러 도크가 있다면 타이머가 가장 짧은(임박한) 것 우선
                k = od.assigned_dest
                if k not in serving or od.loading_timer < serving[k].loading_timer:
                    serving[k] = od

        obs_list = []
        for k, lane in enumerate(self.lanes):
            od = serving.get(k)
            if od is not None:
                od_fill  = float(od.fill_rate)
                od_timer = float(od.loading_timer)
            else:
                od_fill  = 0.0
                if self.compound_trucks:
                    od_timer = 0.0  # 인바운드 단계: urgency 최대 (1/(0+1)=1.0)
                else:
                    od_timer = float(self.outbound_loading_time_max)

            # 인바운드 도어 매칭도
            door_matches = np.zeros(self.num_inbound_doors, dtype=np.float32)
            if self.waiting_trucks:
                if self.compound_trucks:
                    # 단일 목적지 라우팅: 0.5 기반 + 상대 볼륨 반영
                    # → 동일 볼륨 레인에서도 0.5 > 0 이 되어 Heuristic blocking 방지
                    max_q = max(la.queue_volume for la in self.lanes) + 1e-6
                    match_val = 0.5 + 0.5 * (1.0 - lane.queue_volume / max_q)
                    for i, door in enumerate(self.doors):
                        if not door.is_busy:
                            door_matches[i] = match_val
                else:
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
                    od_fill,
                    od_timer,
                    buffer_fill,       # [0, 2] fill ratio (was raw buffer amount)
                    float(idle_inbound),
                    float(waiting),
                    float(scheduled),
                    float(idle_outbound),
                ]
                + door_matches.tolist(),
                dtype=np.float32,
            )
            obs_list.append(obs)
        return obs_list

    def _get_obs_truck_selection(self) -> List[np.ndarray]:
        """
        트럭선택 모드 obs (size = num_lanes*2 + 3 + num_lanes + 2 = 20 for 5 lanes).
        모든 값을 [0, 1] 내외로 정규화하여 RL 수렴 안정성 확보.

        layout:
          [0:5]   timer_norm  = loading_timer / outbound_loading_time_max
          [5:10]  queue_norm  = queue_volume  / outbound_capacity
          [10]    buf_norm    = current_buffer / 500  (capped at 2)
          [11]    idle_norm   = idle_inbound_doors / num_inbound_doors
          [12]    nwait_norm  = waiting_count / arrival_count_max
          [13:18] vol_norm_k  = truck_vol_k / (inbound_vol_max * inbound_max_dest)
          [18]    total_norm  = truck_total / (inbound_vol_max * inbound_max_dest)
          [19]    is_rush
        """
        K  = self.top_k_trucks
        nL = self.num_lanes

        door_timers_by_dest = {
            od.assigned_dest: od.loading_timer
            for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
        }
        _tmax  = float(self.outbound_loading_time_max) + 1e-8
        _qmax  = float(self.outbound_capacity) + 1e-8
        _vmax  = float(self.inbound_vol_max * self.inbound_max_dest) + 1e-8
        _nmax  = float(max(self.arrival_count_max, 1))
        _dmax  = float(max(self.num_inbound_doors, 1))

        timers_norm = np.array(
            [door_timers_by_dest.get(k, self.outbound_loading_time_max) / _tmax
             for k in range(nL)],
            dtype=np.float32,
        )
        queues_norm = np.array(
            [lane.queue_volume / _qmax for lane in self.lanes], dtype=np.float32
        )
        buf_norm   = min(float(self.buffer) / 500.0, 2.0)
        idle_norm  = float(sum(1 for d in self.doors if not d.is_busy)) / _dmax
        nwait_norm = float(len(self.waiting_trucks)) / _nmax

        global_ctx = np.array(
            [*timers_norm, *queues_norm, buf_norm, idle_norm, nwait_norm],
            dtype=np.float32,
        )

        trucks = self.waiting_trucks[:K]
        obs_list = []
        for i in range(K):
            if i < len(trucks):
                t = trucks[i]
                vols_norm = np.array(
                    [float(t.shipments.get(k, 0)) / _vmax for k in range(nL)],
                    dtype=np.float32,
                )
                truck_feat = np.array(
                    [*vols_norm, float(t.total_volume()) / _vmax,
                     float(getattr(t, "is_rush", False))],
                    dtype=np.float32,
                )
            else:
                truck_feat = np.zeros(nL + 2, dtype=np.float32)
            obs_list.append(np.concatenate([global_ctx, truck_feat]))
        return obs_list

    # ------------------------------------------------------------------
    # Stage 2: 아웃바운드 도크 관리
    # ------------------------------------------------------------------

    def _assign_outbound_destinations(self, actions=None) -> None:
        """
        유휴 아웃바운드 도크에 소팅 레인 동적 할당.
        레인 모드에서 action=2인 레인은 타이머 긴급도 순으로 우선 배정.
        나머지 유휴 도크는 queue_volume 기준 greedy 배정.

        Compound mode: 여러 트럭이 같은 목적지를 동시에 로딩 가능.
                       화물 없는 레인에는 배정하지 않음 (트럭은 대기).
        Non-compound mode: 목적지당 하나의 도크만 로딩 (기존 동작).
        """
        idle_ods = [od for od in self.outbound_doors if not od.is_busy]
        if not idle_ods:
            return

        # 더 이상 유입될 화물이 없고 레인도 비었으면 재할당 중단 → 출발 처리로 수렴
        # (compound 모드에서는 outbound_waiting 트럭이 남아있으면 빈 차라도 배정해야 함)
        no_more_incoming = (
            not self.waiting_trucks
            and not self.arrival_schedule
            and not any(d.is_busy for d in self.doors)
        )
        if no_more_incoming and all(lane.queue_volume == 0 for lane in self.lanes):
            # 레인이 모두 비었으면 새 트럭 배정 중단 (compound 포함)
            # → 현재 도킹 중인 트럭만 마저 로딩 후 종료
            return

        if self.compound_trucks:
            # Compound mode: 인바운드 완료 후에만 시작 (race condition 방지)
            if not no_more_incoming:
                return
            if not self.outbound_waiting:
                return
            # 최대 화물 레인에서 로딩 (인바운드 라우팅으로 만든 레인 분포를 활용)
            for od in idle_ods:
                if not self.outbound_waiting:
                    break
                max_vol = max(self.lanes[k].queue_volume for k in range(self.num_lanes))
                if max_vol == 0:
                    break
                self.outbound_waiting.pop(0)
                dest = max(range(self.num_lanes), key=lambda k: self.lanes[k].queue_volume)
                loading_time = int(
                    self.rng.integers(
                        self.outbound_loading_time_min, self.outbound_loading_time_max + 1
                    )
                )
                od.start_loading(dest, loading_time)
        else:
            # Non-compound mode: 목적지당 1개 도크 제한
            already_serving = {
                od.assigned_dest
                for od in self.outbound_doors
                if od.is_busy and od.assigned_dest >= 0
            }

            # action=2 레인: 타이머 긴급도(낮을수록 급함) 순 우선 배정
            if actions is not None and not self.use_truck_selection:
                door_timer_by_dest = {
                    od.assigned_dest: od.loading_timer
                    for od in self.outbound_doors
                    if od.is_busy and od.assigned_dest >= 0
                }
                priority_lanes = sorted(
                    [k for k, a in enumerate(actions)
                     if a == 2 and k not in already_serving],
                    key=lambda k: door_timer_by_dest.get(k, self.outbound_loading_time_max),
                )
            else:
                priority_lanes = []

            # 나머지 레인: queue_volume 내림차순 greedy
            greedy_lanes = sorted(
                [k for k in range(self.num_lanes)
                 if k not in already_serving and k not in priority_lanes],
                key=lambda k: -self.lanes[k].queue_volume,
            )
            ordered = priority_lanes + greedy_lanes

            for od in idle_ods:
                dest = None
                for k in ordered:
                    if k not in already_serving:
                        dest = k
                        break
                if dest is None:
                    break
                loading_time = int(
                    self.rng.integers(
                        self.outbound_loading_time_min, self.outbound_loading_time_max + 1
                    )
                )
                od.start_loading(dest, loading_time)
                already_serving.add(dest)

    def _reassign_empty_serving_docks(self, actions=None) -> None:
        """
        action=2 레인 중 화물이 있고 도크가 없는 레인이 요청할 때,
        현재 빈 레인(queue=0)을 서비스하고 있는 도크를 해당 레인으로 mid-trip 재배정.

        이 기능이 FIFO 대비 RL/Greedy의 핵심 우위: FIFO는 action=2가 없어서
        빈 레인 도크가 타이머만 소진하고 출발(empty departure)하는 반면,
        action=2를 사용하는 정책은 해당 도크를 화물 있는 레인으로 즉시 전환.
        """
        if actions is None or self.use_truck_selection:
            return

        # action=2이고 화물이 있지만 아직 도크가 없는 레인
        already_serving = {
            od.assigned_dest for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
        }
        priority_lanes = sorted(
            [k for k, a in enumerate(actions)
             if a == 2 and self.lanes[k].queue_volume > 0 and k not in already_serving],
            key=lambda k: -self.lanes[k].queue_volume,
        )
        if not priority_lanes:
            return

        # 빈 레인을 서비스 중인 도크 (화물 다 빠져나간 뒤 타이머만 남음)
        # od.loaded == 0 조건: 이미 로딩된 화물이 있으면 재배정하지 않음 (화물 소실 방지)
        empty_serving = [
            od for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
            and self.lanes[od.assigned_dest].queue_volume == 0
            and od.loaded == 0
        ]
        if not empty_serving:
            return

        for od in empty_serving:
            for k in priority_lanes:
                if k in already_serving:
                    continue
                loading_time = int(self.rng.integers(
                    self.outbound_loading_time_min, self.outbound_loading_time_max + 1
                ))
                already_serving.discard(od.assigned_dest)
                od.start_loading(k, loading_time)
                already_serving.add(k)
                break  # 이 도크 재배정 완료, 다음 빈-도크로

    def _progressive_load(self) -> None:
        """Stage 2: 소팅 레인 → 아웃바운드 도크 점진 적재."""
        for od in self.outbound_doors:
            if not od.is_busy or od.assigned_dest < 0:
                continue
            lane = self.lanes[od.assigned_dest]
            if od.space_remaining > 0 and lane.queue_volume > 0:
                transferable = int(min(lane.queue_volume, od.space_remaining))
                if transferable > 0:
                    od.add_load(transferable)
                    lane.take_volume(transferable)
                    self.buffer = max(self.buffer - transferable, 0.0)

    def _depart_outbound(self) -> List[Optional[dict]]:
        """Stage 2: 아웃바운드 도크 틱 → 출발 처리."""
        depart_info = []
        for od in self.outbound_doors:
            departed, info = od.tick()
            depart_info.append(info if departed else None)
        return depart_info

    # ------------------------------------------------------------------
    # Stage 1 helpers (인바운드)
    # ------------------------------------------------------------------

    def _sample_dispatch_timer(self, initial: bool = False, lane_id: int = 0) -> int:
        """하위 호환용 — 아웃바운드 로딩 타임 샘플링."""
        if initial:
            return int(self.rng.integers(1, self.outbound_loading_time_max + 1))
        return int(
            self.rng.integers(self.outbound_loading_time_min, self.outbound_loading_time_max + 1)
        )

    def _build_arrival_schedule(self) -> List[Truck]:
        n = int(self.rng.integers(self.arrival_count_min, self.arrival_count_max + 1))
        time_window = self.arrival_time_window if self.arrival_time_window else self.episode_length

        if self.arrival_pattern == "clustered":
            base = np.linspace(0.1, 0.9, self.arrival_cluster_count)
            jitter = self.rng.uniform(-0.05, 0.05, size=self.arrival_cluster_count)
            centers = np.clip(base + jitter, 0.05, 0.95) * time_window
            cluster_ids = self.rng.integers(0, self.arrival_cluster_count, size=n)
            spread = time_window * 0.08
            raw_times = centers[cluster_ids] + self.rng.normal(0, spread, size=n)
            arrival_times = sorted(int(np.clip(t, 0, time_window - 1)) for t in raw_times)
        else:
            arrival_times = sorted(
                int(t) for t in self.rng.integers(0, time_window, size=n)
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
        if self.use_scheduled_arrivals:
            trucks = []
            while self.arrival_schedule and self.arrival_schedule[0].arrival_time <= self.t:
                trucks.append(self.arrival_schedule.pop(0))
            return trucks

        if self.rng.random() < self.truck_arrival_prob:
            n_dest = int(self.rng.integers(self.inbound_min_dest, self.inbound_max_dest + 1))
            dest_lanes = self.rng.choice(
                self.num_lanes, size=min(n_dest, self.num_lanes), replace=False
            )
            volumes = self.rng.integers(
                max(1, int(self.inbound_vol_min)), int(self.inbound_vol_max) + 1,
                size=len(dest_lanes),
            )
            shipments = {int(k): int(v) for k, v in zip(dest_lanes, volumes)}
            return [Truck(arrival_time=self.t, shipments=shipments)]
        return []

    def _tick_doors(self) -> List[Truck]:
        return [truck for door in self.doors if (truck := door.tick()) is not None]

    def _process_released(self, trucks: List[Truck]) -> int:
        for truck in trucks:
            if self.compound_trucks and truck.routed_lane >= 0:
                vol = int(truck.total_volume())
                self.buffer += vol
                self.lanes[truck.routed_lane].add_volume(vol)
                dwell = self.t - truck.arrival_time
                self.metrics["dwell_time_sum"] += dwell
                self.metrics["dwell_count"]    += 1
            else:
                for lane_id, volume in truck.shipments.items():
                    vol = int(volume)
                    self.buffer += vol
                    self.lanes[lane_id].add_volume(vol)
                    dwell = self.t - truck.arrival_time
                    self.metrics["dwell_time_sum"] += dwell
                    self.metrics["dwell_count"]    += 1
            if self.compound_trucks:
                self.outbound_waiting.append(truck)

        # 버퍼 초과분 계산 (유한 buffer_capacity 설정 시에만 실제 overflow 발생)
        overflow = 0
        if self.buffer > self.buffer_capacity:
            overflow = int(self.buffer - self.buffer_capacity)
            self.buffer = float(self.buffer_capacity)
        return overflow

    def _assign_doors(self, actions: List[int]) -> None:
        if self.use_truck_selection:
            self._assign_doors_truck_selection(actions)
        else:
            self._assign_doors_lane(actions)

    def _assign_doors_lane(self, actions: List[int]) -> None:
        idle_doors = [d for d in self.doors if not d.is_busy and not d.is_failed]
        if not idle_doors or not self.waiting_trucks:
            return

        requesting = [k for k, a in enumerate(actions) if a == 1]
        if not requesting:
            return

        # 긴급도 = 해당 레인을 담당하는 아웃바운드 도크의 loading_timer (짧을수록 급함)
        door_timers_by_dest = {
            od.assigned_dest: od.loading_timer
            for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
        }
        requesting.sort(
            key=lambda k: door_timers_by_dest.get(k, self.outbound_loading_time_max)
        )

        # 긴급도 순으로 정렬된 레인마다 해당 레인 화물이 가장 많은 트럭을 선택
        used_indices: set = set()
        assignments = []  # (door, truck, lane_id)

        for door, lane_id in zip(idle_doors, requesting):
            available = [i for i in range(len(self.waiting_trucks)) if i not in used_indices]
            if not available:
                break
            best_idx = max(available,
                           key=lambda i: self.waiting_trucks[i].volume_for_lane(lane_id))
            used_indices.add(best_idx)
            assignments.append((door, self.waiting_trucks[best_idx], lane_id))

        for idx in sorted(used_indices, reverse=True):
            self.waiting_trucks.pop(idx)

        for door, truck, lane_id in assignments:
            processing_time = int(self.assign_rng.integers(1, self.max_door_processing + 1))
            door.assign(truck, lane_id, processing_time)

    def _assign_doors_truck_selection(self, actions: List[int]) -> None:
        idle_doors = [d for d in self.doors if not d.is_busy and not d.is_failed]
        if not idle_doors or not self.waiting_trucks:
            return

        n_avail  = len(self.waiting_trucks)
        selected = [i for i, a in enumerate(actions) if a == 1 and i < n_avail]
        if not selected:
            # 폴백: 어떤 트럭도 선택되지 않으면 FIFO 순서 유지
            selected = list(range(min(len(idle_doors), n_avail)))

        sel_set = set(selected)
        front = [self.waiting_trucks[i] for i in selected]
        rest  = [t for i, t in enumerate(self.waiting_trucks) if i not in sel_set]
        self.waiting_trucks = front + rest

        door_timers_by_dest = {
            od.assigned_dest: od.loading_timer
            for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
        }
        for door in idle_doors:
            if not self.waiting_trucks:
                break
            truck = self.waiting_trucks.pop(0)
            if truck.shipments:
                lane_id = min(
                    truck.shipments.keys(),
                    key=lambda k: door_timers_by_dest.get(k, self.outbound_loading_time_max),
                )
            else:
                lane_id = 0
            processing_time = int(self.assign_rng.integers(1, self.max_door_processing + 1))
            door.assign(truck, lane_id, processing_time)

    # ------------------------------------------------------------------
    # Rewards & Metrics
    # ------------------------------------------------------------------

    def _compute_rewards(
        self,
        depart_info: List[Optional[dict]],
        overflow: int,
    ) -> List[float]:
        r = -1.0 - float(overflow) * 0.3  # overflow → 추가 패널티
        return [r] * self.num_lanes

    def _apply_disruptions(self) -> None:
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
                    "type": "door_failure", "door_id": door.door_id, "duration": duration,
                    "interrupted_truck": bool(interrupted),
                })
                self.metrics["disruption_door_failures"] += 1
                if interrupted:
                    self.waiting_trucks.insert(0, interrupted)
                    self.metrics["disruption_interrupted_trucks"] += 1

        # 2) 긴급 트럭
        if self.disruption_rush_truck and self.rng.random() < self.disruption_rush_truck_prob:
            volumes = self.rng.uniform(
                self.disruption_rush_vol_min, self.disruption_rush_vol_max, size=self.num_lanes
            ).round(1)
            shipments = {k: float(v) for k, v in enumerate(volumes)}
            rush = Truck(arrival_time=self.t, shipments=shipments, is_rush=True)
            self.waiting_trucks.insert(0, rush)
            self.disruption_log.append({
                "type": "rush_truck", "total_volume": float(rush.total_volume()),
            })
            self.metrics["disruption_rush_trucks"] += 1

        # 3) 타이머 쇼크 — 아웃바운드 도크 로딩 타이머 강제 단축
        if self.disruption_timer_shock:
            for od in self.outbound_doors:
                if od.is_busy and self.rng.random() < self.disruption_timer_shock_prob:
                    shock_val = int(self.rng.integers(
                        self.disruption_timer_shock_min, self.disruption_timer_shock_max + 1
                    ))
                    if od.loading_timer > shock_val:
                        od.loading_timer = shock_val
                        self.disruption_log.append({
                            "type": "timer_shock", "outbound_door_id": od.door_id,
                            "dest": od.assigned_dest, "forced_timer": shock_val,
                        })
                        self.metrics["disruption_timer_shocks"] += 1

    def _update_metrics(self, depart_info: List[Optional[dict]], overflow: int) -> None:
        for d in depart_info:
            if d is not None:
                self.metrics["total_throughput"]    += d["loaded"]
                self.metrics["total_fill_rate"]     += d["fill_rate"]
                self.metrics["outbound_departures"] += 1
                if d["empty"]:
                    self.metrics["empty_departures"] += 1

        self.metrics["buffer_overflow_count"]    += overflow
        self.metrics["door_busy_steps"]          += sum(1 for d in self.doors if d.is_busy)
        self.metrics["outbound_door_busy_steps"] += sum(1 for od in self.outbound_doors if od.is_busy)
        self.metrics["outbound_door_steps_total"] += len(self.outbound_doors)
        self.metrics["total_steps"] += 1

    # ------------------------------------------------------------------
    # 하위 호환: outbound_trucks 동기화 (MILP 코드가 읽음)
    # ------------------------------------------------------------------

    def _sync_outbound_trucks_compat(self) -> None:
        """
        MILP solve_mip.py 가 env.outbound_trucks[k].departure_timer 를 읽으므로,
        목적지 k 를 담당하는 아웃바운드 도크의 loading_timer 를 OutboundTruck 에 동기화.
        """
        door_timer_by_dest = {
            od.assigned_dest: od.loading_timer
            for od in self.outbound_doors
            if od.is_busy and od.assigned_dest >= 0
        }
        self.outbound_trucks = [
            OutboundTruck(
                lane_id=k,
                capacity=self.outbound_capacity,
                departure_timer=door_timer_by_dest.get(k, self.outbound_loading_time_max),
                loaded=0.0,
            )
            for k in range(self.num_lanes)
        ]

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
    def outbound_door_utilization(self) -> float:
        steps = self.metrics["total_steps"]
        if steps == 0:
            return 0.0
        return self.metrics["outbound_door_busy_steps"] / (steps * self.num_outbound_doors)

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
