import numpy as np
from typing import List


class BasePolicy:
    def act(self, obs: np.ndarray, num_doors: int) -> int:
        raise NotImplementedError

    def reset(self):
        pass


class ZeroPolicy(BasePolicy):
    """항상 action=0 (아무것도 하지 않음) — 하한 기준선."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        return 0


class RandomPolicy(BasePolicy):
    """50% 확률로 트럭 요청."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        return int(self.rng.integers(0, 2))


class FIFOPolicy(BasePolicy):
    """대기 트럭이 있고 유휴 도어가 있으면 항상 요청."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        return 1 if (waiting > 0 and idle_doors > 0) else 0


class GreedyPolicy(BasePolicy):
    """
    3-action 레인 모드:
      2 = 아웃바운드 도크 요청/재배정 (화물 있는데 내 레인에 도크가 로딩 중이지 않음)
      1 = 인바운드 트럭 요청 (대기 트럭 + 유휴 도어 있음)
      0 = 대기

    action=2 의 두 가지 효과:
      a) 유휴 도크가 있으면 내 레인에 우선 배정 (idle dock priority)
      b) 빈 레인을 서비스 중인 도크가 있으면 내 레인으로 mid-trip 재배정 (empty dock rescue)
    """

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        lane_queue = obs[0]
        fill_rate  = obs[2]   # 내 레인에 도크가 로딩 중이면 > 0

        # 화물이 있지만 도크가 활발히 로딩하지 않음 → 도크 요청/재배정
        if lane_queue > 0 and fill_rate == 0:
            return 2
        if waiting > 0 and idle_doors > 0:
            return 1
        return 0


# ─────────────────────────────────────────────────────────────────
# Truck-Selection 모드 전용 정책
#
# obs layout (size = num_lanes*2 + 3 + num_lanes + 2 = 20 for 5 lanes):
#   [0:5]   outbound loading_timer per lane  (낮을수록 임박)
#   [5:10]  queue_volume per lane
#   [10]    buffer_remaining
#   [11]    idle_inbound_doors
#   [12]    n_wait (waiting trucks count)
#   [13:18] this truck's cargo volume per lane
#   [18]    this truck's total volume
#   [19]    this truck's is_rush flag
# ─────────────────────────────────────────────────────────────────

class TruckSelFIFOPolicy(BasePolicy):
    """항상 1 반환 → FIFO 순서 그대로."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        return 1


class TruckSelGreedyPolicy(BasePolicy):
    """
    이 트럭이 서비스하는 레인 중 아웃바운드 도크가 임박한 레인이 있으면 우선 선택(1).
    obs[0:5] = timer_norm (정규화 [0,1]), 임박 기준: timer_norm < thresh (기본 0.5).
    """

    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        timers_norm = obs[0:5]
        truck_vols  = obs[13:18]
        for k in range(5):
            if truck_vols[k] > 0 and timers_norm[k] < self.thresh:
                return 1
        return 0


class TruckSelHeuristicPolicy(BasePolicy):
    """
    트럭이 서비스하는 레인의 긴급도를 화물 비율로 가중평균한 점수가
    임계값을 넘으면 선택(1).

    obs[0:5] = timer_norm (정규화 [0,1])
    urgency_k = 1 / (timer_norm_k + 1)  → [0.5, 1.0]
    score = Σ_k [ (vol_k / total_vol) * urgency_k ]
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        timers_norm = obs[0:5]
        truck_vols  = obs[13:18]
        is_rush     = obs[19]
        if is_rush:
            return 1
        total = float(np.sum(truck_vols))
        if total <= 0:
            return 0
        score = sum(
            (truck_vols[k] / total) / (timers_norm[k] + 1)
            for k in range(5)
            if truck_vols[k] > 0
        )
        return 1 if score > self.threshold else 0


class HeuristicPriorityPolicy(BasePolicy):
    """
    3-action 레인 모드:
      2 = 아웃바운드 긴급 (타이머 낮음 AND 레인에 화물 있음)
      1 = 인바운드 요청 (긴급도+매칭도-혼잡도 점수 ≥ threshold AND 버퍼 여유 있음)
      0 = 대기

    obs layout (2-stage, size = 9 + D):
      0: lane_queue,  1: congestion,       2: outbound_fill_rate,
      3: outbound_timer (loading_timer),   4: buffer_fill_ratio [0,2],
      5: idle_inbound_doors,  6: waiting_trucks,  7: scheduled_trucks,
      8: idle_outbound_doors,  9..9+D-1: door_match_i
    """

    def __init__(self, threshold: float = 0.3, buf_full_thresh: float = 1.5):
        self.threshold       = threshold
        self.buf_full_thresh = buf_full_thresh

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        lane_queue = obs[0]
        fill_rate  = obs[2]   # 내 레인 도크 로딩률 (0이면 도크 없음)
        buf_fill   = obs[4]   # buffer fill ratio [0, 2]

        # 화물 있지만 도크가 로딩하지 않음 → 도크 요청/재배정
        if lane_queue > 0 and fill_rate == 0:
            return 2

        if waiting == 0 or idle_doors == 0:
            return 0

        # 버퍼 포화 시 인바운드 억제
        if buf_fill > self.buf_full_thresh:
            return 0

        departure_in = max(obs[3], 0)
        urgency      = 1.0 / (departure_in + 1)
        congestion   = obs[1]
        door_matches = obs[9: 9 + num_doors]
        score = urgency + door_matches.max() - congestion
        return 1 if score > self.threshold else 0
