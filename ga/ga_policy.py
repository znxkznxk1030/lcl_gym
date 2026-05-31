"""
GAPolicy: GA로 최적화된 휴리스틱 확장 정책.

유전자 벡터 (7개):
  [w_urgency, w_match, w_congestion, w_buffer, w_waiting, threshold, outbound_timer_thresh]

행동 결정 (3-action):
  1) lane_queue > 0 AND outbound_timer < outbound_timer_thresh  →  action=2 (아웃바운드 우선)
  2) inbound_score > threshold                                  →  action=1 (인바운드 요청)
  3) 그 외                                                       →  action=0 (대기)

inbound_score:
  urgency    = 1 / (outbound_timer + 1)
  best_match = max(door_match_i)
  buf_fill   = obs[4]  (buffer fill ratio [0, 2], env에서 정규화)
  wait_norm  = min(waiting_trucks / 20, 1)

  score = w_urgency * urgency
        + w_match   * best_match
        - w_congestion * congestion
        + w_buffer  * buf_fill
        + w_waiting * wait_norm
"""
from __future__ import annotations

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.policies import BasePolicy


# ─────────────────────────────────────────────────────────────────
# 유전자 메타데이터
# ─────────────────────────────────────────────────────────────────

GENE_NAMES: list[str] = [
    "w_urgency",
    "w_match",
    "w_congestion",
    "w_buffer",
    "w_waiting",
    "threshold",
    "outbound_timer_thresh",
]
N_GENES: int = len(GENE_NAMES)

# 각 유전자의 (하한, 상한)
GENE_BOUNDS: np.ndarray = np.array(
    [
        [0.0,  5.0],   # w_urgency
        [0.0,  5.0],   # w_match
        [0.0,  5.0],   # w_congestion
        [-3.0, 3.0],   # w_buffer  (음수=버퍼 포화 시 인바운드 억제)
        [0.0,  3.0],   # w_waiting
        [-1.0, 2.0],   # threshold
        [0.0, 28.0],   # outbound_timer_thresh (raw ticks)
    ],
    dtype=np.float64,
)

# HeuristicPriorityPolicy 동작에 대응하는 기준 유전자
HEURISTIC_GENES: np.ndarray = np.array(
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.3, 10.0], dtype=np.float64
)


# ─────────────────────────────────────────────────────────────────
# 정책 클래스
# ─────────────────────────────────────────────────────────────────

class GAPolicy(BasePolicy):
    """
    Parameters
    ----------
    genes : array-like, shape (7,)
        [w_urgency, w_match, w_congestion, w_buffer, w_waiting,
         threshold, outbound_timer_thresh]
    """

    def __init__(self, genes: np.ndarray, buffer_capacity: float = 60.0):
        genes = np.asarray(genes, dtype=np.float64)
        assert genes.shape == (N_GENES,), f"genes 길이는 {N_GENES}이어야 합니다."
        self.genes = genes.copy()
        (
            self.w_urgency,
            self.w_match,
            self.w_congestion,
            self.w_buffer,
            self.w_waiting,
            self.threshold,
            self.outbound_timer_thresh,
        ) = self.genes

    # obs layout (2-stage 레인 에이전트, size = 9 + D):
    #   0: lane_queue     1: congestion       2: outbound_fill_rate
    #   3: outbound_timer 4: buffer_fill[0,2] 5: idle_inbound_doors
    #   6: waiting_trucks 7: scheduled_trucks 8: idle_outbound_doors
    #   9..9+D-1: door_match_i
    def act(self, obs: np.ndarray, num_doors: int) -> int:
        lane_queue = float(obs[0])
        fill_rate  = float(obs[2])   # 내 레인 도크 로딩률
        buf_fill   = float(obs[4])   # [0, 2] fill ratio
        timer      = float(obs[3])

        # 1) 화물 있지만 도크 없음(fill_rate=0) → 도크 요청/재배정 (action=2)
        #    outbound_timer_thresh 이하인 경우도 포함하여 GA가 조건을 fine-tune
        if lane_queue > 0 and (fill_rate == 0 or timer < self.outbound_timer_thresh):
            return 2

        # 2) 인바운드 요청 가능 여부 확인
        if obs[6] == 0 or obs[5] == 0:
            return 0

        urgency    = 1.0 / (max(timer, 0.0) + 1.0)
        congestion = float(obs[1])
        best_match = float(obs[9: 9 + num_doors].max()) if num_doors > 0 else 0.0
        wait_norm  = min(float(obs[6]) / 20.0, 1.0)

        score = (
            self.w_urgency      * urgency
            + self.w_match      * best_match
            - self.w_congestion * congestion
            + self.w_buffer     * buf_fill
            + self.w_waiting    * wait_norm
        )
        return 1 if score > self.threshold else 0

    def __repr__(self) -> str:
        gene_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(GENE_NAMES, self.genes))
        return f"GAPolicy({gene_str})"


# ─────────────────────────────────────────────────────────────────
# Truck-Selection 모드용 GA 정책
# genes (3개): [w_urgency, w_rush, threshold]
# ─────────────────────────────────────────────────────────────────

TRUCKSEL_GENE_NAMES: list[str] = ["w_urgency", "w_rush", "threshold"]
N_TRUCKSEL_GENES: int = len(TRUCKSEL_GENE_NAMES)

TRUCKSEL_GENE_BOUNDS: np.ndarray = np.array(
    [
        [0.0, 3.0],   # w_urgency  (urgency_score ∈ [0.5,1.0] → score ∈ [0,3])
        [0.0, 2.0],   # w_rush
        [0.3, 1.0],   # threshold  (urgency [0.5,1.0] 범위에서 선별)
    ],
    dtype=np.float64,
)

TRUCKSEL_HEURISTIC_GENES: np.ndarray = np.array([1.0, 1.0, 0.6], dtype=np.float64)


class TruckSelGAPolicy(BasePolicy):
    """
    Truck-Selection 모드 GA 정책.

    obs layout (size=20 for 5 lanes):
      [0:5]   outbound loading_timer per lane
      [13:18] this truck's cargo per lane
      [19]    is_rush

    score = w_urgency * Σ_k[(vol_k/total) * 1/(timer_k+1)]
           + w_rush * is_rush
    action = 1 if score > threshold else 0
    """

    def __init__(self, genes: np.ndarray):
        genes = np.asarray(genes, dtype=np.float64)
        assert genes.shape == (N_TRUCKSEL_GENES,)
        self.genes = genes.copy()
        self.w_urgency, self.w_rush, self.threshold = self.genes

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        timers     = obs[0:5]
        truck_vols = obs[13:18]
        is_rush    = float(obs[19])
        total = float(np.sum(truck_vols))
        urgency_score = 0.0
        if total > 0:
            urgency_score = sum(
                (truck_vols[k] / total) / (timers[k] + 1)
                for k in range(5)
                if truck_vols[k] > 0
            )
        score = self.w_urgency * urgency_score + self.w_rush * is_rush
        return 1 if score > self.threshold else 0
