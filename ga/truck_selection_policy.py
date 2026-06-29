"""
truck_selection_policy.py — GA 기반 Truck Selection 정책 (7-gene)

Lane Request 모드와 달리 "어떤 트럭을 먼저 처리할 것인가"를 결정한다.
use_truck_selection=True 환경과 함께 사용한다.

Gene 구조 (7개):
  [w_due, w_dest_match, w_congestion, w_buffer, w_queue_position, w_rush, threshold]

Truck Score:
  score = w_due          × weighted_urgency
        + w_dest_match   × destination_match
        − w_congestion   × expected_congestion
        + w_buffer       × buffer_availability
        − w_queue_position × queue_pressure
        + w_rush         × rush_flag
  action = 1 if score > threshold else 0

Obs layout (size=20, truck-selection mode):
  [0:5]   timer_norm  — outbound loading timer per lane (normalized)
  [5:10]  queue_norm  — lane queue volume per lane (normalized)
  [10]    buf_norm    — buffer fill level (0~2)
  [11]    idle_norm   — idle inbound doors (normalized)
  [12]    nwait_norm  — waiting truck count (normalized)
  [13:18] vol_norm    — this truck's cargo volume per lane (normalized)
  [18]    total_norm  — this truck's total volume (normalized)
  [19]    is_rush     — rush truck flag
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
    "w_due",
    "w_dest_match",
    "w_congestion",
    "w_buffer",
    "w_queue_position",
    "w_rush",
    "threshold",
]
N_GENES: int = len(GENE_NAMES)

GENE_BOUNDS: np.ndarray = np.array(
    [
        [0.0,  3.0],   # w_due
        [0.0,  3.0],   # w_dest_match
        [0.0,  2.0],   # w_congestion  (음수 기여 — 혼잡 시 억제)
        [0.0,  2.0],   # w_buffer
        [0.0,  2.0],   # w_queue_position (음수 기여 — 대기 줄 길수록 패널티)
        [0.0,  2.0],   # w_rush
        [-1.0, 2.0],   # threshold
    ],
    dtype=np.float64,
)

# 초기 시드 유전자 (Heuristic 수준의 합리적 기본값)
HEURISTIC_GENES: np.ndarray = np.array(
    [1.5, 1.0, 0.3, 0.5, 0.2, 1.0, 0.5], dtype=np.float64
)


# ─────────────────────────────────────────────────────────────────
# 정책 클래스
# ─────────────────────────────────────────────────────────────────

class TruckSelectionGAPolicy(BasePolicy):
    """
    Parameters
    ----------
    genes : array-like, shape (7,)
        [w_due, w_dest_match, w_congestion, w_buffer,
         w_queue_position, w_rush, threshold]
    """

    def __init__(self, genes: np.ndarray):
        genes = np.asarray(genes, dtype=np.float64)
        assert genes.shape == (N_GENES,), f"genes 길이는 {N_GENES}이어야 합니다."
        self.genes = genes.copy()
        (
            self.w_due,
            self.w_dest_match,
            self.w_congestion,
            self.w_buffer,
            self.w_queue_position,
            self.w_rush,
            self.threshold,
        ) = self.genes

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        timers     = obs[0:5]    # loading timer per lane (normalized)
        queues     = obs[5:10]   # queue volume per lane (normalized)
        buf_norm   = float(obs[10])
        nwait_norm = float(obs[12])
        truck_vols = obs[13:18]  # this truck's cargo per lane (normalized)
        total_vol  = float(obs[18])
        is_rush    = float(obs[19])

        if total_vol <= 0:
            return 0

        vol_ratios = truck_vols / (total_vol + 1e-8)  # 목적지별 화물 비율

        # 1. weighted_urgency: 내 화물이 향하는 레인의 긴급도를 화물 비율로 가중평균
        #    urgency_k = 1 / (timer_k + 1) ∈ [0.5, 1.0] (timer_norm ∈ [0,1])
        urgency_per_lane = 1.0 / (timers + 1.0)
        weighted_urgency = float(np.sum(vol_ratios * urgency_per_lane))

        # 2. destination_match: 내 화물이 향하는 레인의 수요(queue)를 화물 비율로 가중평균
        #    높을수록 이 트럭이 실제로 필요한 레인에 화물을 공급함
        destination_match = float(np.sum(vol_ratios * queues))

        # 3. expected_congestion: 전체 레인 평균 혼잡도
        expected_congestion = float(np.mean(queues))

        # 4. buffer_availability: 버퍼 여유 (buf_norm=0 → 여유 있음, 2 → 포화)
        buffer_availability = max(0.0, 1.0 - buf_norm / 2.0)

        # 5. queue_pressure: 대기 트럭이 많을수록 패널티 (선택의 긴급성 감소)
        queue_pressure = nwait_norm

        score = (
            self.w_due            * weighted_urgency
            + self.w_dest_match   * destination_match
            - self.w_congestion   * expected_congestion
            + self.w_buffer       * buffer_availability
            - self.w_queue_position * queue_pressure
            + self.w_rush         * is_rush
        )

        return 1 if score > self.threshold else 0

    def __repr__(self) -> str:
        gene_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(GENE_NAMES, self.genes))
        return f"TruckSelectionGAPolicy({gene_str})"
