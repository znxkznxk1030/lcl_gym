"""
GAPolicy: GA로 최적화된 휴리스틱 확장 정책.

유전자 벡터 (6개):
  [w_urgency, w_match, w_congestion, w_buffer, w_waiting, threshold]

점수 계산:
  urgency    = 1 / (departure_timer + 1)
  best_match = max(door_match_i)
  buf_norm   = buffer_remaining / buffer_capacity
  wait_norm  = min(waiting_trucks / 20, 1)

  score = w_urgency  * urgency
        + w_match    * best_match
        - w_congestion * congestion
        + w_buffer   * buf_norm
        + w_waiting  * wait_norm

  action = 1 if score > threshold else 0
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
]
N_GENES: int = len(GENE_NAMES)

# 각 유전자의 (하한, 상한)
GENE_BOUNDS: np.ndarray = np.array(
    [
        [0.0, 5.0],   # w_urgency
        [0.0, 5.0],   # w_match
        [0.0, 5.0],   # w_congestion
        [0.0, 3.0],   # w_buffer
        [0.0, 3.0],   # w_waiting
        [-1.0, 2.0],  # threshold
    ],
    dtype=np.float64,
)

# 기존 HeuristicPriorityPolicy와 동일한 동작을 내는 기준 유전자
HEURISTIC_GENES: np.ndarray = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.3], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
# 정책 클래스
# ─────────────────────────────────────────────────────────────────

class GAPolicy(BasePolicy):
    """
    Parameters
    ----------
    genes : array-like, shape (6,)
        [w_urgency, w_match, w_congestion, w_buffer, w_waiting, threshold]
    buffer_capacity : float
        환경 버퍼 최대 용량 (CBM). buf_norm 정규화에 사용.
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
        ) = self.genes
        self.buffer_capacity = float(buffer_capacity)

    # obs layout (2-stage 레인 에이전트, size = 9 + D):
    #   0: lane_queue     1: congestion       2: outbound_fill_rate
    #   3: outbound_timer 4: buffer_remaining 5: idle_inbound_doors
    #   6: waiting_trucks 7: scheduled_trucks 8: idle_outbound_doors
    #   9..9+D-1: door_match_i
    def act(self, obs: np.ndarray, num_doors: int) -> int:
        if obs[6] == 0 or obs[5] == 0:
            return 0

        urgency    = 1.0 / (max(float(obs[3]), 0.0) + 1.0)
        congestion = float(obs[1])
        best_match = float(obs[9: 9 + num_doors].max()) if num_doors > 0 else 0.0
        buf_norm   = min(float(obs[4]) / 500.0, 1.0)  # 버퍼 현재 적재량 정규화
        wait_norm  = min(float(obs[6]) / 20.0, 1.0)

        score = (
            self.w_urgency     * urgency
            + self.w_match     * best_match
            - self.w_congestion * congestion
            + self.w_buffer    * buf_norm
            + self.w_waiting   * wait_norm
        )
        return 1 if score > self.threshold else 0

    def __repr__(self) -> str:
        gene_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(GENE_NAMES, self.genes))
        return f"GAPolicy({gene_str})"
