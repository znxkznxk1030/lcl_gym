"""
rl_policy.py — DQN 기반 학습 가능한 정책

QLearningPolicy:
  - BasePolicy 인터페이스 상속 → 기존 run_simulation.py와 호환
  - epsilon-greedy 행동 선택
  - 5개 에이전트가 동일한 NumpyMLP 인스턴스를 공유 (Parameter Sharing)
  - 관측 정규화 내장
"""
from __future__ import annotations

import numpy as np
from env.policies import BasePolicy
from rl.networks import NumpyMLP


# 관측 벡터 정규화 상수 (crossdock_env DEFAULT_CONFIG 기준)
# obs layout (size = 8 + num_doors):
#   0: lane_queue, 1: congestion, 2: outbound_fill_rate,
#   3: outbound_departure_in, 4: buffer_remaining,
#   5: idle_doors, 6: waiting_trucks, 7: scheduled_trucks, 8..: door_matches
_OBS_SCALE_BASE = np.array([
    50.0,   # 0: lane_queue
    1.0,    # 1: congestion
    1.0,    # 2: outbound_fill_rate
    28.0,   # 3: outbound_departure_in
    150.0,  # 4: buffer_remaining
    3.0,    # 5: idle_doors
    100.0,  # 6: waiting_trucks  (8도어 환경 최대 ~150대)
    200.0,  # 7: scheduled_trucks
], dtype=np.float32)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    n_door_matches = len(obs) - len(_OBS_SCALE_BASE)
    scale = np.concatenate([
        _OBS_SCALE_BASE,
        np.ones(max(n_door_matches, 0), dtype=np.float32),  # door_match 이미 0~1
    ])
    return obs / (scale + 1e-8)


class QLearningPolicy(BasePolicy):
    """
    파라미터:
        net      : 공유 NumpyMLP 인스턴스
        epsilon  : 탐험 확률 (학습 중 외부에서 조절)
        rng      : 랜덤 제너레이터
    """

    def __init__(
        self,
        net: NumpyMLP,
        epsilon: float = 1.0,
        rng: np.random.Generator = None,
    ):
        self.net = net
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        """epsilon-greedy 행동 선택. 반환: 0 (skip) 또는 1 (request)"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))

        obs_norm = normalize_obs(obs)
        q = self.net.forward(obs_norm)
        return int(np.argmax(q[:2]))

    def reset(self):
        pass
