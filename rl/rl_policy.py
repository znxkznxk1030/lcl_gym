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
_OBS_SCALE = np.array([
    50.0,   # 0: lane_queue        / 50
    1.0,    # 1: congestion        (이미 0~1)
    20.0,   # 2: dispatch_timer    / dispatch_interval
    100.0,  # 3: buffer_remaining  / buffer_capacity
    3.0,    # 4: idle_doors        / num_doors
    10.0,   # 5: waiting_trucks    / soft max
    1.0,    # 6: door_match_0      (이미 0~1)
    1.0,    # 7: door_match_1
    1.0,    # 8: door_match_2
], dtype=np.float32)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    return obs / (_OBS_SCALE[:len(obs)] + 1e-8)


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
        """epsilon-greedy 행동 선택. 반환: 0 ~ num_doors"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, num_doors + 1))

        obs_norm = normalize_obs(obs)
        q = self.net.forward(obs_norm)
        q = q[: num_doors + 1]
        return int(np.argmax(q))

    def reset(self):
        pass
