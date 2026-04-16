"""
replay_buffer.py — 경험 리플레이 버퍼

저장 단위: (obs, action, reward, next_obs, done) per agent
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """
    고정 크기 원형 버퍼 (circular buffer).

    파라미터:
        capacity : 최대 저장 개수 (기본 10,000)
        obs_size : 관측 벡터 크기
        seed     : 샘플링 재현성
    """

    def __init__(self, capacity: int = 10_000, obs_size: int = 9, seed: int = 0):
        self.capacity = capacity
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)

        self._obs      = np.zeros((capacity, obs_size), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int32)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.float32)

        self._ptr  = 0   # 다음 쓸 위치
        self._size = 0   # 현재 저장된 수

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        i = self._ptr
        self._obs[i]      = obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._next_obs[i] = next_obs
        self._dones[i]    = float(done)

        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        반환: (obs, actions, rewards, next_obs, dones) — 모두 numpy 배열
        """
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size
