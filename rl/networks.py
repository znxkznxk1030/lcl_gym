"""
networks.py — numpy 기반 2층 MLP (DQN용)

인터페이스:
    forward(obs)               → Q값 벡터
    update(obs, actions, targets) → TD loss (float)
    copy_weights_from(other)   → target network 동기화
    save(path) / load(path)    → 체크포인트

나중에 이 파일만 TorchMLP로 교체하면
rl_policy.py / train_rl.py는 수정 불필요.
"""
from __future__ import annotations

import numpy as np
from copy import deepcopy


class NumpyMLP:
    """
    2층 완전 연결 신경망 (ReLU + linear output)
    구조: obs_size → hidden → n_actions

    파라미터:
        obs_size  : 입력 차원 (기본 9)
        hidden    : 은닉층 크기 (기본 64)
        n_actions : 출력 차원 = 행동 수 (기본 4)
        lr        : 학습률
        seed      : 재현성
    """

    def __init__(
        self,
        obs_size: int = 9,
        hidden: int = 64,
        n_actions: int = 4,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        self.obs_size = obs_size
        self.hidden = hidden
        self.n_actions = n_actions
        self.lr = lr

        rng = np.random.default_rng(seed)

        # He 초기화
        self.W1 = rng.standard_normal((obs_size, hidden)) * np.sqrt(2.0 / obs_size)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, n_actions)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_actions)

        # Adam 모멘텀
        self._init_adam()
        self.t = 0  # Adam 스텝 카운터

    def _init_adam(self):
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (batch, obs_size) 또는 (obs_size,)
        반환: (batch, n_actions) 또는 (n_actions,) Q값
        """
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]           # (1, obs_size)

        z1 = obs @ self.W1 + self.b1           # (batch, hidden)
        a1 = np.maximum(0.0, z1)               # ReLU
        q  = a1 @ self.W2 + self.b2            # (batch, n_actions)

        return q[0] if single else q

    # ------------------------------------------------------------------
    # Update (TD 역전파)
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,      # (batch, obs_size)
        actions: np.ndarray,  # (batch,) int
        targets: np.ndarray,  # (batch,) float  — TD target
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """
        선택한 action에 대해서만 MSE 역전파.
        나머지 action의 gradient는 0 (stop-gradient trick).
        반환: 평균 TD loss
        """
        batch = obs.shape[0]

        # ── forward ───────────────────────────────────────────────────
        z1 = obs @ self.W1 + self.b1           # (batch, hidden)
        a1 = np.maximum(0.0, z1)               # ReLU
        q  = a1 @ self.W2 + self.b2            # (batch, n_actions)

        # ── TD error ──────────────────────────────────────────────────
        q_pred = q[np.arange(batch), actions]  # (batch,)
        delta  = targets - q_pred              # (batch,)
        loss   = float(np.mean(delta ** 2))

        # ── backward ──────────────────────────────────────────────────
        # dL/dq : action 위치만 -2*delta/batch, 나머지 0
        dq = np.zeros_like(q)                  # (batch, n_actions)
        dq[np.arange(batch), actions] = -2.0 * delta / batch

        dW2 = a1.T @ dq                        # (hidden, n_actions)
        db2 = dq.sum(axis=0)                   # (n_actions,)

        da1 = dq @ self.W2.T                   # (batch, hidden)
        dz1 = da1 * (z1 > 0)                   # ReLU gradient

        dW1 = obs.T @ dz1                      # (obs_size, hidden)
        db1 = dz1.sum(axis=0)                  # (hidden,)

        # ── Adam ──────────────────────────────────────────────────────
        self.t += 1
        t = self.t

        def adam_step(param, grad, m, v):
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        adam_step(self.W1, dW1, self.mW1, self.vW1)
        adam_step(self.b1, db1, self.mb1, self.vb1)
        adam_step(self.W2, dW2, self.mW2, self.vW2)
        adam_step(self.b2, db2, self.mb2, self.vb2)

        return loss

    # ------------------------------------------------------------------
    # Target network 동기화
    # ------------------------------------------------------------------

    def copy_weights_from(self, other: "NumpyMLP"):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self._init_adam()
        self.t = 0
