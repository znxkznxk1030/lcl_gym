import numpy as np
from typing import List


class BasePolicy:
    def act(self, obs: np.ndarray, num_doors: int) -> int:
        raise NotImplementedError

    def reset(self):
        pass


class RandomPolicy(BasePolicy):
    """Uniformly random action in {0, 1, ..., num_doors}."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        return int(self.rng.integers(0, num_doors + 1))


class FIFOPolicy(BasePolicy):
    """항상 도어 1을 요청. 대기 트럭이 있고 유휴 도어가 있을 때만."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        # obs[5]: idle_inbound_doors, obs[6]: waiting_trucks
        idle_doors = obs[5]
        waiting    = obs[6]
        if waiting > 0 and idle_doors > 0:
            return 1
        return 0


class GreedyPolicy(BasePolicy):
    """화물 매칭도가 가장 높은 도어 선택."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        # obs[5]: idle_doors, obs[6]: waiting, obs[7:]: door_matches
        idle_doors = obs[5]
        waiting    = obs[6]
        if waiting == 0 or idle_doors == 0:
            return 0

        door_matches = obs[7: 7 + num_doors]
        best_door = int(np.argmax(door_matches)) + 1
        if door_matches[best_door - 1] > 0:
            return best_door
        return 0


class HeuristicPriorityPolicy(BasePolicy):
    """
    긴급도 + 화물매칭도 - 혼잡도 종합 점수로 도어 선택.

    obs layout:
      0: lane_queue, 1: congestion,
      2: outbound_fill_rate, 3: outbound_departure_in,
      4: buffer_remaining, 5: idle_doors, 6: waiting_trucks,
      7..7+D-1: door_match_i
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        if waiting == 0 or idle_doors == 0:
            return 0

        # 아웃바운드 출발까지 남은 시간 기반 긴급도
        departure_in = max(obs[3], 0)
        urgency      = 1.0 / (departure_in + 1)
        congestion   = obs[1]

        door_matches = obs[7: 7 + num_doors]
        scores = [
            self.alpha * urgency
            + self.beta  * door_matches[i]
            - self.gamma * congestion
            for i in range(num_doors)
        ]
        best_door = int(np.argmax(scores)) + 1
        return best_door
