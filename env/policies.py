import numpy as np
from typing import List


class BasePolicy:
    def act(self, obs: np.ndarray, num_doors: int) -> int:
        raise NotImplementedError

    def reset(self):
        pass


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
    """아웃바운드가 곧 출발하거나 화물 매칭도가 높으면 요청."""

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        if waiting == 0 or idle_doors == 0:
            return 0
        door_matches = obs[8: 8 + num_doors]
        return 1 if door_matches.max() > 0 else 0


class HeuristicPriorityPolicy(BasePolicy):
    """
    긴급도 + 화물매칭도 - 혼잡도 종합 점수가 임계값 이상이면 요청.

    obs layout:
      0: lane_queue, 1: congestion,
      2: outbound_fill_rate, 3: outbound_departure_in,
      4: buffer_remaining, 5: idle_doors, 6: waiting_trucks,
      7: scheduled_trucks, 8..8+D-1: door_match_i
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[5]
        waiting    = obs[6]
        if waiting == 0 or idle_doors == 0:
            return 0

        departure_in = max(obs[3], 0)
        urgency      = 1.0 / (departure_in + 1)
        congestion   = obs[1]
        door_matches = obs[8: 8 + num_doors]
        score = urgency + door_matches.max() - congestion
        return 1 if score > self.threshold else 0
