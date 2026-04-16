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
    """
    Request door 1 (lowest index free door equivalent).
    Represents earliest-first / static preference for door 1.
    Ties at the env level are broken by arrival order.
    """

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[4]  # idle_inbound_doors field
        waiting = obs[5]     # waiting_trucks field
        if waiting > 0 and idle_doors > 0:
            return 1  # always request door 1 (FIFO proxy)
        return 0


class GreedyPolicy(BasePolicy):
    """
    Request the door whose match score with this lane is highest.
    Uses door_match features from the observation.
    door_match_i is the normalised volume that would arrive for this lane
    if door i were assigned.
    """

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        idle_doors = obs[4]
        waiting = obs[5]
        if waiting == 0 or idle_doors == 0:
            return 0

        # door_match values start at index 6
        door_matches = obs[6: 6 + num_doors]
        best_door = int(np.argmax(door_matches)) + 1  # 1-indexed action
        if door_matches[best_door - 1] > 0:
            return best_door
        return 0


class HeuristicPriorityPolicy(BasePolicy):
    """
    Same scoring as the conflict resolver:
        score = urgency + volume_match - congestion
    Picks the door that maximises the agent's own score.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def act(self, obs: np.ndarray, num_doors: int) -> int:
        # obs layout:
        # 0: lane_queue, 1: congestion, 2: time_to_dispatch,
        # 3: buffer_remaining, 4: idle_doors, 5: waiting_trucks,
        # 6..6+num_doors-1: door_match_i
        idle_doors = obs[4]
        waiting = obs[5]
        if waiting == 0 or idle_doors == 0:
            return 0

        time_to_dispatch = max(obs[2], 0)
        urgency = 1.0 / (time_to_dispatch + 1)
        congestion = obs[1]

        door_matches = obs[6: 6 + num_doors]
        scores = [
            self.alpha * urgency
            + self.beta * door_matches[i]
            - self.gamma * congestion
            for i in range(num_doors)
        ]
        best_door = int(np.argmax(scores)) + 1
        return best_door
