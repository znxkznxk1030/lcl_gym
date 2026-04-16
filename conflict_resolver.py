from typing import Dict, List, Optional
from entities import Lane, Door, Truck


class ConflictResolver:
    """
    Resolves conflicts when multiple agents request the same inbound door.

    Score function:
        score(k, i) = alpha * urgency_k + beta * volume_match(k, i) - gamma * congestion_k

    urgency_k    = 1 / (time_to_dispatch + 1)   (higher when deadline is near)
    volume_match = truck.volume_for_lane(k) / (truck.total_volume() + 1e-6)
    congestion_k = lane.congestion
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score(
        self,
        lane: Lane,
        truck: Optional[Truck],
        time_to_dispatch: int,
    ) -> float:
        urgency = 1.0 / (time_to_dispatch + 1)
        if truck is not None:
            volume_match = truck.volume_for_lane(lane.lane_id) / (
                truck.total_volume() + 1e-6
            )
        else:
            volume_match = 0.0
        congestion = lane.congestion
        return (
            self.alpha * urgency
            + self.beta * volume_match
            - self.gamma * congestion
        )

    def resolve(
        self,
        requests: Dict[int, int],   # {lane_id: door_id}  (0 = no request)
        lanes: List[Lane],
        doors: List[Door],
        waiting_trucks: List[Truck],
    ) -> Dict[int, int]:
        """
        Returns allocation: {door_id: lane_id}
        Only free doors and non-zero requests are considered.
        """
        # Group requests by door_id, excluding "no request" (0)
        door_requests: Dict[int, List[int]] = {}  # door_id -> [lane_ids]
        for lane_id, door_id in requests.items():
            if door_id == 0:
                continue
            real_door_idx = door_id - 1  # action i refers to door index i-1
            if real_door_idx < 0 or real_door_idx >= len(doors):
                continue
            if doors[real_door_idx].is_busy:
                continue
            door_requests.setdefault(real_door_idx, []).append(lane_id)

        allocation: Dict[int, int] = {}  # door_idx -> lane_id
        lane_map = {lane.lane_id: lane for lane in lanes}

        # Pick the truck with the most remaining time (FIFO proxy) if available
        oldest_truck = waiting_trucks[0] if waiting_trucks else None

        for door_idx, competing_lanes in door_requests.items():
            if len(competing_lanes) == 1:
                allocation[door_idx] = competing_lanes[0]
            else:
                best_lane = max(
                    competing_lanes,
                    key=lambda lid: self.score(
                        lane_map[lid],
                        oldest_truck,
                        lane_map[lid].dispatch_timer,
                    ),
                )
                allocation[door_idx] = best_lane

        return allocation
