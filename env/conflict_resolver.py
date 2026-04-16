from typing import Dict, List, Optional
from .entities import Lane, Door, Truck, OutboundTruck


class ConflictResolver:
    """
    여러 에이전트가 같은 인바운드 도어를 요청할 때 우선순위 결정.

    score(k, i) = alpha * urgency_k + beta * volume_match(k, i) - gamma * congestion_k

    urgency_k    = 1 / (outbound_departure_timer_k + 1)
    volume_match = truck.volume_for_lane(k) / (truck.total_volume() + 1e-6)
    congestion_k = lane.congestion
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def score(
        self,
        lane: Lane,
        truck: Optional[Truck],
        departure_timer: int,
    ) -> float:
        urgency = 1.0 / (departure_timer + 1)
        if truck is not None:
            volume_match = truck.volume_for_lane(lane.lane_id) / (
                truck.total_volume() + 1e-6
            )
        else:
            volume_match = 0.0
        return (
            self.alpha * urgency
            + self.beta  * volume_match
            - self.gamma * lane.congestion
        )

    def resolve(
        self,
        requests: Dict[int, int],          # {lane_id: door_id}  (0 = no request)
        lanes: List[Lane],
        doors: List[Door],
        waiting_trucks: List[Truck],
        outbound_trucks: List[OutboundTruck],
    ) -> Dict[int, int]:
        """
        반환: {door_idx: lane_id}
        유휴 도어 + 비-0 요청만 처리.
        """
        door_requests: Dict[int, List[int]] = {}
        for lane_id, door_id in requests.items():
            if door_id == 0:
                continue
            real_door_idx = door_id - 1
            if real_door_idx < 0 or real_door_idx >= len(doors):
                continue
            if doors[real_door_idx].is_busy:
                continue
            door_requests.setdefault(real_door_idx, []).append(lane_id)

        allocation: Dict[int, int] = {}
        lane_map    = {lane.lane_id: lane for lane in lanes}
        ob_map      = {ob.lane_id: ob   for ob   in outbound_trucks}
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
                        ob_map[lid].departure_timer,
                    ),
                )
                allocation[door_idx] = best_lane

        return allocation
