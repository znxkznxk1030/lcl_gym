from __future__ import annotations

import math
import random
from typing import Iterable

from .config import ArrivalConfig, LaneConfig
from .models import Shipment


def sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    l_value = math.exp(-lam)
    k = 0
    p_value = 1.0
    while p_value > l_value:
        k += 1
        p_value *= rng.random()
    return k - 1


class ArrivalGenerator:
    def __init__(self, lane_configs: Iterable[LaneConfig], arrival_config: ArrivalConfig, rng: random.Random) -> None:
        self._lane_configs = tuple(lane_configs)
        self._arrival_config = arrival_config
        self._rng = rng
        self._counter = 0

    def generate(self, step: int) -> list[Shipment]:
        shipments: list[Shipment] = []
        for lane in self._lane_configs:
            lam = lane.base_arrival_rate * lane.demand_pattern[step % len(lane.demand_pattern)]
            fixed_lam = lam * (1.0 - self._arrival_config.flexible_ratio)
            flex_lam = lam * self._arrival_config.flexible_ratio

            for _ in range(sample_poisson(self._rng, fixed_lam)):
                shipments.append(self._make_shipment(step, lane, compatible_lanes=(lane.agent_id,), fixed_lane=True))

            for _ in range(sample_poisson(self._rng, flex_lam)):
                shipments.append(self._make_flexible_shipment(step, lane))
        return shipments

    def _make_shipment(
        self,
        step: int,
        anchor_lane: LaneConfig,
        compatible_lanes: tuple[str, ...],
        fixed_lane: bool,
    ) -> Shipment:
        volume_low, volume_high = anchor_lane.volume_range
        base_volume = self._rng.uniform(volume_low, volume_high)
        volume = round(max(0.2, base_volume + self._rng.uniform(-1.0, 1.0) * self._arrival_config.volume_noise_scale), 2)
        deadline = max(
            step + 1,
            step + anchor_lane.deadline_mean + self._rng.randint(-anchor_lane.deadline_jitter, anchor_lane.deadline_jitter),
        )
        slack = max(1, deadline - step)
        urgency = round(anchor_lane.urgency_bias * (1.0 / slack), 4)
        shipment = Shipment(
            shipment_id=f"shp_{self._counter:05d}",
            arrival_step=step,
            compatible_lanes=compatible_lanes,
            volume=volume,
            deadline_step=deadline,
            urgency=urgency,
            anchor_lane=anchor_lane.agent_id,
            fixed_lane=fixed_lane,
            metadata={"destination": anchor_lane.destination},
        )
        self._counter += 1
        return shipment

    def _make_flexible_shipment(self, step: int, anchor_lane: LaneConfig) -> Shipment:
        lane_count = 2
        if len(self._arrival_config.flexible_lane_count_weights) > 1:
            if self._rng.random() > self._arrival_config.flexible_lane_count_weights[0]:
                lane_count = min(3, len(self._lane_configs))
        others = [lane.agent_id for lane in self._lane_configs if lane.agent_id != anchor_lane.agent_id]
        chosen = self._rng.sample(others, k=max(1, lane_count - 1))
        compatible = tuple(sorted((anchor_lane.agent_id, *chosen)))
        return self._make_shipment(step, anchor_lane, compatible_lanes=compatible, fixed_lane=False)
