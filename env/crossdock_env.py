"""
CrossDockEnv — Cross-Docking Multi-Agent RL Environment (MVP)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any

from .entities import Lane, Door, Truck
from .conflict_resolver import ConflictResolver


DEFAULT_CONFIG = {
    "num_lanes": 5,
    "num_inbound_doors": 3,
    "buffer_capacity": 100,
    "episode_length": 100,
    "truck_arrival_prob": 0.3,
    "max_door_processing": 10,
    # dispatch interval: each lane dispatches every N steps
    "dispatch_interval": 20,
    # reward weights
    "reward_alpha": 0.7,   # team weight
    "reward_beta": 0.3,    # local weight
    # conflict resolver weights
    "cr_alpha": 1.0,
    "cr_beta": 1.0,
    "cr_gamma": 1.0,
}


class CrossDockEnv:
    # ------------------------------------------------------------------
    # Construction / Reset
    # ------------------------------------------------------------------

    def __init__(self, config: dict = None, seed: int = 42):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.num_lanes: int = cfg["num_lanes"]
        self.num_inbound_doors: int = cfg["num_inbound_doors"]
        self.buffer_capacity: int = cfg["buffer_capacity"]
        self.episode_length: int = cfg["episode_length"]
        self.truck_arrival_prob: float = cfg["truck_arrival_prob"]
        self.max_door_processing: int = cfg["max_door_processing"]
        self.dispatch_interval: int = cfg["dispatch_interval"]
        self.reward_alpha: float = cfg["reward_alpha"]
        self.reward_beta: float = cfg["reward_beta"]

        self.resolver = ConflictResolver(
            alpha=cfg["cr_alpha"],
            beta=cfg["cr_beta"],
            gamma=cfg["cr_gamma"],
        )

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # observation size: 6 shared + num_doors door-match features
        self.obs_size: int = 6 + self.num_inbound_doors

        self.lanes: List[Lane] = []
        self.doors: List[Door] = []
        self.buffer: float = 0.0
        self.waiting_trucks: List[Truck] = []
        self.t: int = 0

        # episode-level metrics
        self.metrics: Dict[str, Any] = {}

        self.reset()

    def reset(self) -> List[np.ndarray]:
        self.rng = np.random.default_rng(self._seed)
        self.t = 0
        self.buffer = 0.0
        self.waiting_trucks = []

        self.lanes = [
            Lane(
                lane_id=k,
                dispatch_interval=self.dispatch_interval,
                dispatch_timer=self.dispatch_interval,
            )
            for k in range(self.num_lanes)
        ]
        self.doors = [Door(door_id=i) for i in range(self.num_inbound_doors)]

        self.metrics = {
            "total_throughput": 0.0,
            "buffer_overflow_count": 0,
            "late_dispatches": 0,
            "on_time_dispatches": 0,
            "door_busy_steps": 0,
            "total_steps": 0,
            "dwell_time_sum": 0.0,
            "dwell_count": 0,
        }

        return self.get_obs()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        assert len(actions) == self.num_lanes

        # 1. Generate truck arrivals
        new_trucks = self._generate_arrivals()

        # 2. Tick doors — release finished trucks back to buffer/lanes
        released_trucks = self._tick_doors()

        # 3. Move released truck shipments: truck → buffer → lane
        overflow = self._process_released(released_trucks)

        # 4. Add new trucks to waiting queue
        self.waiting_trucks.extend(new_trucks)

        # 5. Collect actions and resolve conflicts
        requests = {k: actions[k] for k in range(self.num_lanes)}
        allocation = self.resolver.resolve(
            requests, self.lanes, self.doors, self.waiting_trucks
        )

        # 6. Assign doors to winning lanes, dequeue truck
        self._assign_doors(allocation)

        # 7. Dispatch lanes whose timer expired
        dispatched_volumes = self._dispatch_lanes()

        # 8. Compute rewards
        rewards = self._compute_rewards(
            dispatched_volumes=dispatched_volumes,
            overflow=overflow,
        )

        # 9. Update metrics
        self._update_metrics(dispatched_volumes, overflow)

        self.t += 1
        done = self.t >= self.episode_length

        obs = self.get_obs()
        info = {"t": self.t, "metrics": self.metrics.copy()} if done else {"t": self.t}
        return obs, rewards, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self) -> List[np.ndarray]:
        idle_doors = sum(1 for d in self.doors if not d.is_busy)
        buffer_remaining = max(self.buffer_capacity - self.buffer, 0.0)
        waiting = len(self.waiting_trucks)

        obs_list = []
        for lane in self.lanes:
            door_matches = np.zeros(self.num_inbound_doors, dtype=np.float32)
            for i, door in enumerate(self.doors):
                if not door.is_busy and self.waiting_trucks:
                    # best match volume across all waiting trucks
                    best = max(
                        t.volume_for_lane(lane.lane_id) / (t.total_volume() + 1e-6)
                        for t in self.waiting_trucks
                    )
                    door_matches[i] = best

            obs = np.array(
                [
                    lane.queue_volume,          # 0
                    lane.congestion,            # 1
                    float(lane.dispatch_timer), # 2
                    float(buffer_remaining),    # 3
                    float(idle_doors),          # 4
                    float(waiting),             # 5
                ]
                + door_matches.tolist(),        # 6..6+num_doors-1
                dtype=np.float32,
            )
            obs_list.append(obs)
        return obs_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_arrivals(self) -> List[Truck]:
        trucks = []
        if self.rng.random() < self.truck_arrival_prob:
            # random shipment split across lanes
            volumes = self.rng.integers(1, 10, size=self.num_lanes)
            shipments = {k: int(volumes[k]) for k in range(self.num_lanes)}
            trucks.append(Truck(arrival_time=self.t, shipments=shipments))
        return trucks

    def _tick_doors(self) -> List[Truck]:
        released = []
        for door in self.doors:
            truck = door.tick()
            if truck is not None:
                released.append(truck)
        return released

    def _process_released(self, trucks: List[Truck]) -> int:
        overflow = 0
        for truck in trucks:
            for lane_id, volume in truck.shipments.items():
                space = self.buffer_capacity - self.buffer
                if space <= 0:
                    overflow += 1
                    continue
                actual = min(volume, space)
                self.buffer += actual
                self.lanes[lane_id].add_volume(actual)
                # dwell time: steps since truck arrived until now
                dwell = self.t - truck.arrival_time
                self.metrics["dwell_time_sum"] += dwell
                self.metrics["dwell_count"] += 1
        return overflow

    def _assign_doors(self, allocation: Dict[int, int]):
        """allocation: {door_idx: lane_id}"""
        for door_idx, lane_id in allocation.items():
            if not self.waiting_trucks:
                break
            truck = self.waiting_trucks.pop(0)
            processing_time = int(
                self.rng.integers(1, self.max_door_processing + 1)
            )
            self.doors[door_idx].assign(truck, lane_id, processing_time)

    def _dispatch_lanes(self) -> List[float]:
        dispatched = []
        for lane in self.lanes:
            vol = lane.tick_dispatch()
            dispatched.append(vol)
            if vol > 0:
                self.buffer = max(self.buffer - vol, 0.0)
        return dispatched

    def _compute_rewards(
        self,
        dispatched_volumes: List[float],
        overflow: int,
    ) -> List[float]:
        throughput = sum(dispatched_volumes)

        # Late dispatch: any lane with 0 volume at dispatch time is "late"
        late = sum(
            1
            for i, vol in enumerate(dispatched_volumes)
            if vol == 0 and self.lanes[i].dispatch_timer == self.dispatch_interval
        )

        r_team = throughput - 0.5 * overflow - 1.0 * late

        rewards = []
        for i, lane in enumerate(self.lanes):
            correct_inflow = dispatched_volumes[i]
            r_local = correct_inflow - 0.1 * lane.congestion - 0.1 * 0.0
            r_final = self.reward_alpha * r_team + self.reward_beta * r_local
            rewards.append(float(r_final))
        return rewards

    def _update_metrics(self, dispatched_volumes: List[float], overflow: int):
        self.metrics["total_throughput"] += sum(dispatched_volumes)
        self.metrics["buffer_overflow_count"] += overflow
        self.metrics["door_busy_steps"] += sum(
            1 for d in self.doors if d.is_busy
        )
        self.metrics["total_steps"] += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def door_utilization(self) -> float:
        steps = self.metrics["total_steps"]
        if steps == 0:
            return 0.0
        return self.metrics["door_busy_steps"] / (steps * self.num_inbound_doors)

    @property
    def avg_dwell_time(self) -> float:
        if self.metrics["dwell_count"] == 0:
            return 0.0
        return self.metrics["dwell_time_sum"] / self.metrics["dwell_count"]
