from __future__ import annotations

import random
from typing import Optional

from .arrivals import ArrivalGenerator
from .baselines import baseline_catalog
from .config import EnvironmentConfig, build_phase_config
from .conflict_resolution import ClaimResolution, resolve_claims, resolve_dispatch_requests, time_to_cutoff
from .metrics import MetricsTracker
from .models import DispatchRecord, LaneAction, LaneState, Shipment
from .observations import build_centralized_state, build_observations
from .reward import compute_rewards


class LCLConsolidationEnv:
    def __init__(self, config: Optional[EnvironmentConfig] = None) -> None:
        self.config = config or build_phase_config(3)
        if len(self.config.lanes) != self.config.num_agents:
            raise ValueError(f"Expected {self.config.num_agents} lanes, got {len(self.config.lanes)}")
        self.agent_ids = self.config.lane_ids()
        self._lane_config_map = {lane.agent_id: lane for lane in self.config.lanes}
        self._rng = random.Random(self.config.seed)
        self._arrival_generator = ArrivalGenerator(self.config.lanes, self.config.arrivals, self._rng)
        self.metrics = MetricsTracker()
        self.shipments: dict[str, Shipment] = {}
        self.lane_states: dict[str, LaneState] = {}
        self.staging_queue: list[str] = []
        self.staging_volume: float = 0.0
        self.current_step: int = 0
        self._visibility: dict[str, list[Optional[str]]] = {}
        self._action_masks: dict[str, dict] = {}
        self._last_observations: dict[str, dict] = {}
        self._done = False
        self.reset(seed=self.config.seed)

    def reset(self, seed: Optional[int] = None) -> tuple[dict[str, dict], dict[str, dict]]:
        if seed is not None:
            self._rng = random.Random(seed)
            self._arrival_generator = ArrivalGenerator(self.config.lanes, self.config.arrivals, self._rng)
        self.metrics = MetricsTracker()
        self.shipments = {}
        self.staging_queue = []
        self.staging_volume = 0.0
        self.current_step = 0
        self._done = False
        self.lane_states = {lane_id: LaneState(lane_id=lane_id) for lane_id in self.agent_ids}
        for lane_id in self.agent_ids:
            self.metrics.ensure_lane(lane_id)
        self._ingest_arrivals(self._arrival_generator.generate(self.current_step))
        observations = self._refresh_observations()
        infos = {
            agent_id: {
                "metrics": self.metrics.snapshot(),
                "centralized_state": self.get_centralized_state(),
            }
            for agent_id in self.agent_ids
        }
        return observations, infos

    def step(self, actions: dict[str, int | dict | LaneAction]) -> tuple[dict[str, dict], dict[str, float], dict[str, bool], dict[str, dict]]:
        if self._done:
            raise RuntimeError("Episode already completed. Call reset() before step().")

        normalized = {agent_id: self._normalize_action(agent_id, actions.get(agent_id)) for agent_id in self.agent_ids}
        step_events = {
            "overflow_shipments": 0,
            "new_missed_deadlines": 0,
            "total_dispatched_volume": 0.0,
            "lane_events": {
                agent_id: {
                    "accepted_count": 0,
                    "dispatch_granted": 0,
                    "dispatch_volume": 0.0,
                    "new_missed_deadlines": 0,
                    "invalid_actions": 0,
                    "waiting_shipments": 0,
                    "queued_volume": 0.0,
                }
                for agent_id in self.agent_ids
            },
        }

        claim_requests = self._extract_claim_requests(normalized, step_events)
        claim_resolution = resolve_claims(
            claims=claim_requests,
            shipments=self.shipments,
            lane_states=self.lane_states,
            lane_configs=self._lane_config_map,
            current_step=self.current_step,
            config=self.config.conflicts,
        )
        self._apply_claim_resolution(claim_resolution, step_events)

        approved, denied = self._resolve_dispatches(normalized)
        for lane_id in denied:
            step_events["lane_events"][lane_id]["invalid_actions"] += 1
            self.lane_states[lane_id].invalid_actions += 1
            self.metrics.ensure_lane(lane_id).invalid_actions += 1
        self._execute_dispatches(approved, step_events)

        step_events["new_missed_deadlines"] += self._mark_missed_deadlines(step_events)
        self.current_step += 1

        if self.current_step < self.config.horizon:
            step_events["overflow_shipments"] += self._ingest_arrivals(self._arrival_generator.generate(self.current_step))
        self.metrics.record_staging_utilization(self.staging_volume / max(1.0, self.config.staging_capacity))

        for agent_id in self.agent_ids:
            lane_events = step_events["lane_events"][agent_id]
            lane_events["queued_volume"] = round(self.lane_states[agent_id].queued_volume, 2)
            lane_events["waiting_shipments"] = len(self.lane_states[agent_id].queued_shipments)
        reward_breakdown = compute_rewards(self, step_events)

        self._done = self.current_step >= self.config.horizon
        observations = self._refresh_observations() if not self._done else self._terminal_observations()
        dones = {agent_id: self._done for agent_id in self.agent_ids}
        dones["__all__"] = self._done
        infos = self._build_infos(step_events, claim_resolution, approved, reward_breakdown.team_component)
        return observations, reward_breakdown.local_components, dones, infos

    def get_action_masks(self) -> dict[str, dict]:
        return self._action_masks

    def get_centralized_state(self) -> dict:
        return build_centralized_state(self)

    def get_metrics(self) -> dict:
        return self.metrics.snapshot()

    def get_baseline_support(self) -> dict:
        return {
            "baselines": baseline_catalog(),
            "joint_action_size": (self.config.observations.top_k_shipments + 1) * 2,
            "supports_centralized_state": True,
        }

    def encode_action(self, action: LaneAction) -> int:
        accept_size = self.config.observations.top_k_shipments + 1
        if action.accept_index < 0 or action.accept_index >= accept_size:
            raise ValueError(f"accept_index out of bounds: {action.accept_index}")
        return action.accept_index * 2 + int(action.request_dispatch)

    def decode_action(self, action_id: int) -> LaneAction:
        accept_size = self.config.observations.top_k_shipments + 1
        if action_id < 0 or action_id >= accept_size * 2:
            raise ValueError(f"action_id out of bounds: {action_id}")
        accept_index, dispatch_flag = divmod(action_id, 2)
        return LaneAction(accept_index=accept_index, request_dispatch=bool(dispatch_flag))

    def inject_shipment(self, shipment: Shipment, lane_id: Optional[str] = None) -> bool:
        self.shipments[shipment.shipment_id] = shipment
        if lane_id is None:
            staged = self._stage_shipment(shipment)
            if not self._done:
                self._refresh_observations()
            return staged
        lane_state = self.lane_states[lane_id]
        lane_cfg = self._lane_config_map[lane_id]
        if lane_state.queued_volume + shipment.volume > lane_cfg.lane_buffer_capacity:
            return False
        shipment.accepted_by = lane_id
        shipment.accepted_step = self.current_step
        lane_state.queued_shipments.append(shipment.shipment_id)
        lane_state.queued_volume = round(lane_state.queued_volume + shipment.volume, 2)
        if not self._done:
            self._refresh_observations()
        return True

    def _terminal_observations(self) -> dict[str, dict]:
        terminal = {}
        for agent_id in self.agent_ids:
            terminal[agent_id] = {
                "agent_id": agent_id,
                "step": self.current_step,
                "phase": self.config.phase_name,
                "done": True,
            }
        return terminal

    def _refresh_observations(self) -> dict[str, dict]:
        observations, visibility, masks = build_observations(self)
        self._visibility = visibility
        self._action_masks = masks
        self._last_observations = observations
        return observations

    def _normalize_action(self, agent_id: str, action: int | dict | LaneAction | None) -> LaneAction:
        if action is None:
            return LaneAction(accept_index=self.config.observations.top_k_shipments, request_dispatch=False)
        if isinstance(action, LaneAction):
            return action
        if isinstance(action, int):
            return self.decode_action(action)
        if isinstance(action, dict):
            return LaneAction(
                accept_index=int(action.get("accept_index", self.config.observations.top_k_shipments)),
                request_dispatch=bool(action.get("request_dispatch", False)),
            )
        raise TypeError(f"Unsupported action type for {agent_id}: {type(action)!r}")

    def _extract_claim_requests(self, actions: dict[str, LaneAction], step_events: dict) -> dict[str, str]:
        claim_requests: dict[str, str] = {}
        none_index = self.config.observations.top_k_shipments
        for agent_id, action in actions.items():
            mask = self._action_masks[agent_id]["accept_mask"]
            if action.accept_index == none_index:
                continue
            if action.accept_index < 0 or action.accept_index >= len(mask) or mask[action.accept_index] == 0:
                step_events["lane_events"][agent_id]["invalid_actions"] += 1
                self.lane_states[agent_id].invalid_actions += 1
                self.metrics.ensure_lane(agent_id).invalid_actions += 1
                continue
            shipment_id = self._visibility[agent_id][action.accept_index]
            if shipment_id is not None:
                claim_requests[agent_id] = shipment_id
        return claim_requests

    def _apply_claim_resolution(self, resolution: ClaimResolution, step_events: dict) -> None:
        for lane_id, shipment_id in resolution.winners.items():
            shipment = self.shipments[shipment_id]
            if shipment_id not in self.staging_queue:
                continue
            self.staging_queue.remove(shipment_id)
            self.staging_volume = round(self.staging_volume - shipment.volume, 2)
            lane_state = self.lane_states[lane_id]
            lane_state.queued_shipments.append(shipment_id)
            lane_state.queued_volume = round(lane_state.queued_volume + shipment.volume, 2)
            lane_state.accepted_count += 1
            shipment.accepted_by = lane_id
            shipment.accepted_step = self.current_step
            step_events["lane_events"][lane_id]["accepted_count"] += 1
            self.metrics.ensure_lane(lane_id).accepted_shipments += 1
        for lane_id, shipment_ids in resolution.rejected.items():
            self.metrics.ensure_lane(lane_id).rejected_claims += len(shipment_ids)

    def _resolve_dispatches(self, actions: dict[str, LaneAction]) -> tuple[list[str], list[str]]:
        valid_requests: list[str] = []
        denied_requests: list[str] = []
        for agent_id, action in actions.items():
            if not action.request_dispatch:
                continue
            if self._action_masks[agent_id]["dispatch_mask"][1] == 0:
                denied_requests.append(agent_id)
                continue
            valid_requests.append(agent_id)
        if not valid_requests:
            return [], denied_requests
        approved, denied_due_to_slots = resolve_dispatch_requests(
            requested_lanes=valid_requests,
            lane_states=self.lane_states,
            lane_configs=self._lane_config_map,
            current_step=self.current_step,
            available_slots=self.config.dispatch_slots_per_step,
        )
        denied_requests.extend(denied_due_to_slots)
        return approved, denied_requests

    def _execute_dispatches(self, approved: list[str], step_events: dict) -> None:
        for lane_id in approved:
            lane_cfg = self._lane_config_map[lane_id]
            lane_state = self.lane_states[lane_id]
            queued_ids = list(lane_state.queued_shipments)
            shipped_ids: list[str] = []
            shipped_volume = 0.0
            prioritized = sorted(
                queued_ids,
                key=lambda shipment_id: (
                    self.shipments[shipment_id].deadline_step,
                    self.shipments[shipment_id].arrival_step,
                ),
            )
            for shipment_id in prioritized:
                shipment = self.shipments[shipment_id]
                if shipped_volume + shipment.volume > lane_cfg.dispatch_capacity and shipped_ids:
                    continue
                if shipped_volume + shipment.volume > lane_cfg.dispatch_capacity and not shipped_ids:
                    shipped_ids.append(shipment_id)
                    shipped_volume += shipment.volume
                    break
                shipped_ids.append(shipment_id)
                shipped_volume += shipment.volume

            if not shipped_ids:
                continue

            for shipment_id in shipped_ids:
                shipment = self.shipments[shipment_id]
                lane_state.queued_shipments.remove(shipment_id)
                lane_state.queued_volume = round(lane_state.queued_volume - shipment.volume, 2)
                shipment.dispatch_step = self.current_step
                shipment.dispatched_by = lane_id
                delay = max(0, self.current_step - shipment.arrival_step)
                self.metrics.completed_shipments += 1
                lane_metrics = self.metrics.ensure_lane(lane_id)
                lane_metrics.dispatched_shipments += 1
                lane_metrics.total_delay += delay
            utilization = shipped_volume / max(1.0, lane_cfg.dispatch_capacity)
            lane_state.dispatch_count += 1
            lane_state.dispatched_volume = round(lane_state.dispatched_volume + shipped_volume, 2)
            lane_state.last_dispatch_step = self.current_step
            self.metrics.dispatch_count += 1
            self.metrics.total_dispatched_volume = round(self.metrics.total_dispatched_volume + shipped_volume, 2)
            lane_metrics = self.metrics.ensure_lane(lane_id)
            lane_metrics.dispatched_volume = round(lane_metrics.dispatched_volume + shipped_volume, 2)
            lane_metrics.dispatch_count += 1
            record = DispatchRecord(
                lane_id=lane_id,
                step=self.current_step,
                shipment_ids=shipped_ids,
                total_volume=round(shipped_volume, 2),
                utilization=round(utilization, 4),
                cost=lane_cfg.dispatch_cost,
            )
            self.metrics.dispatch_records.append(record)
            step_events["lane_events"][lane_id]["dispatch_granted"] = 1
            step_events["lane_events"][lane_id]["dispatch_volume"] = round(shipped_volume, 2)
            step_events["total_dispatched_volume"] += shipped_volume

    def _mark_missed_deadlines(self, step_events: dict) -> int:
        new_missed = 0
        for shipment in self.shipments.values():
            if shipment.missed_deadline or shipment.dispatch_step is not None:
                continue
            if shipment.deadline_step <= self.current_step:
                shipment.missed_deadline = True
                new_missed += 1
                self.metrics.missed_deadlines += 1
                owner = shipment.accepted_by or shipment.anchor_lane
                if owner in self.lane_states:
                    self.lane_states[owner].missed_deadlines += 1
                    self.metrics.ensure_lane(owner).missed_deadlines += 1
                    step_events["lane_events"][owner]["new_missed_deadlines"] += 1
        return new_missed

    def _ingest_arrivals(self, shipments: list[Shipment]) -> int:
        overflowed = 0
        for shipment in shipments:
            self.shipments[shipment.shipment_id] = shipment
            self.metrics.total_arrivals += 1
            if not self._stage_shipment(shipment):
                overflowed += 1
        return overflowed

    def _stage_shipment(self, shipment: Shipment) -> bool:
        if self.staging_volume + shipment.volume > self.config.staging_capacity:
            shipment.overflowed = True
            self.metrics.overflow_shipments += 1
            for lane_id in shipment.compatible_lanes:
                self.metrics.ensure_lane(lane_id).overflow_losses += 1
            return False
        self.staging_queue.append(shipment.shipment_id)
        self.staging_volume = round(self.staging_volume + shipment.volume, 2)
        return True

    def _build_infos(self, step_events: dict, claim_resolution: ClaimResolution, approved: list[str], team_component: float) -> dict[str, dict]:
        metrics_snapshot = self.metrics.snapshot()
        infos: dict[str, dict] = {}
        for agent_id in self.agent_ids:
            infos[agent_id] = {
                "team_reward_component": round(team_component, 4),
                "lane_events": step_events["lane_events"][agent_id],
                "dispatch_approved": agent_id in approved,
                "rejected_claims": claim_resolution.rejected.get(agent_id, []),
                "metrics": metrics_snapshot,
                "centralized_state": self.get_centralized_state(),
            }
        return infos
