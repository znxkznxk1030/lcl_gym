from __future__ import annotations

from dataclasses import dataclass

from .models import LaneAction


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    uses_centralized_state: bool
    uses_action_masks: bool


def baseline_catalog() -> dict[str, BaselineSpec]:
    return {
        "greedy": BaselineSpec(
            name="greedy",
            description="Rule-based decentralized heuristic using partial observations.",
            uses_centralized_state=False,
            uses_action_masks=True,
        ),
        "ippo": BaselineSpec(
            name="ippo",
            description="Independent PPO with per-agent observations and per-agent masks.",
            uses_centralized_state=False,
            uses_action_masks=True,
        ),
        "mappo": BaselineSpec(
            name="mappo",
            description="MAPPO-style centralized critic using per-agent observations plus global state.",
            uses_centralized_state=True,
            uses_action_masks=True,
        ),
        "centralized_ppo": BaselineSpec(
            name="centralized_ppo",
            description="Single centralized policy over concatenated agent observations and global state.",
            uses_centralized_state=True,
            uses_action_masks=True,
        ),
    }


class GreedyHeuristicPolicy:
    def __init__(self, dispatch_utilization_threshold: float = 0.55, deadline_trigger: int = 3) -> None:
        self.dispatch_utilization_threshold = dispatch_utilization_threshold
        self.deadline_trigger = deadline_trigger

    def act(self, observations: dict[str, dict]) -> dict[str, LaneAction]:
        actions: dict[str, LaneAction] = {}
        for agent_id, observation in observations.items():
            visible = observation["visible_shipments"]
            local_state = observation["local_state"]
            masks = observation["action_mask"]
            accept_index = len(masks["accept_mask"]) - 1
            if visible:
                best_idx = min(
                    range(len(visible)),
                    key=lambda idx: (visible[idx]["slack"], -visible[idx]["urgency"], -visible[idx]["volume"]),
                )
                if masks["accept_mask"][best_idx]:
                    accept_index = best_idx
            load_ratio = local_state["queued_volume"] / max(1.0, local_state["dispatch_capacity"])
            most_urgent = min((shipment["slack"] for shipment in visible), default=999)
            request_dispatch = (
                masks["dispatch_mask"][1] == 1
                and (
                    load_ratio >= self.dispatch_utilization_threshold
                    or local_state["overdue_shipments"] > 0
                    or local_state["queued_shipments"] >= 2
                    or most_urgent <= self.deadline_trigger
                )
            )
            actions[agent_id] = LaneAction(accept_index=accept_index, request_dispatch=request_dispatch)
        return actions
