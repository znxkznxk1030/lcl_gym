from __future__ import annotations

import argparse
import json

from .baselines import GreedyHeuristicPolicy
from .config import build_phase_config
from .env import LCLConsolidationEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="LCL MARL logistics coordination simulator demo")
    parser.add_argument("--phase", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print final metrics as JSON only.")
    args = parser.parse_args()

    config = build_phase_config(args.phase, seed=args.seed, horizon=args.horizon)
    env = LCLConsolidationEnv(config)
    policy = GreedyHeuristicPolicy()
    observations, _ = env.reset(seed=args.seed)
    steps = args.steps or args.horizon

    rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    for _ in range(steps):
        actions = policy.act(observations)
        observations, step_rewards, dones, _ = env.step(actions)
        for agent_id, value in step_rewards.items():
            rewards[agent_id] += value
        if dones["__all__"]:
            break

    payload = {
        "phase": config.phase_name,
        "cumulative_rewards": {k: round(v, 3) for k, v in rewards.items()},
        "metrics": env.get_metrics(),
        "baseline_support": {
            name: spec.__dict__
            for name, spec in env.get_baseline_support()["baselines"].items()
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Phase: {payload['phase']}")
    print("Cumulative rewards:")
    for agent_id, value in payload["cumulative_rewards"].items():
        print(f"  {agent_id:<18} {value:>8.3f}")
    print("Metrics:")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
