"""
train_rl.py — IQL + Parameter Sharing DQN 학습 루프

실행 (프로젝트 루트에서):
    python rl/train_rl.py
    python rl/train_rl.py --episodes 2000 --lr 5e-4
    python rl/train_rl.py --no-share
"""
from __future__ import annotations

import os
import sys

# 프로젝트 루트를 sys.path에 추가 (어느 위치에서 실행해도 동작)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from rl.networks import NumpyMLP
from rl.replay_buffer import ReplayBuffer
from rl.rl_policy import QLearningPolicy, normalize_obs


# ---------------------------------------------------------------------------
# Reward Shaping
# ---------------------------------------------------------------------------

def shape_rewards(
    env_rewards: list,
    obs_list: list,
    next_obs_list: list,
    actions: list,
    num_doors: int,
    use_outbound_shaping: bool = True,
    use_inbound_shaping: bool = True,
) -> list:
    """
    3-action shaping (0=skip, 1=request_inbound, 2=boost_outbound):
    - use_outbound_shaping: needs_dock 상황에서 action=2 보너스/패널티
    - use_inbound_shaping:  can_inbound 상황에서 action=1 보너스/패널티
    """
    BUF_FULL = 1.5

    shaped = []
    for r, obs, action in zip(env_rewards, obs_list, actions):
        idle_doors = float(obs[5])
        waiting    = float(obs[6])
        lane_queue = float(obs[0])
        fill_rate  = float(obs[2])
        buf_fill   = float(obs[4])

        can_inbound  = idle_doors > 0 and waiting > 0
        needs_dock   = lane_queue > 0 and fill_rate == 0
        buf_stressed = buf_fill > BUF_FULL

        bonus = 0.0
        if use_outbound_shaping and needs_dock:
            bonus += 0.8 if action == 2 else -0.5

        if use_inbound_shaping and can_inbound:
            if buf_stressed:
                if action == 1:
                    bonus -= 1.0
            else:
                bonus += 0.4 if action == 1 else -0.3

        shaped.append(r + bonus)
    return shaped


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train(
    num_episodes: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden: int = 64,
    buffer_capacity: int = 10_000,
    warmup: int = 500,
    target_sync_interval: int = 50,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    shared_weights: bool = True,
    seed: int = 42,
    log_interval: int = 100,
    save_dir: str = "checkpoints",
    env_config: dict = None,
    use_outbound_shaping: bool = True,
    use_inbound_shaping: bool = True,
) -> dict:

    os.makedirs(save_dir, exist_ok=True)

    config = {**DEFAULT_CONFIG, **(env_config or {})}
    env = CrossDockEnv(config, seed=seed)
    truck_selection_mode = env.use_truck_selection
    n_agents  = env.top_k_trucks if truck_selection_mode else env.num_lanes
    obs_size  = env.obs_size
    n_actions = 2 if truck_selection_mode else 3  # truck-sel: 0/1; lane: 0/1/2

    # ── 네트워크 초기화 ───────────────────────────────────────────────
    if shared_weights:
        shared_net = NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed)
        target_net = NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed)
        target_net.copy_weights_from(shared_net)
        nets = [shared_net] * n_agents
    else:
        nets = [NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed + k)
                for k in range(n_agents)]
        target_net = NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed)
        target_net.copy_weights_from(nets[0])

    # ── 버퍼 & 정책 초기화 ───────────────────────────────────────────
    buffer = ReplayBuffer(buffer_capacity, obs_size, seed=seed)
    epsilon = epsilon_start
    agents = [
        QLearningPolicy(net=nets[k], epsilon=epsilon, rng=np.random.default_rng(seed + k))
        for k in range(n_agents)
    ]

    log_rewards = []
    log_ticks   = []
    log_loss    = []

    print(f"학습 시작 — episodes={num_episodes}, shared={shared_weights}, "
          f"lr={lr}, gamma={gamma}")
    print(f"{'Episode':>8} {'AvgReward':>12} {'AvgTicks':>10} "
          f"{'TDLoss':>10} {'Epsilon':>9}")
    print("-" * 55)

    for episode in range(num_episodes):
        env._seed = seed + episode
        obs_list = env.reset()
        ep_reward = 0.0
        ep_losses = []

        for step in range(env.episode_length):
            actions = [
                agents[k].act(obs_list[k], env.num_inbound_doors)
                for k in range(n_agents)
            ]
            next_obs_list, env_rewards, done, info = env.step(actions)

            if truck_selection_mode:
                team_reward = float(np.mean(env_rewards))
                rewards = [team_reward] * n_agents
            else:
                rewards = shape_rewards(
                    env_rewards, obs_list, next_obs_list, actions, env.num_inbound_doors,
                    use_outbound_shaping=use_outbound_shaping,
                    use_inbound_shaping=use_inbound_shaping,
                )

            for k in range(n_agents):
                buffer.push(
                    normalize_obs(obs_list[k]),
                    actions[k],
                    rewards[k],
                    normalize_obs(next_obs_list[k]),
                    done,
                )

            ep_reward += sum(rewards)

            if len(buffer) >= warmup:
                obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
                q_next     = target_net.forward(next_obs_b)
                max_q_next = q_next.max(axis=1)
                td_targets = rew_b + gamma * max_q_next * (1.0 - done_b)

                unique_nets = list(dict.fromkeys(nets))
                for net in unique_nets:
                    loss = net.update(obs_b, act_b, td_targets)
                    ep_losses.append(loss)

            obs_list = next_obs_list
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        for agent in agents:
            agent.epsilon = epsilon

        if (episode + 1) % target_sync_interval == 0:
            target_net.copy_weights_from(shared_net if shared_weights else nets[0])

        log_rewards.append(ep_reward)
        log_ticks.append(float(env.t))
        log_loss.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (episode + 1) % log_interval == 0:
            w = 100
            print(f"{episode+1:>8} {np.mean(log_rewards[-w:]):>12.1f} "
                  f"{np.mean(log_ticks[-w:]):>10.1f} "
                  f"{np.mean(log_loss[-w:]):>10.4f} {epsilon:>9.3f}")
            nets[0].save(os.path.join(save_dir, f"weights_ep{episode+1}"))

    final_path = os.path.join(save_dir, "weights_final")
    nets[0].save(final_path)
    np.save(os.path.join(save_dir, "episode_rewards.npy"), np.array(log_rewards))
    np.save(os.path.join(save_dir, "ticks_log.npy"),       np.array(log_ticks))
    np.save(os.path.join(save_dir, "td_loss_log.npy"),     np.array(log_loss))

    print(f"\n학습 완료. 가중치 저장: {final_path}.npz")
    return {
        "rewards": np.array(log_rewards),
        "ticks":   np.array(log_ticks),
        "loss":    np.array(log_loss),
        "net":     nets[0],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    args = sys.argv[1:]
    kwargs = {"num_episodes": 1000, "shared_weights": True, "lr": 1e-3, "seed": 42}
    for i, a in enumerate(args):
        if a == "--episodes"  and i + 1 < len(args): kwargs["num_episodes"]   = int(args[i+1])
        if a == "--lr"        and i + 1 < len(args): kwargs["lr"]             = float(args[i+1])
        if a == "--seed"      and i + 1 < len(args): kwargs["seed"]           = int(args[i+1])
        if a == "--save-dir"  and i + 1 < len(args): kwargs["save_dir"]       = args[i+1]
        if a == "--doors"     and i + 1 < len(args):
            d = int(args[i+1])
            kwargs.setdefault("env_config", {})["num_inbound_doors"] = d
            # 트럭 수 비례 증가
            kwargs["env_config"]["arrival_count_min"] = round(50 * d / 3)
            kwargs["env_config"]["arrival_count_max"] = round(70 * d / 3)
        if a == "--no-share":                         kwargs["shared_weights"] = False
        if a == "--disruptions":
            kwargs.setdefault("env_config", {}).update({
                "enable_disruptions": True,
                "disruption_door_failure": True,
                "disruption_rush_truck": True,
                "disruption_timer_shock": True,
            })
    return kwargs


if __name__ == "__main__":
    train(**_parse_args())
