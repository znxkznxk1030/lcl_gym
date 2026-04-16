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
) -> list:
    """
    환경 보상에 추가 shaping 적용 (환경 코드 수정 없음).

    + 1.0 * door_match : 선택한 도어의 화물 매칭도가 높을수록 보너스
    - 0.1 * congestion : 혼잡 억제
    """
    shaped = []
    for k, (r, obs, action) in enumerate(zip(env_rewards, obs_list, actions)):
        congestion = float(obs[1])
        door_match = 0.0
        if action > 0:
            door_idx = action - 1
            match_start = 7
            if match_start + door_idx < len(obs):
                door_match = float(obs[match_start + door_idx])

        bonus = 1.0 * door_match - 0.1 * congestion
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
) -> dict:

    os.makedirs(save_dir, exist_ok=True)

    config = {**DEFAULT_CONFIG}
    env = CrossDockEnv(config, seed=seed)
    n_agents  = env.num_lanes
    obs_size  = env.obs_size
    n_actions = env.num_inbound_doors + 1

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

    log_rewards    = []
    log_throughput = []
    log_overflow   = []
    log_loss       = []

    print(f"학습 시작 — episodes={num_episodes}, shared={shared_weights}, "
          f"lr={lr}, gamma={gamma}")
    print(f"{'Episode':>8} {'AvgReward':>12} {'Throughput':>12} "
          f"{'Overflow':>10} {'TDLoss':>10} {'Epsilon':>9}")
    print("-" * 65)

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
            rewards = shape_rewards(
                env_rewards, obs_list, next_obs_list, actions, env.num_inbound_doors
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

        metrics = info.get("metrics", env.metrics)
        log_rewards.append(ep_reward)
        log_throughput.append(metrics["total_throughput"])
        log_overflow.append(metrics["buffer_overflow_count"])
        log_loss.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (episode + 1) % log_interval == 0:
            w = 100
            print(f"{episode+1:>8} {np.mean(log_rewards[-w:]):>12.1f} "
                  f"{np.mean(log_throughput[-w:]):>12.1f} "
                  f"{np.mean(log_overflow[-w:]):>10.1f} "
                  f"{np.mean(log_loss[-w:]):>10.4f} {epsilon:>9.3f}")
            nets[0].save(os.path.join(save_dir, f"weights_ep{episode+1}"))

    final_path = os.path.join(save_dir, "weights_final")
    nets[0].save(final_path)
    np.save(os.path.join(save_dir, "episode_rewards.npy"),  np.array(log_rewards))
    np.save(os.path.join(save_dir, "throughput_log.npy"),   np.array(log_throughput))
    np.save(os.path.join(save_dir, "overflow_log.npy"),     np.array(log_overflow))
    np.save(os.path.join(save_dir, "td_loss_log.npy"),      np.array(log_loss))

    print(f"\n학습 완료. 가중치 저장: {final_path}.npz")
    return {
        "rewards":    np.array(log_rewards),
        "throughput": np.array(log_throughput),
        "overflow":   np.array(log_overflow),
        "loss":       np.array(log_loss),
        "net":        nets[0],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    args = sys.argv[1:]
    kwargs = {"num_episodes": 1000, "shared_weights": True, "lr": 1e-3, "seed": 42}
    for i, a in enumerate(args):
        if a == "--episodes" and i + 1 < len(args): kwargs["num_episodes"]   = int(args[i+1])
        if a == "--lr"       and i + 1 < len(args): kwargs["lr"]             = float(args[i+1])
        if a == "--seed"     and i + 1 < len(args): kwargs["seed"]           = int(args[i+1])
        if a == "--no-share":                        kwargs["shared_weights"] = False
    return kwargs


if __name__ == "__main__":
    train(**_parse_args())
