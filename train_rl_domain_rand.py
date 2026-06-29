"""
train_rl_domain_rand.py — Domain Randomization 기반 DQN 재학습

매 에피소드마다 disruption_door_failure_prob를 [DR_MIN, DR_MAX]에서 균등 샘플링.
기존 checkpoints_2stage_8door/weights_final.npz 에서 fine-tune하여
checkpoints_domain_rand/weights_final.npz 에 저장.

실행:
    python train_rl_domain_rand.py
    python train_rl_domain_rand.py --episodes 2000 --from-scratch
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from rl.networks import NumpyMLP
from rl.replay_buffer import ReplayBuffer
from rl.rl_policy import QLearningPolicy, normalize_obs
from rl.train_rl import shape_rewards

# ─────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────
BASE_CFG = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 3,
    "num_outbound_doors": 3,
    "buffer_capacity": 80.0,
    "arrival_count_min": 50,
    "arrival_count_max": 70,
    "all_trucks_at_start": False,
    "arrival_pattern": "clustered",
    "arrival_cluster_count": 4,
    "arrival_time_window": 300,
    "compound_trucks": False,
    "use_truck_selection": False,
    "enable_disruptions": True,
    "disruption_door_failure": True,
    "disruption_door_failure_duration_min": 10,
    "disruption_door_failure_duration_max": 20,
}

DR_MIN = 0.02   # 학습 범위 하한
DR_MAX = 0.25   # 학습 범위 상한

PRETRAIN_CKPT = os.path.join(ROOT, "checkpoints_2stage_8door", "weights_final.npz")
SAVE_DIR      = os.path.join(ROOT, "checkpoints_domain_rand")


# ─────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────
def train_domain_rand(
    num_episodes: int = 2000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,          # fine-tune이므로 기존 1e-3보다 낮게
    hidden: int = 64,
    buffer_capacity: int = 10_000,
    warmup: int = 200,
    target_sync_interval: int = 50,
    epsilon_start: float = 0.3,  # fine-tune: 탐색 폭 축소
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
    seed: int = 42,
    log_interval: int = 100,
    from_scratch: bool = False,
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 환경 크기 파악용 (실제 학습은 매 에피소드마다 새 env)
    probe_cfg = {**BASE_CFG, "disruption_door_failure_prob": DR_MIN}
    probe_env = CrossDockEnv(probe_cfg, seed=0)
    obs_size  = probe_env.obs_size
    n_agents  = probe_env.num_lanes
    n_actions = 3

    # ── 네트워크 초기화 ──────────────────────────────
    net    = NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed)
    target = NumpyMLP(obs_size, hidden, n_actions, lr=lr, seed=seed)

    if not from_scratch and os.path.exists(PRETRAIN_CKPT):
        net.load(PRETRAIN_CKPT)
        print(f"[DR] 사전학습 가중치 로드: {PRETRAIN_CKPT}")
    else:
        print("[DR] 처음부터 학습 (from scratch)")

    target.copy_weights_from(net)

    buffer  = ReplayBuffer(buffer_capacity, obs_size, seed=seed)
    epsilon = epsilon_start
    agents  = [QLearningPolicy(net=net, epsilon=epsilon,
                               rng=np.random.default_rng(seed + k))
               for k in range(n_agents)]

    log_rewards, log_ticks, log_loss, log_probs = [], [], [], []

    print(f"\nDomain Randomization 학습 시작")
    print(f"  disruption_prob 범위: [{DR_MIN}, {DR_MAX}]")
    print(f"  episodes={num_episodes}, lr={lr}, ε_start={epsilon_start}")
    print(f"{'Episode':>8} {'AvgReward':>12} {'AvgTicks':>10} "
          f"{'TDLoss':>10} {'Epsilon':>9} {'AvgProb':>9}")
    print("-" * 62)

    for episode in range(num_episodes):
        # ── 매 에피소드: disruption_prob 랜덤 샘플링 ────
        d_prob = float(rng.uniform(DR_MIN, DR_MAX))
        cfg = {**BASE_CFG, "disruption_door_failure_prob": d_prob}
        env = CrossDockEnv(cfg, seed=seed + episode)
        obs_list = env.reset()

        ep_reward, ep_losses = 0.0, []

        for _ in range(env.episode_length):
            actions = [agents[k].act(obs_list[k], env.num_inbound_doors)
                       for k in range(n_agents)]
            next_obs_list, env_rewards, done, _ = env.step(actions)

            rewards = shape_rewards(
                env_rewards, obs_list, next_obs_list, actions,
                env.num_inbound_doors,
                use_outbound_shaping=True,
                use_inbound_shaping=True,
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
                obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(batch_size)
                q_next     = target.forward(nobs_b)
                td_targets = rew_b + gamma * q_next.max(axis=1) * (1.0 - done_b)
                loss = net.update(obs_b, act_b, td_targets)
                ep_losses.append(loss)

            obs_list = next_obs_list
            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        for agent in agents:
            agent.epsilon = epsilon

        if (episode + 1) % target_sync_interval == 0:
            target.copy_weights_from(net)

        log_rewards.append(ep_reward)
        log_ticks.append(float(env.t))
        log_loss.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        log_probs.append(d_prob)

        if (episode + 1) % log_interval == 0:
            w = 100
            net.save(os.path.join(SAVE_DIR, f"weights_ep{episode+1}"))
            print(f"{episode+1:>8} {np.mean(log_rewards[-w:]):>12.1f} "
                  f"{np.mean(log_ticks[-w:]):>10.1f} "
                  f"{np.mean(log_loss[-w:]):>10.4f} "
                  f"{epsilon:>9.3f} "
                  f"{np.mean(log_probs[-w:]):>9.3f}")

    final_path = os.path.join(SAVE_DIR, "weights_final")
    net.save(final_path)
    np.save(os.path.join(SAVE_DIR, "episode_rewards.npy"), np.array(log_rewards))
    np.save(os.path.join(SAVE_DIR, "ticks_log.npy"),       np.array(log_ticks))
    np.save(os.path.join(SAVE_DIR, "td_loss_log.npy"),     np.array(log_loss))
    np.save(os.path.join(SAVE_DIR, "disruption_probs.npy"),np.array(log_probs))

    print(f"\n학습 완료. 가중치 저장: {final_path}.npz")
    return net


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int,   default=2000)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--from-scratch",action="store_true")
    args = parser.parse_args()

    train_domain_rand(
        num_episodes=args.episodes,
        lr=args.lr,
        seed=args.seed,
        from_scratch=args.from_scratch,
    )
