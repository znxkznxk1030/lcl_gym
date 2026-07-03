#!/usr/bin/env python3
"""
RL(IQL+DQN) 기반 Compound Truck 목적지 배정 — "우리 모델"의 배정 최적화 버전.

본 리포트의 RL은 매 tick lane action을 정하는 실시간 정책이라 compound makespan을 좌우하는
'목적지 배정'을 결정하지 못한다. 그래서 makespan을 직접 최적화하도록 RL을 **순차적 배정 MDP**로
재정의한다(논문의 RL-기반 메타휴리스틱과 같은 취지 — RL이 makespan 최소화를 학습).

MDP:
  - 스텝 i = compound 트럭 i 를 (미사용) 목적지 하나에 배정 (총 I 스텝)
  - 상태  = [정규화 f_i(nD), 사용된 목적지 마스크(nD), 잔여 통합수요(nD), 진행도] ∈ R^(3·nD+1)
  - 행동  = 목적지 d ∈ {0..nD-1} (이미 사용된 목적지는 마스킹)
  - 보상  = 종료 시 -makespan/SCALE, 중간 0,  γ=1
  - 네트워크: NumpyMLP(obs → 64 → nD), DQN(타깃망·리플레이·ε-greedy)

학습:  python rl_compound_assignment.py            (가중치 → checkpoints_compound_rl/weights_final.npz)
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.entities import Truck
from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import FIFOPolicy
from rl.networks import NumpyMLP
from compound_baselines import makespan_analytic, assign_greedy

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_compound_rl")
WEIGHTS = os.path.join(CKPT_DIR, "weights_final.npz")        # 무결(disruption 없음) 학습
DR_WEIGHTS = os.path.join(CKPT_DIR, "weights_dr.npz")        # disruption 하 학습 (RL-DR)

I_TRUCKS = 5
N_DEST = 7
SCALE = 100.0  # 보상 정규화 (-makespan/SCALE)

# disruption 학습/평가 공통 설정 (논문 민감도 분석 + 도어 고장)
DR_DEMAND_MAX = 20
DR_T_K = 8
DR_PROB = 0.05  # 도어 고장 확률/스텝 (resume-on-failure 모델)


def disrupt_cfg(prob=DR_PROB, demand_max=DR_DEMAND_MAX, t_k=DR_T_K):
    return {
        **DEFAULT_CONFIG, "compound_trucks": True, "partial_unloading": True,
        "all_trucks_at_start": True, "use_scheduled_arrivals": True,
        "num_lanes": N_DEST, "num_destinations": N_DEST, "num_inbound_doors": 5,
        "num_outbound_doors": N_DEST, "num_compound_trucks": I_TRUCKS,
        "outbound_capacity": 1e6, "buffer_capacity": 1e9, "episode_length": 200000,
        "unit_load_time": t_k, "demand_min": 0, "demand_max": demand_max,
        "enable_disruptions": prob > 0, "disruption_door_failure": prob > 0,
        "disruption_door_failure_prob": prob,
        "disruption_door_failure_duration_min": 10,
        "disruption_door_failure_duration_max": 20,
    }


def peek_env_trucks(cfg, seed):
    env = CrossDockEnv(config=cfg, seed=seed); env.reset()
    return [t for t in env.waiting_trucks if t.truck_type == "compound"]


def sim_makespan_env(cfg, seed, override):
    c = {**cfg, "compound_dest_override": override}
    env = CrossDockEnv(config=c, seed=seed); obs = env.reset()
    pols = [FIFOPolicy() for _ in range(env.num_lanes)]
    done = False
    while not done:
        obs, _, done, _ = env.step(
            [pols[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)])
    return env.t


# ──────────────────────────────────────────────────────────────────────
# 인스턴스 생성 / 상태
# ──────────────────────────────────────────────────────────────────────
def gen_instance(rng, demand_max, t_k, I=I_TRUCKS, nD=N_DEST):
    comp = []
    for _ in range(I):
        ship = {d: int(rng.integers(0, demand_max + 1)) for d in range(nD)}
        DE = int(rng.integers(3, 11)); DL = int(rng.integers(3, 11))
        comp.append(Truck(arrival_time=0, shipments=ship, truck_type="compound", DE=DE, DL=DL))
    out = []
    for _ in range(nD - I):
        DL = int(rng.integers(3, 11))
        out.append(Truck(arrival_time=0, shipments={0: 0}, truck_type="outbound", DL=DL))
    return comp, out


def build_state(comp, step, used, demand_max, nD=N_DEST):
    """배정 중인 트럭 step 에 대한 상태 벡터."""
    I = len(comp)
    f = np.array([comp[step].dest_volume(d) for d in range(nD)], float) / max(demand_max, 1)
    mask = np.array([1.0 if d in used else 0.0 for d in range(nD)], float)
    # 잔여 통합수요: 아직 배정 안 된 트럭(step..I-1)들의 목적지별 화물 합
    rem = np.zeros(nD)
    for j in range(step, I):
        for d in range(nD):
            rem[d] += comp[j].dest_volume(d)
    rem = rem / (max(demand_max, 1) * I)
    return np.concatenate([f, mask, rem, [step / I]]).astype(np.float32)


def valid_actions(used, nD=N_DEST):
    return [d for d in range(nD) if d not in used]


# ──────────────────────────────────────────────────────────────────────
# 정책 (학습된 가중치로 greedy 배정)
# ──────────────────────────────────────────────────────────────────────
def assign_rl(net, comp, demand_max, nD=N_DEST):
    used = set(); assign = {}
    for i in range(len(comp)):
        s = build_state(comp, i, used, demand_max, nD)
        q = net.forward(s)
        va = valid_actions(used, nD)
        d = max(va, key=lambda d: q[d])
        assign[i] = d; used.add(d)
    return assign


# ──────────────────────────────────────────────────────────────────────
# 학습 (DQN)
# ──────────────────────────────────────────────────────────────────────
def train(episodes=40000, hidden=64, lr=5e-4, batch=128,
          buf_cap=40000, seed=0,
          demand_choices=(10, 20, 30), t_k_choices=(4, 6, 8, 10), verbose=True):
    """Monte-Carlo 회귀(γ=1, 고정 5-스텝) + greedy 대비 상대개선 보상.

    보상 = (greedy_makespan − makespan) / (greedy_makespan + 1)   ∈ 스케일 불변
      → 한 배정의 모든 transition 에 동일 적용(MC). Q(s,a) ≈ 그 행동으로 도달하는 기대 상대개선.
      → 부트스트랩 불안정성 제거, t_k·demand 스케일 차이에 무관하게 학습.
    """
    nD = N_DEST
    obs_size = 3 * nD + 1
    rng = np.random.default_rng(seed)
    net = NumpyMLP(obs_size=obs_size, hidden=hidden, n_actions=nD, lr=lr, seed=seed)
    buf = deque(maxlen=buf_cap)
    eps, eps_min, eps_decay = 1.0, 0.05, 0.9997
    recent_gap = deque(maxlen=1000)

    for ep in range(episodes):
        demand_max = int(rng.choice(demand_choices))
        t_k = float(rng.choice(t_k_choices))
        comp, out = gen_instance(rng, demand_max, t_k)
        g_mk = makespan_analytic(comp, out, t_k, nD, assign_greedy(comp, nD), partial=True)
        used = set(); assign = {}
        traj = []
        for i in range(I_TRUCKS):
            s = build_state(comp, i, used, demand_max, nD)
            va = valid_actions(used, nD)
            if rng.random() < eps:
                a = int(rng.choice(va))
            else:
                q = net.forward(s)
                a = max(va, key=lambda d: q[d])
            used.add(a); assign[i] = a
            traj.append((s, a))
        mk = makespan_analytic(comp, out, t_k, nD, assign, partial=True)
        ret = (g_mk - mk) / (g_mk + 1.0)          # greedy 대비 상대개선 (MC return)
        recent_gap.append(ret)
        for s, a in traj:
            buf.append((s, a, ret))

        if len(buf) >= batch:
            idx = rng.choice(len(buf), size=batch, replace=False)
            S = np.array([buf[j][0] for j in idx], np.float32)
            A = np.array([buf[j][1] for j in idx], np.int64)
            T = np.array([buf[j][2] for j in idx], np.float32)
            net.update(S, A, T)

        eps = max(eps_min, eps * eps_decay)
        if verbose and ep % 4000 == 0 and recent_gap:
            # 양수면 greedy보다 우수 (탐험 포함 평균)
            print(f"ep {ep:6d}  eps={eps:.3f}  recent_rel_vs_greedy={np.mean(recent_gap):+.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    net.save(WEIGHTS)
    if verbose:
        print(f"[저장] {WEIGHTS}")
    return net


def train_disrupt(episodes=8000, hidden=128, lr=5e-4, batch=128, buf_cap=40000,
                  seed=0, n_train_seeds=2500, prob=DR_PROB, verbose=True):
    """Door disruption 하에서 makespan을 최소화하도록 배정을 학습 (RL-DR).

    disruption이 있으면 해석 공식이 무효 → 시뮬레이터 makespan을 보상으로 사용.
    보상 = (greedy_sim − rl_sim)/(greedy_sim+1), 동일 seed(트럭+고장 realization 고정).
    greedy_sim 은 seed별 1회 캐시. 학습 seed 풀(1000~)과 평가 seed(0~19)를 분리.
    """
    nD = N_DEST; obs_size = 3 * nD + 1
    rng = np.random.default_rng(seed)
    net = NumpyMLP(obs_size=obs_size, hidden=hidden, n_actions=nD, lr=lr, seed=seed)
    buf = deque(maxlen=buf_cap)
    eps, eps_min, eps_decay = 1.0, 0.05, 0.9996
    cfg = disrupt_cfg(prob=prob)
    train_seeds = list(range(1000, 1000 + n_train_seeds))
    greedy_cache, trucks_cache = {}, {}
    recent = deque(maxlen=1000)

    def get(seed):
        if seed not in trucks_cache:
            comp = peek_env_trucks(cfg, seed)
            trucks_cache[seed] = comp
            greedy_cache[seed] = sim_makespan_env(cfg, seed, assign_greedy(comp, nD))
        return trucks_cache[seed], greedy_cache[seed]

    for ep in range(episodes):
        s = int(rng.choice(train_seeds))
        comp, g_mk = get(s)
        used = set(); assign = {}; traj = []
        for i in range(I_TRUCKS):
            st = build_state(comp, i, used, DR_DEMAND_MAX, nD)
            va = valid_actions(used, nD)
            a = int(rng.choice(va)) if rng.random() < eps else max(va, key=lambda d: net.forward(st)[d])
            used.add(a); assign[i] = a; traj.append((st, a))
        rl_mk = sim_makespan_env(cfg, s, assign)
        ret = (g_mk - rl_mk) / (g_mk + 1.0)
        recent.append(ret)
        for st, a in traj:
            buf.append((st, a, ret))
        if len(buf) >= batch:
            idx = rng.choice(len(buf), size=batch, replace=False)
            S = np.array([buf[j][0] for j in idx], np.float32)
            A = np.array([buf[j][1] for j in idx], np.int64)
            T = np.array([buf[j][2] for j in idx], np.float32)
            net.update(S, A, T)
        eps = max(eps_min, eps * eps_decay)
        if verbose and ep % 1000 == 0 and recent:
            print(f"ep {ep:6d}  eps={eps:.3f}  rel_vs_greedy={np.mean(recent):+.4f}  "
                  f"(cached_seeds={len(greedy_cache)})", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    net.save(DR_WEIGHTS)
    if verbose:
        print(f"[저장] {DR_WEIGHTS}")
    return net


def _load(path):
    if not os.path.exists(path):
        return None
    hidden = int(np.load(path)["W1"].shape[1])
    net = NumpyMLP(obs_size=3 * N_DEST + 1, hidden=hidden, n_actions=N_DEST)
    net.load(path)
    return net


def load_policy():
    """무결(disruption 없음) 학습 RL 배정 정책. 가중치 없으면 None."""
    return _load(WEIGHTS)


def load_policy_dr():
    """Door disruption 하 학습 RL-DR 배정 정책. 가중치 없으면 None."""
    return _load(DR_WEIGHTS)


if __name__ == "__main__":
    train()
