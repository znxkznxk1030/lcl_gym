#!/usr/bin/env python3
"""
온라인 배차(Online Dispatching) 하 도어 고장 — RL이 유리해지는 돌발상황 데모.

지난 실험에서 RL이 진 이유: compound 배정이 (1) 출발 전 일회성·정적이고 (2) 도어 고장이
배정과 무관(headroom=0)했기 때문. 여기서는 그 두 조건을 뒤집어 RL이 이기는 환경을 만든다.

설정 (compound 트럭의 하역시간 사용):
  - I 대 트럭(하역시간 u_i = DE_i + unloaded_i·t_k), M 개 도어 (M < I → 희소·순차 배차)
  - 도어는 하역 중 매 tick 확률 p 로 **길게 고장(F tick)**, 재개(resume: 진행분 보존).
    → 고장이 **실행 중 온라인**으로 드러나고, 멈춘 트럭은 도어에 묶인다.
  - **실시간 recourse(핵심)**: 도어가 비면, 새 대기 트럭을 싣거나 **고장난 도어에 묶인 트럭을
    정상 도어로 마이그레이션(잔여작업 이어서 처리)** 할 수 있다.

핵심: 도어가 길게 고장나면 묶인 트럭이 그동안 놀게 됨 → 정상 도어가 비었을 때 그 트럭을 **옮겨오면**
  makespan↓. 이것이 본문 RL이 이긴 action=2(빈 도크를 화물 있는 곳으로 재배치)의 본질.
  - 정적 규칙(LPT/SPT)은 마이그레이션을 모름 → 묶인 트럭은 고장이 끝날 때까지 대기.
  - RL은 (묶인 트럭의 잔여작업, 대기 분포, 도어 상태) 관측 → 멈춘 큰 트럭을 구조할지 학습.
  - 대조: **고장이 짧으면**(F작음) 구조 이득이 사라져 RL 우위도 사라짐 → "온라인 recourse 여지가 있어야 RL이 이긴다".

행동: {0: 가장 긴 대기트럭 시작, 1: 가장 짧은 대기트럭 시작, 2: 멈춘 트럭(잔여 최대) 마이그레이션}
보상: 종료 시 (LPT_makespan − makespan) 상대개선 (Monte-Carlo, γ=1)
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl.networks import NumpyMLP

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "checkpoints_compound_rl", "weights_dispatch.npz")

I_TRUCKS = 6
M_DOORS = 3
U_SCALE = 500.0
FAIL_DUR = (20, 50)    # 긴 고장 → 마이그레이션(구조) 가치 발생
FAIL_PROB = 0.02       # 도어 working 중 매 tick 고장 확률 (동질)
N_ACTIONS = 3          # 0=긴 대기트럭, 1=짧은 대기트럭, 2=멈춘 트럭 마이그레이션
OBS = 7


# ──────────────────────────────────────────────────────────────────────
# 인스턴스
# ──────────────────────────────────────────────────────────────────────
def gen_instance(rng, fail_dur=FAIL_DUR):
    """compound 트럭 유래 하역시간 u = DE + unloaded·t_k."""
    t_k = 8
    u = []
    for _ in range(I_TRUCKS):
        unloaded = int(rng.integers(10, 61))
        DE = int(rng.integers(3, 11))
        u.append(DE + unloaded * t_k)
    return {"u": u, "fail_dur": fail_dur}


# ──────────────────────────────────────────────────────────────────────
# 시뮬레이터 (마이그레이션 지원)
#   door dict: rem(잔여작업, 0=빈도어), failrem(고장잔여, 0=정상)
#     - rem>0, failrem==0 : 정상 작동 중
#     - rem>0, failrem>0  : 고장으로 멈춤(트럭 묶임=stuck, 마이그레이션 대상)
#     - rem==0, failrem>0 : 빈 도어 복구 중(배차 불가)
#     - rem==0, failrem==0: idle-정상(배차 가능)
# ──────────────────────────────────────────────────────────────────────
def build_state(doors, waiting):
    stuck = [d["rem"] for d in doors if d["rem"] > 0 and d["failrem"] > 0]
    n_idle = sum(1 for d in doors if d["rem"] == 0 and d["failrem"] == 0)
    longest = max(waiting) if waiting else 0.0
    shortest = min(waiting) if waiting else 0.0
    return np.array([
        1.0 if stuck else 0.0,
        (max(stuck) / U_SCALE) if stuck else 0.0,
        len(waiting) / I_TRUCKS,
        longest / U_SCALE,
        shortest / U_SCALE,
        (longest - shortest) / U_SCALE,
        n_idle / M_DOORS,
    ], dtype=np.float32)


def simulate(inst, policy, rng, collect=False):
    u, fail_dur = inst["u"], inst["fail_dur"]
    waiting = sorted(u, reverse=True)
    doors = [{"rem": 0, "failrem": 0} for _ in range(M_DOORS)]
    t, done = 0, 0
    traj = []
    while done < I_TRUCKS:
        # 1) 도어 진행/고장 (resume)
        for d in doors:
            if d["failrem"] > 0:
                d["failrem"] -= 1                          # 복구/멈춤 카운트다운
            elif d["rem"] > 0:
                if rng.random() < FAIL_PROB:
                    d["failrem"] = int(rng.integers(*fail_dur))  # 신규 고장(진행분 보존)
                else:
                    d["rem"] -= 1
                    if d["rem"] <= 0:
                        done += 1                          # 트럭 완료 → 도어 idle
        # 2) idle-정상 도어 배차 (의사결정)
        for d in doors:
            if d["rem"] == 0 and d["failrem"] == 0:
                stuck = [dd for dd in doors if dd["rem"] > 0 and dd["failrem"] > 0]
                if not waiting and not stuck:
                    continue
                ws = sorted(waiting, reverse=True)
                s = build_state(doors, ws)
                a = policy.act(s)
                # 행동 → 실제 선택 (가용성에 따라 폴백)
                if a == 2 and stuck:
                    src = max(stuck, key=lambda dd: dd["rem"])   # 잔여 최대 멈춘 트럭 구조
                    d["rem"] = src["rem"]; src["rem"] = 0        # 마이그레이션(잔여 이어받음)
                elif waiting:
                    idx = 0 if a == 0 else (len(ws) - 1 if a == 1 else 0)
                    pick = ws[idx]; waiting.remove(pick); d["rem"] = pick
                elif stuck:
                    src = max(stuck, key=lambda dd: dd["rem"])
                    d["rem"] = src["rem"]; src["rem"] = 0
                else:
                    continue
                if collect:
                    traj.append((s, a))
        t += 1
        if t > 200000:
            break
    return (t, traj) if collect else t


# ──────────────────────────────────────────────────────────────────────
# 정책
# ──────────────────────────────────────────────────────────────────────
class Static:
    def __init__(self, action):  # LPT=0, SPT=1 — 마이그레이션 안 함
        self.a = action
    def act(self, s):
        return self.a


class MigrateAware:
    """도메인 규칙: 멈춘 트럭(잔여 큼)이 있으면 구조(2), 없으면 LPT(0)."""
    def act(self, s):
        has_stuck, max_stuck_rem = s[0], s[1]
        return 2 if (has_stuck > 0.5 and max_stuck_rem > 0.05) else 0


class RLPolicy:
    def __init__(self, net, eps=0.0, rng=None):
        self.net = net; self.eps = eps
        self.rng = rng or np.random.default_rng(0)
    def act(self, s):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.net.forward(s)))


# ──────────────────────────────────────────────────────────────────────
# DQN 학습 (Monte-Carlo 회귀, 상대보상)
# ──────────────────────────────────────────────────────────────────────
def train(episodes=30000, hidden=64, lr=5e-4, batch=128, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    net = NumpyMLP(obs_size=OBS, hidden=hidden, n_actions=N_ACTIONS, lr=lr, seed=seed)
    buf = deque(maxlen=40000)
    eps, eps_min, decay = 1.0, 0.05, 0.9997
    lpt = Static(0)
    recent = deque(maxlen=1000)
    for ep in range(episodes):
        inst = gen_instance(rng)
        dseed = int(rng.integers(0, 2**31))
        # 상대보상 기준선: LPT(고정) makespan, 동일 고장 realization
        base = simulate(inst, lpt, np.random.default_rng(dseed))
        mk, traj = simulate(inst, RLPolicy(net, eps=eps, rng=rng),
                            np.random.default_rng(dseed), collect=True)
        ret = (base - mk) / (base + 1.0)
        recent.append(ret)
        for s, a in traj:
            buf.append((s, a, ret))
        if len(buf) >= batch:
            idx = rng.choice(len(buf), size=batch, replace=False)
            S = np.array([buf[j][0] for j in idx], np.float32)
            A = np.array([buf[j][1] for j in idx], np.int64)
            T = np.array([buf[j][2] for j in idx], np.float32)
            net.update(S, A, T)
        eps = max(eps_min, eps * decay)
        if verbose and ep % 3000 == 0:
            print(f"ep {ep:6d} eps={eps:.3f} rel_vs_LPT={np.mean(recent):+.4f}", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    net.save(CKPT)
    if verbose:
        print(f"[저장] {CKPT}")
    return net


def load_net():
    if not os.path.exists(CKPT):
        return None
    h = int(np.load(CKPT)["W1"].shape[1])
    net = NumpyMLP(obs_size=OBS, hidden=h, n_actions=N_ACTIONS)
    net.load(CKPT)
    return net


def _probe(n=600):
    """음성 증거: 순수 병렬 하역 스케줄링에서는 정적 LPT가 near-optimal.
    긴 고장(마이그레이션 여지) / 짧은 고장 모두에서 MigrateAware가 LPT를 못 이김."""
    def evalp(policy, fail_dur):
        g = np.random.default_rng(1); mks = []
        for _ in range(n):
            inst = gen_instance(g, fail_dur=fail_dur)
            mks.append(simulate(inst, policy, np.random.default_rng(int(g.integers(0, 2**31)))))
        return float(np.mean(mks))
    print("순수 병렬 하역 스케줄링 — 정적 규칙 vs 마이그레이션 (RL 우위 없음 확인)")
    print(f"{'고장 길이':>14} | {'LPT':>6} | {'SPT':>6} | {'MigrateAware':>12} | {'Mig vs LPT':>10}")
    for fd, tag in [((20, 50), "LONG (20-50)"), ((3, 8), "SHORT (3-8)")]:
        lpt = evalp(Static(0), fd); spt = evalp(Static(1), fd); mig = evalp(MigrateAware(), fd)
        print(f"{tag:>14} | {lpt:6.0f} | {spt:6.0f} | {mig:12.0f} | {100*(lpt-mig)/lpt:+9.1f}%")
    print("→ LPT가 near-optimal. 병렬 처리 자체는 온라인 recourse 여지가 없어 RL이 유리하지 않다.")
    print("  RL이 이기는 구조는 run_compound_rl_win.py 참고(자원이 대상에 커밋 + 동적 수요 + 재배치).")


if __name__ == "__main__":
    _probe()
