# Cross-Dock MARL Simulator

크로스도킹(Cross-Docking) 물류 환경을 다중 에이전트 강화학습(MARL)용으로 구현한 시뮬레이터입니다.

> **크로스도킹이란?** 입고된 화물을 창고에 보관하지 않고, 즉시 목적지별 레인으로 분류해 출고하는 물류 방식입니다.

---

## 시뮬레이션 환경 개요

### 전체 흐름

```
[인바운드 트럭] → [인바운드 도어 3개] → [정렬 버퍼]
                                               ↓
                                    [레인 큐 5개 (에이전트)]
                                               ↓
                              [아웃바운드 도크 3개] → [출발]
```

에피소드는 **모든 화물이 아웃바운드 도크를 통해 출발할 때까지** 진행되며,
총 소요 시간(Total Ticks)이 짧을수록 좋은 성과입니다.

### 2-Stage 구조

| 스테이지 | 역할 | 에이전트 관여 |
|---|---|---|
| **Stage 1 (인바운드)** | 대기 트럭을 유휴 인바운드 도어에 배정 → 하역 후 버퍼/레인으로 분류 | action=1 (트럭 요청) |
| **Stage 2 (아웃바운드)** | 레인 큐의 화물을 아웃바운드 도크에 적재 → 타이머 만료 시 출발 | action=2 (도크 우선 배정) |

---

## 핵심 엔티티

### Truck (인바운드 트럭)

- **혼재 화물**: 한 트럭에 **2~3개 목적지의 화물이 혼재**. 예: {레인0: 3CBM, 레인2: 1.5CBM}
- 스케줄된 도착 시간에 등장 (clustered 패턴, 4개 배치)
- 에피소드당 50~70대 도착

### InboundDoor (인바운드 도어)

- 트럭 하역 전용 도어. 처리 중에는 점유 상태(busy)
- 하역 시간: 1~10 스텝 (랜덤)
- **돌발사항**: 일정 확률로 도어 고장 발생 (10~20 스텝 비가동)

### Buffer (정렬 버퍼)

- 하역 완료된 화물이 레인으로 이동하기 전 대기하는 공유 공간
- 용량: **80 CBM** (유한). 초과 시 overflow 패널티
- 버퍼에서 레인으로의 분류는 자동(즉시)

### Lane (정렬 레인, 에이전트)

- 목적지별 5개 레인 (0~4). 각 레인이 하나의 **에이전트**에 해당
- 레인별 큐에 화물이 쌓이면 아웃바운드 도크를 통해 출하
- 레인 에이전트가 매 스텝 action(0/1/2)을 결정

### OutboundDoor (아웃바운드 도크)

- **아웃바운드 도크 3개 < 레인 5개** → 도크 희소성 발생
- 도크는 특정 레인에 배정되어 화물을 점진적으로 적재(progressive loading)
- 최대 적재량: 15 CBM. 배정 후 12~28 스텝 내 출발
- 타이머 만료 시 적재 화물을 싣고 출발 (미적재 시 empty departure)

---

## 에이전트 행동 / 관측 / 보상

### 행동 공간 (3-Action)

```
0 → skip          — 아무것도 하지 않음
1 → 인바운드 요청  — 유휴 도어에 트럭 배정 요청
2 → 아웃바운드 부스트 — 내 레인에 아웃바운드 도크 우선 배정 요청
```

- **action=1**: 여러 레인이 동시에 요청하면 아웃바운드 타이머 긴급도 순으로 도어 배정
- **action=2 (핵심 메커니즘)**:
  - 유휴 도크가 있으면 내 레인에 우선 배정
  - 빈 레인을 서비스 중인 도크를 내 레인으로 **mid-trip 재배정** (cargo 없이 카운트다운 중인 도크 구조조정)

### 관측 벡터 (크기 = 9 + D, D=인바운드 도어 수)

```
obs[0]  lane_queue          — 내 레인의 현재 화물량 (CBM)
obs[1]  lane_congestion     — 레인 혼잡도 (0~1 정규화)
obs[2]  outbound_fill_rate  — 내 레인 아웃바운드 도크의 현재 탑재율 (0: 도크 없음)
obs[3]  outbound_timer      — 내 레인 도크의 출발까지 남은 스텝
obs[4]  buffer_fill_ratio   — 버퍼 충전율 [0, 2] (1.0=100%, 2.0=200% 포화)
obs[5]  idle_inbound_doors  — 현재 유휴 인바운드 도어 수
obs[6]  waiting_trucks      — 도착해 대기 중인 트럭 수
obs[7]  scheduled_trucks    — 미도착 스케줄 트럭 수
obs[8]  idle_outbound_doors — 현재 유휴 아웃바운드 도크 수
obs[9..9+D-1]  door_match_i — 각 인바운드 도어별 매칭 점수 (내 레인 화물 비율)
```

### 보상 구조

```
R_team  =  이번_스텝_전체_출발_화물량  − 1.0 × 빈_출발 수
R_local =  내_레인_출발_화물량
overflow_penalty = −1.0 − 0.3 × overflow_volume  (버퍼 초과 시)

R_final = 0.7 × R_team + 0.3 × R_local  (+ overflow_penalty)
```

---

## 돌발사항 (Disruptions)

| 이벤트 | 확률 | 효과 |
|---|---|---|
| **도어 고장** | 2%/스텝 | 인바운드 도어 1개가 10~20 스텝 비가동 |

에피소드당 평균 6~7회 도어 고장 발생. 에이전트는 관측값(`idle_inbound_doors` 감소)을 통해 간접 감지.

---

## 정책 설명

### 베이스라인 3종

| 정책 | 전략 |
|---|---|
| `RandomPolicy` | 매 스텝 50% 확률로 action=0 또는 1 선택 |
| `FIFOPolicy` | 대기 트럭과 유휴 도어가 있으면 항상 action=1. action=2 사용 안 함 |
| `GreedyPolicy` | 화물 있는데 도크 없으면 action=2, 대기 트럭+유휴 도어 있으면 action=1 |

### HeuristicPriorityPolicy

```
1) lane_queue > 0 AND fill_rate == 0  →  action=2 (도크 재배정 요청)
2) buffer_fill > 1.5                  →  action=0 (버퍼 포화 시 인바운드 억제)
3) urgency + best_match − congestion ≥ threshold  →  action=1
4) 그 외  →  action=0
```

### MILP (Mixed Integer Linear Programming)

매 스텝 `maximize Σ x_{j,i} · score_j` 를 CBC 솔버로 풀어 트럭-도어 배정을 최적화합니다.

- `score_j = Σ_k v_{j,k} / (departure_timer_k + 1)` — 긴급도 가중 화물량
- 트럭·도어 각각 단일 배정 제약
- action=2 조건: `fill_rate == 0` → 도크 재배정 트리거
- 평균 ~9ms/스텝

### RL (IQL + Parameter Sharing DQN)

numpy 기반 2층 MLP를 5개 레인 에이전트가 가중치 공유하여 학습합니다.

```
입력(9+D) → Linear(64) → ReLU → Linear(3) → Q값 {Q_skip, Q_inbound, Q_outbound}
```

- **학습**: 2000 에피소드, lr=0.001, γ=0.99, ε: 1.0→0.05 (decay 0.995)
- **Target Network**: 50 에피소드마다 동기화
- **리플레이 버퍼**: 용량 10,000, 배치 64
- **Reward Shaping**: `needs_dock`(화물 있는데 도크 없음) 상황에서 action=2 보너스 +0.8
- 가중치 저장: `checkpoints_2stage_8door/weights_final.npz`

### GA (Genetic Algorithm, 7-gene)

7개 유전자로 점수 기반 의사결정을 진화시킵니다.

```python
genes = [w_urgency, w_match, w_congestion, w_buffer, w_waiting, threshold, outbound_timer_thresh]

# action=2 조건 (도크 재배정)
if lane_queue > 0 and (fill_rate == 0 or timer < outbound_timer_thresh):
    return 2

# action=1 조건 (인바운드 요청)
score = w_urgency × urgency + w_match × best_match
      − w_congestion × congestion + w_buffer × buf_fill + w_waiting × wait_norm
if score > threshold:
    return 1
```

- **설정**: POP=30, GEN=50, 평가 에피소드=5회 평균
- 유전자 저장: `ga/best_genes_2stage.json`

---

## 실험 결과

### 공통 환경 설정

| 파라미터 | 값 |
|---|---|
| Lanes (에이전트) | 5 |
| 인바운드 도어 | 3 |
| 아웃바운드 도크 | 3 |
| 버퍼 용량 | 80 CBM |
| 트럭 수 / 에피소드 | 50 ~ 70대 |
| 트럭 화물 목적지 | 2 ~ 3개 레인 혼재 |
| 목적지별 화물량 | 1 ~ 5 CBM |
| 아웃바운드 도크 용량 | 15 CBM |
| 인바운드 처리 시간 | 1 ~ 10 스텝 |
| 아웃바운드 로딩 시간 | 12 ~ 28 스텝 |
| 도착 패턴 | Clustered (4 batch, 300 ticks 이내) |
| 돌발사항 | 도어 고장 · 2%/스텝 · 지속 10~20 스텝 |
| 에피소드 종료 조건 | 모든 화물 처리 완료 (최대 10,000 스텝) |

---

### 실험 1 — Lane-mode 3-action · FIFO 입고 (20260531_004)

인바운드 트럭 배정: **FIFO** (대기열 맨 앞 트럭)

| 정책 | Avg Ticks ↓ | Std | seed42 | 빈 출발 ↓ | 출발 횟수 |
|---|---:|---:|---:|---:|---:|
| 🥇 **RL**  | **339.3** | 12.9 | 345 | **3.35** | **48.85** |
| Random     | 340.6 | 13.2 | 353 | 4.75 | 50.35 |
| MILP       | 340.8 | 12.0 | 353 | 5.00 | 50.35 |
| FIFO       | 340.9 | 11.8 | 353 | 4.75 | 50.40 |
| Heuristic  | 347.6 | 19.8 | 350 | 4.10 | 50.55 |
| GA         | 348.6 | 19.5 | 362 | 4.20 | 50.45 |
| Greedy     | 349.2 | 21.4 | 350 | 4.50 | 50.60 |

---

### 실험 2 — Lane-mode 3-action · Best-Match 입고 (20260531_005)

인바운드 트럭 배정: **Best-Match** (내 레인 화물이 가장 많은 트럭 우선)

| 정책 | Avg Ticks ↓ | Std | seed42 | 빈 출발 ↓ | 출발 횟수 | FIFO 대비 |
|---|---:|---:|---:|---:|---:|---:|
| 🥇 **RL**  | **339.1** | 11.9 | 366 | **2.70** | **48.80** | -0.2 |
| Random     | 339.5 | 12.5 | 354 | 4.75 | 50.25 | -1.1 |
| MILP       | 341.1 | 12.0 | 354 | 5.00 | 50.50 | +0.3 |
| FIFO       | 341.6 | 12.4 | 354 | 4.85 | 50.50 | +0.7 |
| **GA**     | **344.9** | 17.5 | 349 | 4.10 | 50.45 | **-3.7** |
| Heuristic  | 349.6 | 20.6 | 368 | 4.55 | 50.25 | +2.0 |
| Greedy     | 350.9 | 17.1 | 335 | 4.50 | 50.45 | +1.7 |

> **처리량**: 모든 정책이 444.8 ± 56.8 CBM으로 동일 (에피소드는 모든 화물 처리 후 종료)  
> **승부 지표**: 동일 화물을 얼마나 빨리 처리하는가 → Total Ticks 최소화

### Best-Match 변경 효과

- **GA +3.7 ticks 개선**: 유전자 스코어링이 올바른 트럭을 받으면서 효과 극대화
- **RL 빈 출발 3.35 → 2.70**: 레인 큐가 충실해져 도크 낭비 감소
- **FIFO · Greedy · Heuristic 소폭 증가**: action 로직에서 트럭 선택 이점을 충분히 활용하지 못함

### RL이 FIFO를 이기는 이유

```
FIFO는 action=2를 사용하지 않음
→ 빈 레인을 서비스하는 도크가 헛되이 카운트다운 → Empty Departure 증가

RL은 action=2로 mid-trip 재배정을 선택적으로 트리거
→ 빈 도크를 화물 있는 레인으로 전환 → 총 출발 횟수·빈 출발 모두 감소
→ 동일 처리량을 더 적은 트립으로 완료 → Total Ticks 단축
```

---

## Action=2 Mid-trip 재배정 메커니즘

```python
# _reassign_empty_serving_docks() 핵심 로직
# 조건: 화물 없는 레인을 서비스 중이고, 아직 cargo를 싣지 않은 도크만 재배정
empty_serving = [
    od for od in self.outbound_doors
    if od.is_busy
    and self.lanes[od.assigned_dest].queue_volume == 0  # 빈 레인
    and od.loaded == 0  # 아직 cargo 없음 (소실 방지)
]
# action=2를 선택한 레인 중 화물이 있는 레인으로 재배정
```

`od.loaded == 0` 조건이 핵심: 이미 화물을 실은 도크를 재배정하면 cargo 소실 버그 발생.

---

## 파일 구성

```
lcl_gym/
├── env/
│   ├── entities.py               # Truck, OutboundTruck, Door, Lane, OutboundDoor 데이터 클래스
│   ├── crossdock_env.py          # CrossDockEnv 메인 환경 (2-Stage, 3-action)
│   └── policies.py               # 베이스라인 정책 (Random, FIFO, Greedy, Heuristic)
│
├── rl/
│   ├── networks.py               # numpy 2층 MLP (Adam 역전파)
│   ├── replay_buffer.py          # 경험 리플레이 버퍼
│   ├── rl_policy.py              # QLearningPolicy (epsilon-greedy, 3-action)
│   └── train_rl.py               # DQN 학습 루프 + Reward Shaping + 체크포인트 저장
│
├── mip/
│   └── solve_mip.py              # pulp/CBC 기반 매 스텝 MILP 배정
│
├── ga/
│   ├── ga_policy.py              # GA 정책 (7-gene chromosome)
│   ├── train_ga.py               # GA 학습 루프 (POP=30, GEN=50)
│   └── best_genes_2stage.json    # 최적 유전자 저장
│
├── viz/
│   ├── index2d.html              # Canvas 기반 2D 뷰어 (정책 비교)
│   ├── 20260531_004/             # 최신 실험 결과 (Lane-mode 3-action)
│   │   ├── sim_2stage_*.json     # 7개 정책별 시뮬레이션 JSON (seed=42)
│   │   └── benchmark_2stage_8door.json  # 20 에피소드 집계 벤치마크
│   └── (이전 실험 디렉토리...)
│
├── checkpoints_2stage_8door/
│   ├── weights_final.npz         # RL 학습 가중치 (2000 에피소드)
│   └── weights_ep*.npz           # 체크포인트 (100ep 단위)
│
└── run_all_experiments_2stage.py # 전체 실험 파이프라인 (RL학습 + GA학습 + 벤치마크)
```

---

## 실행 방법

### 요구 사항

```bash
python >= 3.8
numpy
pulp   # MILP 솔버 (pip install pulp)
```

### 전체 실험 파이프라인

```bash
# RL 학습(2000ep) + GA 학습(50gen) + 전 정책 벤치마크(20ep) → viz/YYYYMMDD_NNN/ 저장
python run_all_experiments_2stage.py
```

결과 파일:
- `viz/YYYYMMDD_NNN/sim_2stage_{policy}.json` — 각 정책 시뮬레이션 JSON (7개)
- `viz/YYYYMMDD_NNN/benchmark_2stage_8door.json` — 20 에피소드 집계 통계
- `checkpoints_2stage_8door/weights_final.npz` — RL 최종 가중치
- `ga/best_genes_2stage.json` — GA 최적 유전자

### 2D 시각화

```bash
open viz/index2d.html   # macOS
```

드롭다운에서 정책별 JSON을 선택해 재생. 우측 상단 벤치마크 버튼으로 20ep 통계 비교.

### 환경 직접 사용

```python
from env.crossdock_env import CrossDockEnv, DEFAULT_CONFIG
from env.policies import HeuristicPriorityPolicy

cfg = {
    **DEFAULT_CONFIG,
    "num_inbound_doors": 3,
    "num_outbound_doors": 3,
    "buffer_capacity": 80.0,
    "enable_disruptions": True,
    "disruption_door_failure": True,
}
env = CrossDockEnv(config=cfg, seed=42)
policies = [HeuristicPriorityPolicy() for _ in range(env.num_lanes)]

obs_list = env.reset()
while True:
    actions = [policies[k].act(obs_list[k], env.num_inbound_doors)
               for k in range(env.num_lanes)]
    obs_list, rewards, done, info = env.step(actions)
    if done:
        m = info["metrics"]
        print(f"Ticks: {env.t}  Throughput: {m['total_throughput']:.1f} CBM"
              f"  Empty Deps: {m['empty_departures']}")
        break
```

---

## 출력 메트릭

| 메트릭 | 설명 | 방향 |
|---|---|---|
| `total_ticks` | 에피소드 총 소요 시간 (모든 화물 처리 완료까지) | ↓ |
| `total_throughput` | 아웃바운드로 출발한 총 화물량 (CBM) | ↑ |
| `avg_fill_rate` | 출발 아웃바운드 도크 평균 탑재율 | ↑ |
| `outbound_departures` | 총 아웃바운드 출발 횟수 | — |
| `empty_departures` | 탑재율 10% 미만으로 출발한 횟수 | ↓ |
| `buffer_overflow_count` | 버퍼 초과 발생 횟수 | ↓ |
| `door_utilization` | 인바운드 도어 평균 점유율 | ↑ |
| `outbound_door_utilization` | 아웃바운드 도크 평균 점유율 | ↑ |
| `avg_dwell_time` | 트럭 도착~하역 완료까지 평균 대기 시간 | ↓ |
| `disruption_door_failures` | 도어 고장 발생 횟수 | — |

---

## DQN 상태 · 보상 · 손실 함수

### 상태 (State)

관측 벡터 크기 = **12** (9 고정 + D=3 도어 매칭)

$$s = \left[\, \underbrace{o_0,\, o_1,\, \ldots,\, o_8}_{\text{9개 공유 정보}},\; \underbrace{m_0,\, m_1,\, m_2}_{\text{도어 매칭}}\, \right] \in \mathbb{R}^{12}$$

| 인덱스 | 변수 | 의미 | 정규화 분모 |
|---|---|---|---|
| 0 | $o_0$ | 내 레인 화물량 (CBM) | 50 |
| 1 | $o_1$ | 레인 혼잡도 | 1 |
| 2 | $o_2$ | 아웃바운드 도크 탑재율 (0이면 도크 없음) | 1 |
| 3 | $o_3$ | 아웃바운드 출발까지 남은 스텝 | 28 |
| 4 | $o_4$ | 버퍼 충전율 ∈ [0, 2] | 2 |
| 5 | $o_5$ | 유휴 인바운드 도어 수 | 10 |
| 6 | $o_6$ | 현재 대기 트럭 수 | 200 |
| 7 | $o_7$ | 미도착 스케줄 트럭 수 | 300 |
| 8 | $o_8$ | 유휴 아웃바운드 도크 수 | 10 |
| 9–11 | $m_0, m_1, m_2$ | 도어별 매칭 점수 (내 레인 화물 비율) | 1 (이미 정규화) |

$$\hat{s}_i = \frac{s_i}{c_i + 10^{-8}}$$

### 보상 (Reward)

**① 환경 기본 보상**

$$R_{\text{team}} = \text{이번 스텝 전체 출발 화물량} - 1.0 \times \text{빈 출발 수}$$

$$R_{\text{local}} = \text{내 레인 출발 화물량}$$

$$R_{\text{env}} = 0.7\, R_{\text{team}} + 0.3\, R_{\text{local}} \;+\; P_{\text{overflow}}$$

$$P_{\text{overflow}} = -1.0 - 0.3 \times \text{overflow 화물량} \quad \text{(버퍼 초과 시)}$$

**② Reward Shaping** (학습 가속용 보너스)

| 상황 | 조건 | 선택 행동 | 보너스 |
|---|---|---|---|
| 도크 없음 | $o_0 > 0$ AND $o_2 = 0$ | $a = 2$ | $+0.8$ |
| | | $a \neq 2$ | $-0.5$ |
| 인바운드 가능 + 버퍼 여유 | $o_5 > 0$ AND $o_6 > 0$ AND $o_4 \leq 1.5$ | $a = 1$ | $+0.4$ |
| | | $a \neq 1$ | $-0.3$ |
| 인바운드 가능 + 버퍼 포화 | $o_4 > 1.5$ | $a = 1$ | $-1.0$ |

$$R = R_{\text{env}} + b$$

### 손실 함수 (Loss)

**TD 타깃 (Bellman Equation)**

$$y_i = r_i + \gamma \cdot \max_{a'} Q^{-}(\hat{s}_i',\, a') \cdot (1 - d_i)$$

**MSE TD Loss** (선택한 행동에만 역전파, stop-gradient)

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \bigl(\, y_i - Q(\hat{s}_i,\, a_i) \,\bigr)^2$$

$$\frac{\partial \mathcal{L}}{\partial Q_{i,a'}} = \begin{cases} -\dfrac{2\,\delta_i}{B} & a' = a_i \\ 0 & a' \neq a_i \end{cases}, \quad \delta_i = y_i - Q(\hat{s}_i, a_i)$$

| 기호 | 의미 | 값 |
|---|---|---|
| $B$ | 배치 크기 | 64 |
| $\gamma$ | 할인율 | 0.99 |
| $Q^-$ | Target Network | 50 에피소드마다 $\theta^- \leftarrow \theta$ |
| $d_i$ | done 플래그 | 1 (종료) / 0 (진행 중) |
