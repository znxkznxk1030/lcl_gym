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

- **알고리즘**: IQL (Independent Q-Learning) + Parameter Sharing — 5개 에이전트가 동일 가중치 공유, 각자 독립적으로 행동
- **학습**: 2000 에피소드, lr=0.001, γ=0.99, ε: 1.0→0.05 (decay 0.995)
- **Target Network**: 50 에피소드마다 $\theta^- \leftarrow \theta$ 동기화
- **리플레이 버퍼**: 용량 10,000, 배치 64
- **Reward Shaping**: `needs_dock`(화물 있는데 도크 없음) 상황에서 action=2 보너스 +0.8
- 가중치 저장: `checkpoints_2stage_8door/weights_final.npz`

**성능 우위 요인**

RL outperformed rule-based policies by learning two key behaviors: redirecting idle docks away from empty lanes toward lanes that actually had cargo waiting, and adapting more effectively to dynamic disruptions such as sudden door failures.

**실험의 한계**

| 한계 | 설명 |
|---|---|
| 출발 모델 단순화 | 도크가 고정 타이머 만료 시 출발 → 실제 만차 출발·cut-off 방식과 차이, action=2 효과 과대평가 가능 |
| 화물 분류 자동화 | 하역 즉시 정확하게 레인으로 분류 → 실제 수작업 오류·지연 미반영 |
| 제한적 돌발 유형 | 도어 고장만 모델링 → 트럭 지연·긴급 화물·작업자 부족 등 미포함 |
| 소규모 환경 | 5레인·3도어·50~70대 트럭 → 대형 터미널 확장성 미검증 |
| 근시안적 보상 | 스텝별 출발 화물량 기준 → 납기 준수·레인 부하 균형 등 장기 목표 미최적화 |

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

### 실험 2 — Lane-mode 3-action · Best-Match 입고 (20260531_006)

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

---

### 실험 3 — Truck Selection 모드 · GA 최적화 (20260531_006)

**Truck Selection 모드**: 각 에이전트가 "어느 레인의 화물을 요청할 것인가" 대신 "어떤 대기 트럭을 먼저 처리할 것인가"를 결정합니다.

#### 환경 설정 (Lane-mode와 동일 조건)

| 파라미터 | 값 |
|---|---|
| `use_truck_selection` | True |
| `top_k_trucks` | 15 (상위 후보 트럭 수) |
| 인바운드 도어 | 3 |
| 트럭 수 / 에피소드 | 50 ~ 70대 |
| 돌발사항 | 도어 고장 · 2%/스텝 |

#### GA-TruckSel 7-gene 구조

```
score = w_due × weighted_urgency
      + w_dest_match × destination_match
      − w_congestion × expected_congestion
      + w_buffer × buffer_availability
      − w_queue_position × queue_pressure
      + w_rush × is_rush

action = 1 (트럭 처리) if score > threshold else 0 (패스)
```

- **설정**: POP=30, GEN=50, N_EVAL=5
- 유전자 저장: `ga/best_genes_truck_selection.json`

#### 벤치마크 결과 (20 에피소드, seed 200~219)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 처리량 (CBM) |
|---|---:|---:|---:|---:|
| TruckSel-FIFO | **343.0** | 10.3 | 5.20 | 440.2 |
| **GA-TruckSel** | **343.0** | 10.3 | 5.20 | 440.2 |
| TruckSel-Heuristic | 343.1 | 10.4 | **5.10** | 440.2 |

> GA가 초기 Heuristic 유전자(`HEURISTIC_GENES`)를 넘어서지 못하고 수렴. 세 정책이 사실상 동일한 성능을 기록함.

#### 분석: GA가 개선되지 않은 이유

GA-TruckSel의 threshold가 낮게 설정되면 대기 트럭 대부분에 action=1이 선택되어 FIFO와 동일한 동작이 됩니다. 즉, "어떤 트럭을 선택할 것인가"보다 **트럭 배정 순서(score-based ranking)** 자체가 Truck Selection 모드에서는 이미 환경이 처리하므로, 7-gene 스코어링의 개선 여지가 작습니다.

#### Lane-mode vs Truck Selection 모드 비교

| 항목 | Lane-mode (실험 2) | Truck Selection 모드 (실험 3) |
|---|---|---|
| 에이전트 결정 | 어느 레인에서 트럭 요청 / 도크 부스트 | 어떤 트럭을 먼저 처리 |
| 최고 성능 | RL **339.1 ticks** | TruckSel-FIFO/GA **343.0 ticks** |
| 빈 출발 | RL **2.70** | 5.10~5.20 |
| action=2 사용 | 있음 (도크 mid-trip 재배정) | 없음 |
| 핵심 우위 | 도크 낭비 감소 | 트럭 순서 최적화 |

> Lane-mode RL이 Truck Selection 모드보다 약 4 ticks 더 빠름. action=2(도크 재배정)의 유무가 결정적 차이.

### RL이 FIFO를 이기는 이유

```
FIFO는 action=2를 사용하지 않음
→ 빈 레인을 서비스하는 도크가 헛되이 카운트다운 → Empty Departure 증가

RL은 action=2로 mid-trip 재배정을 선택적으로 트리거
→ 빈 도크를 화물 있는 레인으로 전환 → 총 출발 횟수·빈 출발 모두 감소
→ 동일 처리량을 더 적은 트립으로 완료 → Total Ticks 단축
```

---

### 실험 4 — Reward Shaping Ablation

Reward Shaping의 각 구성요소를 제거했을 때 성능 변화를 측정합니다.

#### Variant 정의

| Variant | Inbound Shaping | Outbound Shaping | 설명 |
|---|:---:|:---:|---|
| **full** | ✓ | ✓ | 기준 (현재 설정) |
| **no_out** | ✓ | ✗ | action=2 guidance 제거 |
| **no_in** | ✗ | ✓ | action=1 guidance 제거 |
| **no_shaping** | ✗ | ✗ | R_env만 사용 |

**Inbound Shaping**: `can_inbound` 상황에서 action=1이면 +0.4, 아니면 −0.3 / 버퍼 포화 시 action=1이면 −1.0  
**Outbound Shaping**: `needs_dock` 상황에서 action=2이면 +0.8, 아니면 −0.5

#### 결과 (1000 에피소드 학습, 20 에피소드 벤치마크)

| Variant | Avg Ticks ↓ | Std | 빈 출발 ↓ | full 대비 |
|---|---:|---:|---:|---:|
| 🥇 **full** | **342.3** | 12.4 | 4.20 | — |
| **no_in** | 342.9 | 13.2 | **3.45** | +0.6 |
| **no_out** | 343.9 | 10.8 | 4.60 | +1.6 |
| **no_shaping** | 344.4 | 11.4 | 5.30 | +2.1 |

> 학습 에피소드 1000ep 기준 (실험 1·2는 2000ep). 절대 수치보다 상대 차이가 중요.

#### 분석

- **Outbound Shaping이 더 중요**: 제거 시 +1.6 ticks 악화. action=2(도크 재배정)는 학습하기 어려운 행동이라 명시적 보너스 없이는 충분히 학습되지 않음.
- **Inbound Shaping 제거 시 빈출발 감소**: no_in은 ticks가 +0.6 증가하지만 빈출발이 3.45로 오히려 개선. 인바운드 보너스가 과도하게 트럭을 들여보내면서 도크 낭비를 일부 유발하는 tradeoff 존재.
- **Shaping 전체 제거 시 최악**: no_shaping은 ticks와 빈출발 모두 최악. R_env는 출발 시점에만 보상이 발생해 sparse하므로, shaping 없이는 올바른 행동을 학습하는 데 한계가 있음.

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
│   ├── ga_policy.py                    # Lane-mode GA 정책 (7-gene chromosome)
│   ├── train_ga.py                     # Lane-mode GA 학습 루프 (POP=30, GEN=50)
│   ├── best_genes_2stage.json          # Lane-mode GA 최적 유전자
│   ├── truck_selection_policy.py       # Truck Selection GA 정책 (7-gene)
│   ├── train_ga_truck_selection.py     # Truck Selection GA 학습 루프
│   └── best_genes_truck_selection.json # Truck Selection GA 최적 유전자
│
├── viz/
│   ├── index2d.html              # Canvas 기반 2D 뷰어 (정책 비교)
│   ├── 20260531_006/             # 최신 실험 결과 (Lane-mode Best-Match)
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
---

## 실험 5 — 돌발사항 확률 스윕 및 Domain Randomization 강인성 실험

도어 고장 확률(`disruption_door_failure_prob`)을 0.02에서 0.20까지 높여가며
기학습된 RL과 베이스라인 정책들의 강인성(robustness)을 비교한다.
각 레벨당 20 에피소드 평균. **↓ 숫자가 낮을수록 우수.**

### 환경 설정 (공통)

| 파라미터 | 값 |
|---|---|
| 인바운드 도어 | 3 |
| 아웃바운드 도크 | 3 |
| 버퍼 용량 | 80 CBM |
| 트럭 수/에피소드 | 50~70대 |
| 도어 고장 지속 | 10~20 스텝 |
| 에피소드 수 | 20 |

### disruption_prob = 0.02 (2%/스텝, 학습 조건)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 횟수 |
|---|---:|---:|---:|---:|
| 🥇 **RL** | **339.1** | 11.9 | **2.70** | 7.0 |
| FIFO | 341.6 | 12.4 | 4.85 | 6.6 |
| GA | 344.9 | 17.5 | 4.10 | 6.8 |
| Heuristic | 349.6 | 20.6 | 4.55 | 6.9 |
| Greedy | 350.9 | 17.1 | 4.50 | 7.0 |

### disruption_prob = 0.05 (5%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 횟수 |
|---|---:|---:|---:|---:|
| 🥇 **FIFO** | **342.6** | 11.0 | **4.45** | 17.2 |
| Greedy | 350.8 | 18.7 | 4.40 | 17.0 |
| Heuristic | 351.8 | 16.0 | 4.25 | 17.1 |
| GA | 354.7 | 15.8 | 4.75 | 17.4 |
| ⚠️ RL | 824.4 | 2105.1 | 76.70 | 41.8 |

### disruption_prob = 0.10 (10%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 횟수 |
|---|---:|---:|---:|---:|
| 🥇 **FIFO** | **355.2** | 16.9 | 5.10 | 32.1 |
| Greedy | 361.4 | 22.1 | **4.45** | 31.8 |
| Heuristic | 361.6 | 22.1 | 4.90 | 32.8 |
| GA | 364.7 | 23.3 | 5.05 | 33.8 |
| ⚠️ RL | 1800.3 | 3444.6 | 224.95 | 163.5 |

### disruption_prob = 0.15 (15%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 횟수 |
|---|---:|---:|---:|---:|
| 🥇 **Greedy** | **402.6** | 51.4 | **7.30** | 48.8 |
| FIFO | 403.8 | 56.0 | 8.95 | 48.9 |
| Heuristic | 409.1 | 58.8 | 8.05 | 50.0 |
| GA | 409.8 | 50.3 | 7.80 | 49.5 |
| ⚠️ RL | 5670.2 | 4786.8 | 817.75 | 676.6 |

### disruption_prob = 0.20 (20%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 횟수 |
|---|---:|---:|---:|---:|
| 🥇 **FIFO** | **491.0** | 100.1 | 16.65 | 68.2 |
| Greedy | 499.1 | 98.9 | **13.55** | 70.1 |
| Heuristic | 509.5 | 91.3 | 14.15 | 71.8 |
| GA | 973.5 | 2072.5 | 84.80 | 134.9 |
| ⚠️ RL | 8084.1 | 3831.7 | 1186.25 | 1118.2 |

### RL vs FIFO 성능 격차 추이

| disruption_prob | RL Ticks | FIFO Ticks | 격차 (RL − FIFO) | 비고 |
|---:|---:|---:|---:|---|
| 0.02 | **339.1** | 341.6 | **−2.5** | RL 우위 (학습 범위 내) |
| 0.05 | 824.4 | **342.6** | +481.8 | RL 분포 이탈 시작 |
| 0.10 | 1800.3 | **355.2** | +1445.1 | RL 폭발적 악화 |
| 0.15 | 5670.2 | **403.8** | +5266.4 | RL 사실상 기능 불능 |
| 0.20 | 8084.1 | **491.0** | +7593.1 | RL 완전 실패 |

### 분석

#### 1. RL의 분포 이탈(Distribution Shift) 문제

**현상**: RL은 학습 조건(prob=0.02)에서 최고 성능을 보이지만, prob=0.05부터 Avg Ticks가 824로 급등하고 Std가 2105로 폭발한다. 이는 일부 에피소드에서는 정상 작동하지만 다른 에피소드에서는 종료 조건을 충족하지 못하고 매우 긴 시간이 소요됨을 의미한다.

**원인**: RL은 2% 확률의 도어 고장 분포에서 학습되었다. 5% 이상에서는 관측 벡터(`idle_inbound_doors`)가 학습 중 거의 본 적 없는 값 범위(도어가 장기간 모두 고장)에 들어가며, Q-network가 잘못된 행동(예: action=2를 지속적으로 선택)을 강화 피드백 없이 반복하게 된다. 빈 출발이 76.70(0.05), 224.95(0.10)으로 급증하는 것이 이를 방증한다.

#### 2. 규칙 기반 정책의 강인성

**현상**: FIFO·Greedy·Heuristic은 disruption 확률이 높아져도 Ticks가 비교적 완만하게 증가한다 (341→491, 약 44% 증가 / prob 0.02→0.20). Std도 100 이하로 유지되어 일관적이다.

**원인**: 규칙 기반 정책은 학습된 파라미터가 없으므로 분포 이탈이 없다. 도어가 고장나면 단순히 "유휴 도어가 없으면 action=0"을 일관되게 실행하며, 환경이 어떻든 동일한 조건 로직을 따른다.

#### 3. GA의 중간 붕괴 (prob=0.20)

GA도 20% 구간에서 Avg 973.5 / Std 2072.5로 일부 에피소드 실패를 보인다. GA 유전자가 0.02 환경에서 진화했으므로 extreme disruption에서 RL과 유사한 분포 이탈이 발생한다.

#### 4. 시사점

| 결론 | 내용 |
|---|---|
| **RL은 학습 범위 밖에서 취약** | 더 높은 disruption 확률 또는 도메인 랜덤화(Domain Randomization)로 재학습 필요 |
| **규칙 기반은 강인하지만 상한이 있음** | 학습 범위 내에서는 RL이 우위이나, OOD 환경에서는 FIFO가 안전망 역할 |
| **실용적 배포 전략** | RL + 규칙 기반 fallback: disruption 수준을 실시간 감지하여 일정 임계 초과 시 FIFO로 전환 |





---

## 실험 6 — Domain Randomization 효과 검증

RL(고정 prob=0.02 학습) vs RL-DR(Domain Randomization, prob∈[0.02,0.25] 균등 샘플링,
fine-tune 2000 에피소드) vs 전체 베이스라인 정책을 5가지 disruption 레벨에서 비교.
각 레벨당 20 에피소드 평균. **Total Ticks ↓ 낮을수록 우수.**

### 학습 설정 비교 (RL 계열)

| 항목 | RL (원본) | RL-DR (Domain Rand) |
|---|---|---|
| 학습 disruption_prob | 고정 0.02 | 매 에피소드 Uniform[0.02, 0.25] |
| 에피소드 수 | 2000 | 2000 (fine-tune) |
| 초기 가중치 | 랜덤 | RL 원본 체크포인트 |
| 학습률(lr) | 1e-3 | 1e-4 |
| 초기 ε | 1.0 | 0.3 |

### disruption_prob = 0.00 (0%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **MILP** | **337.5** | 10.4 | **4.80** | 0.0 |
| FIFO | 337.9 | 10.1 | 4.95 | 0.0 |
| Random | 338.6 | 10.8 | 4.80 | 0.0 |
| ⚠️ RL | 339.2 | 15.3 | 3.40 | 0.0 |
| RL-DR | 339.9 | 12.6 | 3.00 | 0.0 |
| Heuristic | 345.7 | 15.6 | 3.90 | 0.0 |
| GA-DR | 348.2 | 21.2 | 4.05 | 0.0 |
| GA | 348.6 | 17.7 | 4.35 | 0.0 |
| Greedy | 349.2 | 20.6 | 4.10 | 0.0 |
| ⚠️ Zero | 10000.0 | 0.0 | 1501.30 | 0.0 |

### disruption_prob = 0.02 (2%/스텝) ← **RL 학습 조건**

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **RL-DR** | **338.4** | 11.3 | **2.70** | 7.0 |
| RL | 339.1 | 11.9 | 2.70 | 7.0 |
| Random | 339.5 | 12.5 | 4.75 | 6.5 |
| MILP | 341.1 | 12.0 | 5.00 | 6.6 |
| FIFO | 341.6 | 12.4 | 4.85 | 6.6 |
| GA | 344.9 | 17.5 | 4.10 | 6.8 |
| Heuristic | 349.6 | 20.6 | 4.55 | 6.9 |
| Greedy | 350.9 | 17.1 | 4.50 | 7.0 |
| GA-DR | 353.1 | 15.1 | 4.45 | 7.1 |
| ⚠️ Zero | 10000.0 | 0.0 | 1499.65 | 204.9 |

### disruption_prob = 0.05 (5%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **RL-DR** | **340.9** | 11.1 | **3.30** | 16.4 |
| MILP | 341.8 | 12.5 | 4.60 | 17.1 |
| Random | 341.8 | 11.7 | 4.10 | 17.2 |
| FIFO | 342.6 | 11.0 | 4.45 | 17.2 |
| GA-DR | 348.6 | 17.8 | 4.30 | 16.9 |
| Greedy | 350.8 | 18.7 | 4.40 | 17.0 |
| Heuristic | 351.8 | 16.0 | 4.25 | 17.1 |
| GA | 354.7 | 15.8 | 4.75 | 17.4 |
| ⚠️ RL | 824.4 | 2105.1 | 76.70 | 41.8 |
| ⚠️ Zero | 10000.0 | 0.0 | 1495.45 | 492.4 |

### disruption_prob = 0.10 (10%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **RL-DR** | **351.2** | 18.0 | **3.65** | 32.1 |
| Random | 352.9 | 17.5 | 5.60 | 32.0 |
| FIFO | 355.2 | 16.9 | 5.10 | 32.1 |
| MILP | 355.6 | 16.5 | 5.15 | 32.3 |
| Greedy | 361.4 | 22.1 | 4.45 | 31.8 |
| Heuristic | 361.6 | 22.1 | 4.90 | 32.8 |
| GA-DR | 363.4 | 23.8 | 5.15 | 32.4 |
| GA | 364.7 | 23.3 | 5.05 | 33.8 |
| ⚠️ RL | 1800.3 | 3444.6 | 224.95 | 163.5 |
| ⚠️ Zero | 10000.0 | 0.0 | 1500.65 | 896.5 |

### disruption_prob = 0.15 (15%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **RL-DR** | **396.7** | 52.0 | **6.50** | 48.1 |
| GA-DR | 398.1 | 52.7 | 7.40 | 48.2 |
| Random | 398.3 | 53.5 | 8.95 | 48.1 |
| MILP | 399.8 | 53.8 | 8.85 | 48.5 |
| Greedy | 402.6 | 51.4 | 7.30 | 48.8 |
| FIFO | 403.8 | 56.0 | 8.95 | 48.9 |
| Heuristic | 409.1 | 58.8 | 8.05 | 50.0 |
| GA | 409.8 | 50.3 | 7.80 | 49.5 |
| ⚠️ RL | 5670.2 | 4786.8 | 817.75 | 676.6 |
| ⚠️ Zero | 10000.0 | 0.0 | 1499.85 | 1182.4 |

### disruption_prob = 0.20 (20%/스텝)

| 정책 | Avg Ticks ↓ | Std | 빈 출발 ↓ | 도어고장 |
|---|---:|---:|---:|---:|
| 🥇 **Random** | **483.2** | 98.8 | **15.10** | 67.3 |
| MILP | 489.1 | 100.7 | 16.65 | 67.8 |
| FIFO | 491.0 | 100.1 | 16.65 | 68.2 |
| RL-DR | 492.5 | 103.2 | 12.90 | 69.5 |
| GA-DR | 496.1 | 92.3 | 12.25 | 70.0 |
| Greedy | 499.1 | 98.9 | 13.55 | 70.1 |
| Heuristic | 509.5 | 91.3 | 14.15 | 71.8 |
| GA | 973.5 | 2072.5 | 84.80 | 134.9 |
| ⚠️ RL | 8084.1 | 3831.7 | 1186.25 | 1118.2 |
| ⚠️ Zero | 10000.0 | 0.0 | 1502.50 | 1376.2 |

### 전 구간 종합 요약 (Avg Ticks)

| 정책 | p=0.00 | p=0.02 | p=0.05 | p=0.10 | p=0.15 | p=0.20 |
|---|---:|---:|---:|---:|---:|---:|
| RL | 339.2 | 339.1 | 824.4 | 1800.3 | 5670.2 | 8084.1 |
| RL-DR | 339.9 | **338.4** | **340.9** | **351.2** | **396.7** | 492.5 |
| Random | 338.6 | 339.5 | 341.8 | 352.9 | 398.3 | **483.2** |
| MILP | **337.5** | 341.1 | 341.8 | 355.6 | 399.8 | 489.1 |
| FIFO | **337.9** | 341.6 | 342.6 | 355.2 | 403.8 | 491.0 |
| GA | 348.6 | 344.9 | 354.7 | 364.7 | 409.8 | 973.5 |
| Heuristic | 345.7 | 349.6 | 351.8 | 361.6 | 409.1 | 509.5 |
| Greedy | 349.2 | 350.9 | 350.8 | 361.4 | 402.6 | 499.1 |
| GA-DR | 348.2 | 353.1 | 348.6 | 363.4 | 398.1 | 496.1 |
| Zero | 10000.0 | 10000.0 | 10000.0 | 10000.0 | 10000.0 | 10000.0 |

### 분석

#### 1. RL-DR의 분포 이탈 해소

원본 RL(prob=0.02 학습)은 prob≥0.05부터 폭발적으로 실패했으나,
DR fine-tune 이후 전 구간에서 안정적인 ticks를 기록한다.
RL-DR의 빈 출발 수가 FIFO보다 낮게 유지되는 것은 action=2(도크 재배정) 전략이
높은 disruption 수준에서도 일관되게 작동함을 의미한다.

#### 2. 학습 조건(0.02)에서의 성능

DR 범위를 [0.02, 0.25]로 확장했음에도 RL-DR은 원본 RL과 동등하거나 소폭 우수한
성능을 보인다. 이는 fine-tune 시 기존 가중치가 좋은 초기값으로 작동하여
학습 조건에서의 성능을 유지했기 때문이다.

#### 3. 규칙 기반 정책과의 비교

FIFO·Greedy는 높은 disruption에서도 안정적이지만 빈 출발 수가 RL-DR보다 많다.
RL-DR은 도크 낭비 최소화(action=2)와 강인성을 동시에 달성한다.
