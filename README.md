# Cross-Dock MARL Simulator

크로스도킹(Cross-Docking) 물류 환경을 다중 에이전트 강화학습(MARL)용으로 구현한 시뮬레이터입니다.

> **크로스도킹이란?** 입고된 화물을 창고에 보관하지 않고, 즉시 목적지별 레인으로 분류해 출고하는 물류 방식입니다.

---

## 환경 구조

### 주요 엔티티

| 엔티티 | 설명 |
|---|---|
| `Truck` (Inbound) | **2~3개 목적지 화물이 혼재**된 인바운드 트럭. 미리 스케줄된 도착 시간에 등장 |
| `OutboundTruck` | **목적지 1개 전용** 아웃바운드 트럭. 레인 큐에서 화물을 싣고 레인별 다른 타이머로 출발 |
| `Door` | 인바운드 트럭이 하역하는 입고 도어 (처리 중엔 점유됨) |
| `Lane` | 목적지별 레인. 각각 하나의 **에이전트**에 해당 |
| `Buffer` | 화물이 레인으로 이동하기 전 대기하는 공유 스테이징 공간 |

### 기본 설정값

```yaml
num_lanes: 5                  # 에이전트 수
num_inbound_doors: 3          # 입고 도어 수
buffer_capacity: 60           # 버퍼 최대 용량 (CBM)
episode_length: 100           # 에피소드 길이 (타임스텝)
max_door_processing: 10       # 도어 처리 최대 소요 시간 (1~10 균등 랜덤)

# 스케줄 기반 입고
arrival_count_min: 50         # 에피소드당 최소 인바운드 트럭 수
arrival_count_max: 70         # 에피소드당 최대 인바운드 트럭 수
arrival_pattern: "clustered"  # 배치(batch) 도착 패턴 ("uniform" | "clustered")
arrival_cluster_count: 4      # 배치 수 (에피소드 전반에 균등 배치)

# 스케줄 기반 출고 (레인마다 독립 타이머)
dispatch_interval_min: 12     # 아웃바운드 출발 주기 최솟값 (스텝)
dispatch_interval_max: 28     # 아웃바운드 출발 주기 최댓값 (스텝)

inbound_min_dest: 2           # 인바운드 트럭 최소 목적지 수
inbound_max_dest: 3           # 인바운드 트럭 최대 목적지 수
inbound_vol_min: 0.5          # 목적지당 최소 화물량 (CBM)
inbound_vol_max: 5.0          # 목적지당 최대 화물량 (CBM)
outbound_capacity: 15.0       # 아웃바운드 트럭 1대 최대 적재량 (CBM)
```

---

## 에이전트 행동 / 관측 / 보상

### 행동 공간 (Action Space)

각 에이전트(레인)는 매 스텝 이진 결정을 내립니다.

```
0 → 아무것도 안 함 (skip)
1 → 트럭 요청 — 유휴 도어에 배정 요청
```

여러 에이전트가 동시에 `1`을 선택하면, **긴급도(아웃바운드 출발 임박 순) 기준**으로 유휴 도어 수만큼 병렬 배정됩니다. 도어 3개가 유휴 상태이고 3개 이상의 레인이 요청하면 **3개 도어 동시 처리**가 가능합니다.

### 관측 벡터 (Observation, 에이전트별, 크기 = 8 + num_doors)

```python
obs = [
    lane_queue,              # 0: 레인 현재 화물 적재량 (CBM)
    lane_congestion,         # 1: 혼잡도 (0~1 정규화)
    outbound_fill_rate,      # 2: 아웃바운드 트럭 현재 탑재율 (0~1)
    outbound_departure_in,   # 3: 아웃바운드 출발까지 남은 타임스텝
    buffer_remaining,        # 4: 버퍼 여유 용량 (CBM)
    idle_inbound_doors,      # 5: 현재 유휴 도어 수
    waiting_trucks,          # 6: 도착해서 대기 중인 트럭 수
    scheduled_trucks,        # 7: 아직 도착 전 스케줄 트럭 수
    door_match_0,            # 8: 대기 트럭 중 내 레인 화물 최대 매칭도
    door_match_1,            # 9: (도어 1)
    door_match_2,            # 10: (도어 2)
]
```

> `door_match_i`: 유휴 도어 i에 대해, 대기 트럭 중 이 레인으로 오는 화물 비율의 최댓값. 도어가 점유 중이면 0.

### 보상 구조

```python
R_team  = 이번_스텝_출발_화물량 - 0.5 × 버퍼_초과 - 2.0 × 빈_출발
R_local = 내_레인_출발_화물량 - 0.1 × 혼잡도

R_final = 0.7 × R_team + 0.3 × R_local   # 팀:개인 = 7:3
```

> `빈_출발`: 탑재율(fill_rate) < 10%인 채로 출발한 아웃바운드 트럭

### 스텝 실행 순서

```
1. 스케줄된 트럭 중 arrival_time ≤ t인 트럭 대기열 이동
2. 도어 상태 갱신 (처리 완료된 트럭 방출)
3. 방출된 트럭의 화물 → 버퍼 → 레인 이동
4. 대기열에 새 트럭 추가
5. 에이전트 행동 수집 → 긴급도 순 정렬 → 유휴 도어에 병렬 배정
5.5. 레인 큐 → 아웃바운드 트럭 점진 적재 (매 스텝)
6. 아웃바운드 트럭 출발 처리 (타이머 만료 시)
7. 보상 계산
```

---

## 파일 구성

```
lcl_gym/
├── env/                          # 시뮬레이션 환경
│   ├── __init__.py
│   ├── entities.py               # Truck, OutboundTruck, Door, Lane 데이터 클래스
│   ├── crossdock_env.py          # CrossDockEnv 메인 환경
│   └── policies.py               # 베이스라인 정책 4종
│
├── rl/                           # 강화학습
│   ├── __init__.py
│   ├── networks.py               # numpy 2층 MLP (forward, Adam 역전파)
│   ├── replay_buffer.py          # 경험 리플레이 버퍼
│   ├── rl_policy.py              # QLearningPolicy (epsilon-greedy)
│   ├── train_rl.py               # DQN 학습 루프 + 체크포인트 저장
│   └── evaluate_rl.py            # 학습 곡선 요약 + 베이스라인 비교
│
├── viz/                          # 3D 시각화 도구
│   ├── export_simulation.py      # 에피소드 → JSON 익스포트 스크립트
│   ├── index.html                # Three.js 기반 3D 뷰어
│   └── simulation_data.json      # 기본 출력 JSON (export 결과)
│
├── run_simulation.py             # 베이스라인 벤치마크 실행
├── checkpoints/                  # 학습 가중치 및 로그 저장
└── README.md
```

---

## 3D 시각화 (viz/)

시뮬레이션 에피소드를 Three.js 기반 3D 뷰어로 재생할 수 있습니다.

### 사용 순서

**1단계 — JSON 생성**

```bash
# greedy 정책으로 에피소드 실행 → viz/simulation_data.json 저장
python viz/export_simulation.py

# 정책 / 시드 지정
python viz/export_simulation.py --policy heuristic --seed 7
python viz/export_simulation.py --policy rl
python viz/export_simulation.py --policy random --output viz/sim_random.json
```

사용 가능한 정책: `greedy` / `fifo` / `random` / `heuristic` / `rl`

**2단계 — 브라우저에서 뷰어 열기**

```bash
open viz/index.html   # macOS
```

JSON 파일을 파일 열기 버튼 또는 드래그 앤 드롭으로 불러옵니다.

### 뷰어 화면 구성

```
┌────────────────────────────────────────┐
│ Header: Policy · Seed · Steps          │
├───────────────────┬────────────────────┤
│ Step Info 패널    │   Metrics 패널     │
│ - 현재 스텝       │ - 총 처리량        │
│ - 버퍼 점유량     │ - 평균 탑재율      │
│ - 대기 트럭 수    │ - 버퍼 오버플로우  │
│ - 스케줄 트럭 수  │ - 도어 활용률      │
├───────────────────┴────────────────────┤
│          Three.js 3D 뷰포트            │
│  [예정 트럭] → [대기] → [도어] → [버퍼]│
│           → [레인 큐] → [아웃바운드]   │
├────────────────────────────────────────┤
│ Lanes 범례 (레인별 Q / 탑재% / 타이머) │
├────────────────────────────────────────┤
│ Timeline: ◀ ▶ Play · 슬라이더 · fps   │
└────────────────────────────────────────┘
```

### 조작 단축키

| 키 / 마우스 | 동작 |
|---|---|
| `Space` | 재생 / 일시정지 |
| `←` / `→` | 이전 / 다음 스텝 |
| `+` / `−` | 재생 속도 변경 |
| 마우스 드래그 | 카메라 회전 |
| 마우스 휠 | 줌 인 / 아웃 |

---

## 실행 방법

### 요구 사항

```bash
python >= 3.8
numpy
```

### 베이스라인 정책 비교

```bash
python run_simulation.py
```

### 환경 직접 사용

```python
from env.crossdock_env import CrossDockEnv
from env.policies import HeuristicPriorityPolicy

env = CrossDockEnv(seed=42)
policies = [HeuristicPriorityPolicy() for _ in range(env.num_lanes)]

obs = env.reset()
for t in range(env.episode_length):
    actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
    obs, rewards, done, info = env.step(actions)
    if done:
        print(f"처리량: {info['metrics']['total_throughput']:.1f}")
        break
```

---

## 베이스라인 정책 설명

| 정책 | 설명 |
|---|---|
| `RandomPolicy` | 매 스텝 50% 확률로 트럭 요청 |
| `FIFOPolicy` | 대기 트럭과 유휴 도어가 있으면 항상 요청 |
| `GreedyPolicy` | 내 레인으로 오는 화물이 있는 트럭이 있을 때 요청 |
| `HeuristicPriorityPolicy` | 긴급도 + 매칭도 - 혼잡도 종합 점수가 임계값 이상일 때 요청 |

---

## RL 학습 (IQL + Parameter Sharing DQN)

numpy만으로 구현된 DQN 기반 MARL 학습 파이프라인입니다.

### 알고리즘 구조

**IQL (Independent Q-Learning) + Parameter Sharing**

5개 에이전트가 동일한 가중치(NumpyMLP)를 공유해 학습합니다.

```
에이전트 1 ──┐
에이전트 2 ──┤
에이전트 3 ──┼──→ 공유 NumpyMLP → Q(obs, action) → {Q_skip, Q_request}
에이전트 4 ──┤
에이전트 5 ──┘
```

- 5개 에이전트의 경험이 동시에 같은 네트워크를 업데이트 → 데이터 효율 5배
- 평가 시에는 각 에이전트가 독립적으로 행동 (분산 실행)

### 네트워크 구조

```
입력(11) → Linear(64) → ReLU → Linear(2) → Q값 {Q_skip, Q_request}
```

- 역전파: 선택한 action 위치만 TD error로 gradient 계산
- 옵티마이저: Adam (β1=0.9, β2=0.999)
- Target Network: 50 에피소드마다 동기화

### Reward Shaping

```python
R_shaped = R_env
         + 1.0 × door_match    # 요청 시 대기 트럭의 최대 매칭도 보너스
         - 0.1 × congestion    # 혼잡 억제
```

### 학습 실행

```bash
python rl/train_rl.py
python rl/train_rl.py --episodes 2000 --lr 5e-4
python rl/train_rl.py --no-share   # 에이전트별 독립 가중치
```

---

## 정책 비교 결과 (2026-04-28, 30 에피소드)

### 환경 설정

- 에피소드당 인바운드 트럭: **50~70대** (clustered 패턴, 4개 배치)
- 아웃바운드 출발 주기: 레인별 독립 타이머 **12~28 스텝**
- 인바운드 도어: 3개 / 에피소드 길이: 100 스텝

### 결과

| Policy | Throughput | AvgFillRate | Overflow | DoorUtil | DwellTime |
|---|---:|---:|---:|---:|---:|
| Random | 273.5 ± 25.5 | 73.6% ± 7.3% | 0.1 ± 0.6 | 90.6% | 12.8 |
| FIFO | 253.8 ± 25.0 | 68.9% ± 6.1% | 0.0 ± 0.0 | 81.7% | 16.2 |
| Greedy | 253.8 ± 25.0 | 68.9% ± 6.1% | 0.0 ± 0.0 | 81.7% | 16.2 |
| Heuristic | 253.3 ± 23.7 | 68.8% ± 6.1% | 0.0 ± 0.0 | 81.6% | 16.3 |
| **RL (DQN)** | **246.8 ± 19.0** | **68.0% ± 6.6%** | **0.0 ± 0.0** | **80.5%** | **18.3** |

> RL 학습: 2000 에피소드, lr=1e-3, shared weights, seed=42

### 분석

**RL vs Heuristic**

| 지표 | Heuristic | RL (DQN) | 차이 |
|---|---:|---:|---:|
| Throughput | 253.3 | 246.8 | **-6.5 (-2.6%)** |
| AvgFillRate | 68.8% | 68.0% | -0.8%p |
| Overflow | 0.0 | 0.0 | 동일 |
| DoorUtil | 81.6% | 80.5% | -1.1%p |
| DwellTime | 16.3 | 18.3 | +2.0 스텝 |

RL은 현재 Heuristic 대비 처리량이 소폭 낮고 체류 시간이 길어요. 아직 수렴이 불완전한 상태로, 2000 에피소드로는 충분하지 않을 수 있습니다.

**Random이 FIFO/Greedy/Heuristic보다 처리량이 높은 이유**

결정론적 정책(FIFO/Greedy)은 5개 레인이 모두 동시에 요청하거나 동시에 스킵합니다. 이로 인해 긴급도가 낮은 레인이 반복적으로 배정 기회를 잃어 **부하 불균형**이 발생합니다. Random은 각 레인이 50% 확률로 독립 결정하므로 어떤 스텝에서도 다양한 레인 조합이 요청하게 되어 도어 이용률이 더 높아집니다 (81% → 90%).

**현재 MARL 과제**

파라미터 공유(shared weights) 구조에서 5개 에이전트가 동일 관측을 보면 동일한 행동을 냅니다. **에이전트 간 협력 (서로 다른 타이밍에 요청)** 을 학습하려면 레인 ID 임베딩 추가 또는 QMIX 같은 중앙집중식 학습 방식이 필요합니다.

### 개선 방향

| 방향 | 기대 효과 |
|---|---|
| 관측에 레인 ID one-hot 추가 | 에이전트 간 행동 다양화 → 도어 이용률 개선 |
| 에피소드 수 증가 (5000+) | RL 수렴 개선 |
| QMIX / MAPPO | 팀 보상 분해로 협력 학습 |
| TorchMLP 교체 | 더 큰 네트워크, GPU 활용 가능 |

---

## 출력 메트릭

| 메트릭 | 설명 | 방향 |
|---|---|---|
| `total_throughput` | 아웃바운드에 탑재된 총 화물량 (CBM) | ↑ |
| `total_fill_rate` | 아웃바운드 출발 탑재율 합산 | ↑ |
| `outbound_departures` | 총 아웃바운드 출발 횟수 | — |
| `empty_departures` | 탑재율 10% 미만으로 출발한 횟수 | ↓ |
| `buffer_overflow_count` | 버퍼 용량 초과 발생 횟수 | ↓ |
| `avg_dwell_time` | 트럭 도착 ~ 화물 처리까지 평균 대기 시간 | ↓ |
| `door_utilization` | 도어 평균 점유율 (0~1) | ↑ |
| `avg_fill_rate` | 출발 아웃바운드 트럭 평균 탑재율 | ↑ |
