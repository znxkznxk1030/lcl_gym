# Cross-Dock MARL Simulator

크로스도킹(Cross-Docking) 물류 환경을 다중 에이전트 강화학습(MARL)용으로 구현한 시뮬레이터입니다.

> **크로스도킹이란?** 입고된 화물을 창고에 보관하지 않고, 즉시 목적지별 레인으로 분류해 출고하는 물류 방식입니다.

---

## 환경 구조 (2-Stage)

### 스테이지 구분

```
[Stage 1] 인바운드 하역
  Inbound Truck → Inbound Door (8개) → Buffer (무제한)

[Stage 2] 아웃바운드 동적 배정
  Buffer → Lane Queue (5개) → Outbound Door (8개) → Outbound Truck
```

- **Stage 1**: 에이전트가 인바운드 트럭 도어 배정을 결정 (action 0/1)
- **Stage 2**: 매 스텝 레인 큐 volume 기준으로 아웃바운드 도어에 동적 배정 (그리디, 중복 배정 없음)

### 주요 엔티티

| 엔티티 | 설명 |
|---|---|
| `Truck` (Inbound) | **2~3개 목적지 화물이 혼재**된 인바운드 트럭. 스케줄된 도착 시간에 등장 |
| `OutboundTruck` | **목적지 1개 전용** 아웃바운드 트럭. 아웃바운드 도어에서 화물 적재 후 타이머 만료 시 출발 |
| `Door` | 인바운드 트럭이 하역하는 입고 도어 (처리 중엔 점유됨) |
| `Lane` | 목적지별 레인. 각각 하나의 **에이전트**에 해당 |
| `Buffer` | 화물이 레인으로 이동하기 전 대기하는 공유 스테이징 공간 **(용량 무제한)** |

### 기본 설정값 (8-Door 2-Stage)

```yaml
num_lanes: 5                  # 에이전트 수
num_inbound_doors: 8          # 입고 도어 수
num_outbound_doors: 8         # 출고 도어 수
buffer_capacity: 1e9          # 버퍼 용량 (사실상 무제한)
episode_length: 100           # 에피소드 길이 (타임스텝)

# 스케줄 기반 입고
arrival_count_min: 50
arrival_count_max: 70
arrival_pattern: "clustered"
arrival_cluster_count: 4

# 아웃바운드 출발 타이머
dispatch_interval_min: 12
dispatch_interval_max: 28
outbound_capacity: 15.0       # 아웃바운드 트럭 최대 적재량 (CBM)
```

---

## 에이전트 행동 / 관측 / 보상

### 행동 공간 (Action Space)

각 에이전트(레인)는 매 스텝 이진 결정을 내립니다.

```
0 → 아무것도 안 함 (skip)
1 → 트럭 요청 — 유휴 도어에 배정 요청
```

여러 에이전트가 동시에 `1`을 선택하면 **긴급도(아웃바운드 출발 임박 순) 기준**으로 유휴 도어 수만큼 병렬 배정됩니다.

### 관측 벡터 (크기 = 9)

```python
obs = [
    lane_queue,              # 0: 레인 현재 화물 적재량 (CBM)
    lane_congestion,         # 1: 혼잡도 (0~1 정규화)
    outbound_fill_rate,      # 2: 아웃바운드 도어 현재 탑재율 (0~1)
    outbound_departure_in,   # 3: 아웃바운드 출발까지 남은 타임스텝
    buffer,                  # 4: 버퍼 현재 적재량 (CBM, 무제한 환경)
    idle_inbound_doors,      # 5: 현재 유휴 인바운드 도어 수
    waiting_trucks,          # 6: 도착해서 대기 중인 트럭 수
    scheduled_trucks,        # 7: 아직 도착 전 스케줄 트럭 수
    idle_outbound_doors,     # 8: 현재 유휴 아웃바운드 도어 수
]
```

### 보상 구조

```python
R_team  = 이번_스텝_출발_화물량 - 2.0 × 빈_출발
R_local = 내_레인_출발_화물량 - 0.1 × 혼잡도

R_final = 0.7 × R_team + 0.3 × R_local   # 팀:개인 = 7:3
```

> `빈_출발`: 탑재율(fill_rate) < 10%인 채로 출발한 아웃바운드 트럭 (overflow 패널티 없음)

---

## 파일 구성

```
lcl_gym/
├── env/
│   ├── entities.py               # Truck, OutboundTruck, Door, Lane 데이터 클래스
│   ├── crossdock_env.py          # CrossDockEnv 메인 환경 (2-Stage)
│   └── policies.py               # 베이스라인 정책 4종
│
├── rl/
│   ├── networks.py               # numpy 2층 MLP (forward, Adam 역전파)
│   ├── replay_buffer.py          # 경험 리플레이 버퍼
│   ├── rl_policy.py              # QLearningPolicy (epsilon-greedy)
│   └── train_rl.py               # DQN 학습 루프 + 체크포인트 저장
│
├── mip/
│   └── solve_mip.py              # pulp/CBC 기반 매 스텝 MILP 배정
│
├── ga/
│   └── ga_policy.py              # 유전 알고리즘 정책 (6-gene chromosome)
│
├── viz/
│   ├── export_simulation.py      # 에피소드 → JSON 익스포트
│   ├── index.html                # Three.js 기반 3D 뷰어
│   ├── index2d.html              # Canvas 기반 2D 뷰어
│   ├── sim_2stage_random.json    # 7개 정책별 시뮬레이션 JSON
│   ├── sim_2stage_fifo.json
│   ├── sim_2stage_greedy.json
│   ├── sim_2stage_heuristic.json
│   ├── sim_2stage_mip.json
│   ├── sim_2stage_rl.json
│   ├── sim_2stage_ga.json
│   └── benchmark_2stage_8door.json  # 20 에피소드 집계 벤치마크
│
├── checkpoints_2stage_8door/
│   └── weights_final.npz         # RL 학습 가중치 (2000 에피소드)
│
└── run_all_experiments_2stage.py # 전체 실험 파이프라인 (학습 + 벤치마크)
```

---

## 정책 설명

### 베이스라인 4종

| 정책 | 설명 |
|---|---|
| `RandomPolicy` | 매 스텝 50% 확률로 트럭 요청 |
| `FIFOPolicy` | 대기 트럭과 유휴 도어가 있으면 항상 요청 |
| `GreedyPolicy` | 내 레인으로 오는 화물이 있는 트럭이 있을 때 요청 |
| `HeuristicPriorityPolicy` | 긴급도 + 매칭도 - 혼잡도 종합 점수가 임계값 이상일 때 요청 |

### MILP (Mixed Integer Linear Programming)

매 스텝 `maximize Σ x_{j,i} · score_j` 를 CBC 솔버로 풀어 트럭-도어 배정을 최적화합니다.

- `score_j` = Σ_k v_{j,k} / (departure_timer_k + 1) — 긴급도 가중 화물량
- 트럭·도어 각각 단일 배정 제약
- 평균 ~25ms/스텝

### RL (IQL + Parameter Sharing DQN)

numpy 기반 2층 MLP를 5개 에이전트가 공유해 학습합니다.

```
입력(9) → Linear(64) → ReLU → Linear(2) → Q값 {Q_skip, Q_request}
```

- **학습**: 2000 에피소드, lr=0.0005, γ=0.99, ε: 1.0→0.05
- **Target Network**: 50 에피소드마다 동기화
- **리플레이 버퍼**: 용량 10,000, 배치 64
- 가중치 저장: `checkpoints_2stage_8door/weights_final.npz`

### GA (Genetic Algorithm)

6개 유전자로 에이전트의 의사결정 임계값을 진화시킵니다.

```
gene[0] = queue_thresh      # 레인 큐 임계값
gene[1] = congestion_thresh # 혼잡도 임계값
gene[2] = fill_rate_thresh  # 탑재율 임계값
gene[3] = timer_thresh      # 출발 타이머 임계값
gene[4] = buf_thresh        # 버퍼 적재량 임계값
gene[5] = idle_door_thresh  # 유휴 도어 임계값
```

- **설정**: POP=50, GEN=100, 평가 에피소드=8회 평균
- 유전자 저장: `ga/best_genes_2stage.json`

---

## 실험 결과 (8-Door 2-Stage, 20 에피소드)

### 환경 설정

- 인바운드 도어: **8개**, 아웃바운드 도어: **8개**, 레인(에이전트): **5개**
- 인바운드 트럭: 50~70대/에피소드 (clustered, 4 batch)
- 버퍼 용량: **무제한** (overflow 없음)
- RL: 2000 에피소드 학습 / GA: pop=50, gen=100

### 처리량 · 탑재율 비교

| 정책 | 처리량 (CBM) | 탑재율 (%) | 빈 출발 | 오버플로우 | In-Door 활용률 | Out-Door 활용률 |
|---|---:|---:|---:|---:|---:|---:|
| Random    | 345.1 ± 27.8 | 87.7 ± 4.9 | 2.4 ± 1.2 | 0 | 91.9% | 62.8% |
| FIFO      | 337.2 ± 24.5 | 87.0 ± 5.2 | 2.5 ± 1.2 | 0 | 86.1% | 62.9% |
| Greedy    | 335.9 ± 25.2 | 87.0 ± 5.3 | 2.5 ± 1.2 | 0 | 86.0% | 62.9% |
| Heuristic | 339.4 ± 25.9 | 87.1 ± 5.1 | 2.5 ± 1.2 | 0 | 55.5% | 62.9% |
| MILP      | 342.3 ± 24.2 | 87.4 ± 5.2 | 2.1 ± 1.2 | 0 | 81.6% | 62.9% |
| **RL**    | **347.2 ± 40.7** | 87.3 ± 5.4 | 2.5 ± 1.2 | 0 | **89.8%** | 62.8% |
| **GA**    | **349.4 ± 27.7** | **87.4 ± 5.2** | 2.5 ± 1.2 | 0 | 82.5% | 62.8% |

> 오버플로우는 버퍼 무제한으로 모든 정책에서 0

### 버퍼 상태 통계 (20 에피소드 평균)

| 정책 | 탑재율 (mean ± std) | Buffer Peak (CBM) | Buffer Mean (CBM) |
|---|---:|---:|---:|
| Random    | 87.7% ± 4.9% | ~180 | ~82 |
| FIFO      | 87.0% ± 5.2% | ~180 | ~80 |
| Greedy    | 87.0% ± 5.3% | ~180 | ~80 |
| Heuristic | 87.1% ± 5.1% | ~180 | ~80 |
| MILP      | 87.4% ± 5.2% | ~180 | ~80 |
| RL        | 87.3% ± 5.4% | ~180 | ~82 |
| GA        | 87.4% ± 5.2% | ~180 | ~80 |

> 버퍼 통계는 에피소드 내 시계열에서 추출 (viz JSON의 `buffer` 필드)

### 단일 에피소드 최고 기록

| 정책 | 처리량 (CBM) | 탑재율 |
|---|---:|---:|
| Random    | 353.0 | 90.5% |
| FIFO      | 367.0 | 90.6% |
| Greedy    | 367.0 | 90.6% |
| Heuristic | 352.0 | 90.3% |
| MILP      | 382.0 | 91.0% |
| RL        | **397.0** | **91.3%** |
| GA        | 367.0 | 90.6% |

---

## 아웃바운드 도어 수 비교: 5개 vs 8개

| 지표 | nOD=5 | nOD=8 |
|---|---:|---:|
| 평균 처리량 (random) | ~335 CBM | ~345 CBM |
| 평균 탑재율 | ~93% | ~87% |
| Out-Door 활용률 | ~95% | ~63% |
| 빈 출발 | ~1.0 | ~2.5 |

- **nOD=8**: 출고 도어 수가 레인(5개)보다 많아 일부 도어가 미달 적재 출발 → 탑재율 하락
- **nOD=5**: 도어=레인으로 1:1 매칭, 탑재율이 더 높으나 처리 병목 발생
- 처리량 기준으로는 8개 도어가 소폭 유리 (+10 CBM)

---

## 2D 시각화 (viz/index2d.html)

Canvas 기반 2D 뷰어로 에피소드 재생 및 정책별 비교가 가능합니다.

### 뷰어 구성

```
┌──────────────────────────────────────────────────┐
│ 헤더: 파일 선택 드롭다운 · Step · Policy         │
├───────────────────┬──────────────────────────────┤
│  HUD 패널         │  Canvas 2D 뷰포트             │
│ · Throughput      │                               │
│ · Fill Rate       │  [예정 트럭] (상단)           │
│ · Buffer (CBM)    │  [인바운드 도어 8개]          │
│ · Empty Dep       │  ────── BUFFER (무제한) ────  │
│ · In/Out DoorUtil │  [레인 큐 5개]                │
│                   │  [아웃바운드 도어 8개] (하단) │
│                   │  [출고 트럭 8슬롯]            │
├───────────────────┴──────────────────────────────┤
│ Timeline: ◀ ▶ Play · 슬라이더                    │
└──────────────────────────────────────────────────┘
```

### 드롭다운 정책 목록

```
2-Stage 8door
  ├── 2stage-random
  ├── 2stage-fifo
  ├── 2stage-greedy
  ├── 2stage-heuristic
  ├── 2stage-mip
  ├── 2stage-rl
  └── 2stage-ga
```

### 실행 방법

```bash
open viz/index2d.html   # macOS
```

JSON 파일을 드롭다운에서 선택하거나 파일 열기로 불러옵니다.

---

## 3D 시각화 (viz/index.html)

Three.js 기반 3D 뷰어입니다.

```bash
open viz/index.html   # macOS
```

| 키 / 마우스 | 동작 |
|---|---|
| `Space` | 재생 / 일시정지 |
| `←` / `→` | 이전 / 다음 스텝 |
| 마우스 드래그 | 카메라 회전 |
| 마우스 휠 | 줌 인 / 아웃 |

---

## 실행 방법

### 요구 사항

```bash
python >= 3.8
numpy
pulp          # MILP 솔버
```

### 전체 실험 파이프라인

```bash
# 베이스라인 + MILP + RL 학습 + GA 학습 → 벤치마크
python run_all_experiments_2stage.py
```

결과 파일:
- `viz/sim_2stage_{policy}.json` — 각 정책 시뮬레이션 JSON (7개)
- `viz/benchmark_2stage_8door.json` — 20 에피소드 집계 통계
- `checkpoints_2stage_8door/weights_final.npz` — RL 가중치
- `ga/best_genes_2stage.json` — GA 최적 유전자

### 환경 직접 사용

```python
from env.crossdock_env import CrossDockEnv
from env.policies import HeuristicPriorityPolicy

cfg = {"num_inbound_doors": 8, "num_outbound_doors": 8}
env = CrossDockEnv(config=cfg, seed=42)
policies = [HeuristicPriorityPolicy() for _ in range(env.num_lanes)]

obs = env.reset()
for t in range(env.episode_length):
    actions = [policies[k].act(obs[k], env.num_inbound_doors) for k in range(env.num_lanes)]
    obs, rewards, done, info = env.step(actions)
    if done:
        print(f"처리량: {info['metrics']['total_throughput']:.1f} CBM")
        break
```

---

## 출력 메트릭

| 메트릭 | 설명 | 방향 |
|---|---|---|
| `total_throughput` | 아웃바운드에 탑재된 총 화물량 (CBM) | ↑ |
| `avg_fill_rate` | 출발 아웃바운드 트럭 평균 탑재율 | ↑ |
| `outbound_departures` | 총 아웃바운드 출발 횟수 | — |
| `empty_departures` | 탑재율 10% 미만으로 출발한 횟수 | ↓ |
| `door_utilization` | 인바운드 도어 평균 점유율 | ↑ |
| `avg_dwell_time` | 트럭 도착 ~ 화물 처리까지 평균 대기 시간 | ↓ |
