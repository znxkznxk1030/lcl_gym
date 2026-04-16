# Cross-Dock MARL Simulator

크로스도킹(Cross-Docking) 물류 환경을 다중 에이전트 강화학습(MARL)용으로 구현한 시뮬레이터입니다.

> **크로스도킹이란?** 입고된 화물을 창고에 보관하지 않고, 즉시 목적지별 레인으로 분류해 출고하는 물류 방식입니다.

---

## 환경 구조

### 주요 엔티티

| 엔티티 | 설명 |
|---|---|
| `Truck` (Inbound) | **2~3개 목적지 화물이 혼재**된 인바운드 트럭. 도착 후 도어에 배정되어 하역 |
| `OutboundTruck` | **목적지 1개 전용** 아웃바운드 트럭. 레인 큐에서 화물을 싣고 주기적으로 출발 |
| `Door` | 인바운드 트럭이 하역하는 입고 도어 (처리 중엔 점유됨) |
| `Lane` | 목적지별 레인. 각각 하나의 **에이전트**에 해당 |
| `Buffer` | 화물이 레인으로 이동하기 전 대기하는 공유 스테이징 공간 |

### 기본 설정값

```yaml
num_lanes: 5               # 에이전트 수
num_inbound_doors: 3       # 입고 도어 수
buffer_capacity: 150       # 버퍼 최대 용량
episode_length: 100        # 에피소드 길이 (타임스텝)
truck_arrival_prob: 0.4    # 매 스텝 트럭 도착 확률
max_door_processing: 10    # 도어 처리 최대 소요 시간
inbound_min_dest: 2        # 인바운드 트럭 최소 목적지 수
inbound_max_dest: 3        # 인바운드 트럭 최대 목적지 수
inbound_vol_min: 5         # 목적지당 최소 화물량
inbound_vol_max: 20        # 목적지당 최대 화물량
outbound_capacity: 50      # 아웃바운드 트럭 1대 최대 적재량
dispatch_interval: 20      # 아웃바운드 출발 주기 (타임스텝)
```

---

## 에이전트 행동 / 관측 / 보상

### 행동 공간 (Action Space)

```
0 → 아무것도 안 함
1 → 도어 1 요청
2 → 도어 2 요청
3 → 도어 3 요청
```

### 관측 벡터 (Observation, 에이전트별, 크기 = 7 + num_doors)

```python
obs = [
    lane_queue,              # 0: 레인 현재 적재량
    lane_congestion,         # 1: 혼잡도 (0~1 정규화)
    outbound_fill_rate,      # 2: 아웃바운드 트럭 현재 탑재율 (0~1)
    outbound_departure_in,   # 3: 아웃바운드 출발까지 남은 타임스텝
    buffer_remaining,        # 4: 버퍼 여유 용량
    idle_inbound_doors,      # 5: 현재 유휴 도어 수
    waiting_trucks,          # 6: 대기 중인 트럭 수
    door_match_1,            # 7: 도어 1의 화물 매칭도
    door_match_2,            # 8: 도어 2의 화물 매칭도
    door_match_3,            # 9: 도어 3의 화물 매칭도
]
```

### 보상 구조

```python
R_team  = 이번_스텝_출발_화물량 - 0.5 × 버퍼_초과 - 2.0 × 빈_출발
R_local = 내_레인_출발_화물량 - 0.1 × 혼잡도

R_final = 0.7 × R_team + 0.3 × R_local   # 팀:개인 = 7:3
```

> `빈_출발`: 탑재율(fill_rate) < 10%인 채로 출발한 아웃바운드 트럭

### 스텝 실행 순서

```
1. 트럭 도착 생성 (Bernoulli 확률, 2~3 목적지 혼재)
2. 도어 상태 갱신 (처리 완료된 트럭 방출)
3. 방출된 트럭의 화물 → 버퍼 → 레인 이동
4. 대기열에 새 트럭 추가
5. 에이전트 행동 수집 및 충돌 해결
6. 도어 배정 (점수 기반 우선순위)
7. 아웃바운드 트럭 출발 처리 (타이머 만료 시)
8. 보상 계산
```

### 충돌 해결 (Conflict Resolution)

여러 에이전트가 같은 도어를 요청하면 아래 점수로 우선순위를 결정합니다.

```
score(k, i) = alpha × 긴급도_k + beta × 화물매칭도(k, i) - gamma × 혼잡도_k

긴급도        = 1 / (아웃바운드_출발까지_남은_시간 + 1)
화물매칭도    = 트럭에서 레인 k로 가는 물량 / 트럭 전체 물량
혼잡도        = 레인 현재 적재량 / 50 (정규화)

기본 가중치: alpha=1.0, beta=1.0, gamma=1.0
```

---

## 파일 구성

```
lcl_gym/
├── env/                          # 시뮬레이션 환경
│   ├── __init__.py
│   ├── entities.py               # Truck, OutboundTruck, Door, Lane 데이터 클래스
│   ├── conflict_resolver.py      # 점수 기반 도어 충돌 해결기
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
├── run_simulation.py             # 베이스라인 벤치마크 실행
├── checkpoints/                  # 학습 가중치 및 로그 저장
└── README.md
```

---

## 실행 방법

### 요구 사항

```bash
python >= 3.8
numpy
```

별도 RL 프레임워크 불필요 (numpy만 사용).

### 베이스라인 정책 비교 실행

```bash
python run_simulation.py
```

### 스텝별 상세 진행 확인 (Verbose 모드)

에피소드 내부에서 각 스텝이 어떻게 진행되는지 확인할 수 있습니다.

```bash
# 기본 (Greedy 정책, 10스텝)
python run_simulation.py --verbose

# 정책 및 스텝 수 지정
python run_simulation.py --verbose --policy fifo       --steps 30
python run_simulation.py --verbose --policy random     --steps 20
python run_simulation.py --verbose --policy heuristic  --steps 25
```

사용 가능한 정책: `greedy` / `fifo` / `random` / `heuristic`

출력 예시:

```
[Step   5]
  Waiting trucks : 1              ← 대기 중인 트럭 수
  Buffer         : 12 / 150       ← 현재 버퍼 점유 / 최대 용량
  Doors          : D0[idle]  D1[idle]  D2[idle]
  Lanes          : L0(q=0 ob=0% dep=20)  L1(q=12 ob=24% dep=18)  ...
  Actions        : L0→door0  L1→door0  L2→door0  ...   ← 에이전트 행동
  Rewards        : L0:+0.0  L1:+0.0  ...
  Doors (after)  : D0[busy:3t]  D1[idle]  D2[idle]     ← 3t = 3스텝 후 완료
  Lanes (after)  : L0(q=0 ob=0% dep=19)  ...
```

각 항목 의미:

| 항목 | 설명 |
|---|---|
| `Waiting trucks` | 도어 배정을 기다리는 트럭 수 |
| `Buffer` | 현재 화물 적재량 / 최대 용량 |
| `D0[idle]` | 도어가 비어 있음 |
| `D0[busy:Nt]` | 도어가 점유 중이며 N스텝 후 완료 |
| `q=N` | 레인의 현재 화물 적재량 |
| `ob=X%` | 아웃바운드 트럭 현재 탑재율 |
| `dep=N` | 아웃바운드 출발까지 남은 스텝 수 (0이 되면 출발) |

### 환경 직접 사용

```python
from env.crossdock_env import CrossDockEnv
from env.policies import HeuristicPriorityPolicy

config = {
    "num_lanes": 5,
    "num_inbound_doors": 3,
    "episode_length": 100,
}

env = CrossDockEnv(config, seed=42)
policies = [HeuristicPriorityPolicy() for _ in range(env.num_lanes)]

obs = env.reset()

for t in range(env.episode_length):
    actions = [
        policies[k].act(obs[k], env.num_inbound_doors)
        for k in range(env.num_lanes)
    ]
    obs, rewards, done, info = env.step(actions)

    if done:
        metrics = info["metrics"]
        print(f"총 처리량: {metrics['total_throughput']:.1f}")
        print(f"평균 탑재율: {env.avg_fill_rate:.2%}")
        print(f"도어 이용률: {env.door_utilization:.2%}")
        print(f"평균 체류 시간: {env.avg_dwell_time:.2f} steps")
        break
```

---

## 베이스라인 정책 설명

| 정책 | 설명 | 특징 |
|---|---|---|
| `RandomPolicy` | 매 스텝 무작위 행동 | 기준선 |
| `FIFOPolicy` | 항상 도어 1 요청 | 버퍼 초과 낮음, 도어 이용률 낮음 |
| `GreedyPolicy` | 화물 매칭도가 가장 높은 도어 선택 | 처리량 높음 |
| `HeuristicPriorityPolicy` | 긴급도 + 매칭도 - 혼잡도 종합 점수로 선택 | 충돌 해결기와 동일한 로직 |

---

## RL 학습 (IQL + Parameter Sharing DQN)

numpy만으로 구현된 DQN 기반 MARL 학습 파이프라인이 포함되어 있습니다.

### 알고리즘 구조

**IQL (Independent Q-Learning) + Parameter Sharing**

5개 에이전트가 동일한 가중치(NumpyMLP)를 공유해 학습합니다.

```
에이전트 1 ──┐
에이전트 2 ──┤
에이전트 3 ──┼──→ 공유 NumpyMLP → Q(obs, action)
에이전트 4 ──┤
에이전트 5 ──┘
```

- 5개 에이전트의 경험이 동시에 같은 네트워크를 업데이트 → 데이터 효율 5배
- 평가 시에는 각 에이전트가 독립적으로 행동 (분산 실행)

### 네트워크 구조

```
입력(10) → Linear(64) → ReLU → Linear(4) → Q값
```

- 역전파: 선택한 action 위치만 TD error로 gradient 계산, 나머지는 0 (stop-gradient)
- 옵티마이저: Adam (β1=0.9, β2=0.999)
- Target Network: 50 에피소드마다 동기화 (학습 안정화)

### Reward Shaping

환경 보상에 아래 항목을 추가해 학습 신호를 강화합니다 (환경 코드 수정 없음).

```python
R_shaped = R_env
         + 1.0 × door_match    # 선택한 도어의 화물 매칭도가 높을수록 보너스
         - 0.1 × congestion    # 혼잡 억제
```

### 학습 실행

```bash
# 기본 (1000 에피소드)
python rl/train_rl.py

# 옵션 지정
python rl/train_rl.py --episodes 2000 --lr 5e-4
python rl/train_rl.py --episodes 1000 --no-share   # 에이전트별 독립 가중치
python rl/train_rl.py --episodes 1000 --seed 0
```

학습 중 100 에피소드마다 진행상황이 출력됩니다:

```
 Episode    AvgReward   Throughput   Overflow     TDLoss   Epsilon
-----------------------------------------------------------------
     100       2580.0        695.2       32.1   145.3012     0.606
     500       2610.5        704.3       29.8     6.2145     0.082
    1000       2624.7        707.1       27.3     8.9102     0.050
```

학습 완료 후 `checkpoints/` 폴더에 저장됩니다:

```
checkpoints/
├── weights_final.npz       # 최종 가중치
├── weights_ep100.npz       # 중간 체크포인트
├── episode_rewards.npy     # 에피소드별 누적 보상
├── throughput_log.npy      # 에피소드별 처리량
├── overflow_log.npy        # 에피소드별 버퍼 초과 횟수
└── td_loss_log.npy         # 에피소드별 TD 손실
```

### 평가 실행

```bash
# 학습 곡선 요약 + 베이스라인 비교
python rl/evaluate_rl.py

# 옵션 지정
python rl/evaluate_rl.py --episodes 50
python rl/evaluate_rl.py --weights checkpoints/weights_ep500
```

### 관측/행동 크기

```python
env = CrossDockEnv()
print(env.obs_size)              # 10  (7 공통 + num_doors 매칭도)
print(env.num_inbound_doors + 1) # 4   (0=skip, 1~3=도어 요청)
print(env.num_lanes)             # 5   (에이전트 수)
```

### PyTorch 확장 방법

`networks.py`의 `NumpyMLP`를 `TorchMLP`로 교체하면 나머지 파일은 수정 없이 동작합니다.

```python
# networks.py 에 추가
import torch
import torch.nn as nn

class TorchMLP(nn.Module):
    """NumpyMLP와 동일한 인터페이스"""
    def forward(self, obs): ...
    def update(self, obs, actions, targets) -> float: ...
    def copy_weights_from(self, other): ...
    def save(self, path): ...
    def load(self, path): ...

# train_rl.py 에서
shared_net = TorchMLP(obs_size=10, hidden=64, n_actions=4)  # 이것만 교체
```

이후 확장 경로:

```
NumpyMLP → TorchMLP (DQN)
         → QMIX     (팀 보상 분해, R_team 활용)
         → MAPPO    (Actor-Critic, 안정적 수렴)
```

### 설정 커스터마이징

```python
config = {
    "num_lanes": 8,
    "num_inbound_doors": 4,
    "buffer_capacity": 200,
    "episode_length": 200,
    "truck_arrival_prob": 0.5,
    "outbound_capacity": 80,
    "dispatch_interval": 25,
    "reward_alpha": 0.7,   # 팀 보상 비중
    "reward_beta": 0.3,    # 개인 보상 비중
    "cr_alpha": 1.0,       # 충돌 해결: 긴급도 가중치
    "cr_beta": 1.0,        # 충돌 해결: 매칭도 가중치
    "cr_gamma": 1.0,       # 충돌 해결: 혼잡도 가중치
}
env = CrossDockEnv(config, seed=42)
```

---

## 출력 메트릭

에피소드 종료 시 `info["metrics"]`로 접근 가능합니다.

| 메트릭 | 설명 | 방향 |
|---|---|---|
| `total_throughput` | 에피소드 전체 아웃바운드 탑재된 화물 총량 | 높을수록 좋음 |
| `total_fill_rate` | 아웃바운드 트럭 출발 시 탑재율 합산 | 높을수록 좋음 |
| `outbound_departures` | 총 아웃바운드 출발 횟수 | — |
| `empty_departures` | 탑재율 10% 미만으로 빈 채 출발한 횟수 | 낮을수록 좋음 |
| `buffer_overflow_count` | 버퍼 용량 초과 발생 횟수 | 낮을수록 좋음 |
| `door_busy_steps` | 도어가 점유된 누적 스텝 수 | — |
| `avg_dwell_time` | 트럭 도착 ~ 화물 처리 완료까지 평균 대기 시간 | 낮을수록 좋음 |
| `door_utilization` | 도어 평균 점유율 (0~1) | 1에 가까울수록 도어 활용도 높음 |
| `avg_fill_rate` | 출발한 아웃바운드 트럭의 평균 탑재율 | 높을수록 좋음 |

---

## 베이스라인 비교 결과 해석

```
Policy            throughput      overflow        reward     door_util       dwell_t
------------------------------------------------------------------------------------
Random            693.75±35.2     30.95±11.3   2565.87±134.6      0.63±0.1      6.06±1.4
FIFO              522.25±57.0      1.30±1.8    1954.47±221.5      0.32±0.0     29.10±4.8
Greedy            708.85±36.2     28.50±9.0    2627.41±130.9      0.65±0.1      8.11±1.4
Heuristic         708.95±32.3     29.35±7.9    2627.04±124.3      0.66±0.1      8.43±1.9
```

### Random / Greedy / Heuristic — 세 정책이 거의 동일

처리량(~700), 보상(~2600), 도어 이용률(0.64)이 사실상 같습니다.

**원인: 충돌 해결기(ConflictResolver)의 평탄화 효과**
에이전트들이 어떤 도어를 요청하든, 충돌이 발생하면 점수 기반으로 최적 에이전트에게 도어가 배정됩니다.
정책이 달라도 충돌 해결기가 결과를 균일하게 만들어버리기 때문에 차별이 없습니다.

> Greedy와 Heuristic이 거의 동일한 이유는, 단일 타임스텝에서 두 정책이 실질적으로 같은 도어를 선택하기 때문입니다.

### FIFO — 혼자 다른 패턴

FIFO는 항상 **도어 1만** 요청합니다.

```
도어 2, 3은 대부분 놀고 있음 → door_util 0.32 (vs 0.64)
→ 트럭 처리 속도 저하
→ 버퍼에 화물이 천천히 쌓임 → overflow 1 (vs ~30)
→ 처리량 감소 (522 vs 700)
→ 트럭 대기 시간 급증 → dwell_t 29 (vs 6~8)
```

즉, FIFO의 낮은 overflow는 **"잘 관리해서"가 아니라 "병목이 생겨서"** 발생한 결과입니다.

### 핵심 시사점

| 관점 | 결론 |
|---|---|
| 베이스라인의 한계 | 충돌 해결기가 결과를 평탄화하여 Random / Greedy / Heuristic 간 차별이 거의 없음 |
| MARL 학습의 목표 | 에이전트가 **충돌 자체를 줄이는 협력 전략** (서로 다른 도어를 나눠 요청)을 학습해야 overflow와 dwell_t를 동시에 개선 가능 |
| RL로 달성할 것 | throughput ↑, overflow ↓, avg_fill_rate ↑ — 빈 채 출발하는 아웃바운드 트럭을 줄이는 것이 핵심 목표 |
