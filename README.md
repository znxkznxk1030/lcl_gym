# Cross-Dock MARL Simulator

크로스도킹(Cross-Docking) 물류 환경을 다중 에이전트 강화학습(MARL)용으로 구현한 시뮬레이터입니다.

> **크로스도킹이란?** 입고된 화물을 창고에 보관하지 않고, 즉시 목적지별 레인으로 분류해 출고하는 물류 방식입니다.

---

## 환경 구조

### 주요 엔티티

| 엔티티 | 설명 |
|---|---|
| `Truck` | 여러 레인에 배달할 화물을 싣고 도착하는 트럭 |
| `Door` | 트럭이 하역하는 입고 도어 (처리 중엔 점유됨) |
| `Lane` | 목적지별 레인. 각각 하나의 **에이전트**에 해당 |
| `Buffer` | 화물이 레인으로 이동하기 전 대기하는 공유 스테이징 공간 |

### 기본 설정값

```yaml
num_lanes: 5               # 에이전트 수
num_inbound_doors: 3       # 입고 도어 수
buffer_capacity: 100       # 버퍼 최대 용량
episode_length: 100        # 에피소드 길이 (타임스텝)
truck_arrival_prob: 0.3    # 매 스텝 트럭 도착 확률
max_door_processing: 10    # 도어 처리 최대 소요 시간
dispatch_interval: 20      # 레인 출고 주기 (타임스텝)
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

### 관측 벡터 (Observation, 에이전트별, 크기 = 6 + num_doors)

```python
obs = [
    lane_queue,           # 0: 레인 현재 적재량
    lane_congestion,      # 1: 혼잡도 (0~1 정규화)
    time_to_dispatch,     # 2: 출고까지 남은 타임스텝
    buffer_remaining,     # 3: 버퍼 여유 용량
    idle_inbound_doors,   # 4: 현재 유휴 도어 수
    waiting_trucks,       # 5: 대기 중인 트럭 수
    door_match_1,         # 6: 도어 1의 화물 매칭도
    door_match_2,         # 7: 도어 2의 화물 매칭도
    door_match_3,         # 8: 도어 3의 화물 매칭도
]
```

### 보상 구조

```python
R_team  = 처리량 - 0.5 × 버퍼_초과 - 1.0 × 지연_출고
R_local = 내_레인_처리량 - 0.1 × 혼잡도

R_final = 0.7 × R_team + 0.3 × R_local   # 팀:개인 = 7:3
```

### 스텝 실행 순서

```
1. 트럭 도착 생성 (Bernoulli 확률)
2. 도어 상태 갱신 (처리 완료된 트럭 방출)
3. 방출된 트럭의 화물 → 버퍼 → 레인 이동
4. 에이전트 행동 수집 및 충돌 해결
5. 도어 배정 (점수 기반 우선순위)
6. 레인 디스패치 (타이머 만료 시 출고)
7. 보상 계산
```

### 충돌 해결 (Conflict Resolution)

여러 에이전트가 같은 도어를 요청하면 아래 점수로 우선순위를 결정합니다.

```
score(k, i) = alpha × 긴급도_k + beta × 화물매칭도(k, i) - gamma × 혼잡도_k

긴급도        = 1 / (출고까지_남은_시간 + 1)
화물매칭도    = 트럭에서 레인 k로 가는 물량 / 트럭 전체 물량
혼잡도        = 레인 현재 적재량 / 50 (정규화)

기본 가중치: alpha=1.0, beta=1.0, gamma=1.0
```

---

## 파일 구성

```
lcl_gym/
├── entities.py           # Truck, Door, Lane 데이터 클래스
├── conflict_resolver.py  # 점수 기반 도어 충돌 해결기
├── policies.py           # 베이스라인 정책 4종
├── crossdock_env.py      # CrossDockEnv 메인 환경
└── run_simulation.py     # 정책 비교 벤치마크 실행
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

출력 예시:

```
Policy            throughput      overflow        reward     door_util       dwell_t
------------------------------------------------------------------------------------
Random            466.05±31.1     45.00±14.9   1689.09±118.3      0.52±0.1      5.64±0.8
FIFO              390.40±34.5      4.90±5.3   1472.73±127.9      0.31±0.0     23.12±6.5
Greedy            465.85±32.4     46.15±17.6   1686.22±123.4      0.52±0.1      7.08±0.9
Heuristic         465.85±32.4     46.15±17.6   1686.22±123.4      0.52±0.1      7.08±0.9
```

### 환경 직접 사용

```python
from crossdock_env import CrossDockEnv
from policies import HeuristicPriorityPolicy

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

## MARL 알고리즘 학습 방법

이 환경은 표준 `(obs, reward, done, info)` 인터페이스를 따르므로 어떤 MARL 알고리즘도 붙일 수 있습니다.

### 기본 학습 루프

```python
from crossdock_env import CrossDockEnv

env = CrossDockEnv(seed=0)
num_agents = env.num_lanes

# 에이전트 초기화 (예: 각자 독립적인 Q-네트워크)
agents = [YourAgent(obs_size=env.obs_size, act_size=env.num_inbound_doors + 1)
          for _ in range(num_agents)]

for episode in range(1000):
    obs = env.reset()
    total_reward = 0.0

    for t in range(env.episode_length):
        # 각 에이전트가 독립적으로 행동 선택
        actions = [agents[k].act(obs[k]) for k in range(num_agents)]

        next_obs, rewards, done, info = env.step(actions)

        # 경험 저장 및 학습
        for k in range(num_agents):
            agents[k].store(obs[k], actions[k], rewards[k], next_obs[k], done)
            agents[k].train()

        obs = next_obs
        total_reward += sum(rewards)

        if done:
            break

    print(f"Episode {episode} | Total Reward: {total_reward:.1f}")
```

### 권장 알고리즘

| 알고리즘 | 특징 | 적합한 이유 |
|---|---|---|
| **IQL** (Independent Q-Learning) | 에이전트별 독립 학습 | 구현 단순, 빠른 베이스라인 |
| **VDN / QMIX** | 팀 보상 분해 | `R_team` 공유 보상 구조에 적합 |
| **MAPPO** | 중앙집중 학습 + 분산 실행 | 안정적인 학습, 확장성 좋음 |

### 관측/행동 크기

```python
env = CrossDockEnv()
print(env.obs_size)              # 9  (6 + num_doors)
print(env.num_inbound_doors + 1) # 4  (0~num_doors 포함)
print(env.num_lanes)             # 5  (에이전트 수)
```

### 설정 커스터마이징

```python
config = {
    "num_lanes": 8,
    "num_inbound_doors": 4,
    "buffer_capacity": 200,
    "episode_length": 200,
    "truck_arrival_prob": 0.5,
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

| 메트릭 | 설명 |
|---|---|
| `total_throughput` | 에피소드 전체 처리된 화물 총량 |
| `buffer_overflow_count` | 버퍼 용량 초과 발생 횟수 |
| `door_busy_steps` | 도어가 점유된 누적 스텝 수 |
| `avg_dwell_time` | 트럭 도착 ~ 화물 처리 완료까지 평균 시간 |
| `door_utilization` | 도어 평균 이용률 (0~1) |
