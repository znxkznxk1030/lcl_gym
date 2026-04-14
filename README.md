# LCL MARL 시뮬레이터

LCL 통합을 위한 이산시간 멀티에이전트 강화학습 환경입니다. 이 프로젝트는 그리드월드 위를 이동하는 대신, 공유 물류 병목 아래에서 협업하는 5개의 목적지 레인 에이전트에 초점을 맞춥니다.

## 핵심 환경

- `5 agents = 5 destination lanes`
- 컷오프 일정, 배차 비용, 배차 용량, 레인 버퍼 용량, 수요 패턴, 긴급도 프로필 측면에서 레인별 이질성 존재
- 시간에 따라 확률적으로 발생하는 화물 도착
- 고정 레인 화물과 다중 레인에 유연하게 배정 가능한 화물 지원
- 각 타임스텝에서 에이전트별 의사결정:
  - 보이는 호환 화물 하나를 수락하거나 아무것도 수락하지 않음
  - 배차를 요청할지 여부 결정
- 공유 병목:
  - 타임스텝당 제한된 배차 슬롯
  - 대기 화물을 위한 제한된 공용 스테이징 용량
- 부분 관측:
  - 로컬 레인 상태
  - 상위 K개의 보이는 호환 화물
  - 압축된 전역 요약 정보
  - 요약된 다른 레인 상태
- 유효하지 않은 행동은 마스킹되며, 패널티와 함께 no-op로 거부됨
- 보상은 팀 수준 요소와 로컬 레인 수준 요소를 함께 사용
- 에이전트별 관측값, 보상, 종료 플래그, info 딕셔너리, 액션 마스크, 중앙집중식 상태를 제공
- 진단 지표에는 활용률, 지연, 마감 미준수, 오버플로우, 배차 횟수, 레인별 메트릭이 포함됨

## 패키지 구성

새로운 MARL 환경은 `lcl_marl/`에 있습니다.

- `lcl_marl/env.py`: 메인 환경 API
- `lcl_marl/arrivals.py`: 확률적 도착 프로세스
- `lcl_marl/observations.py`: 에이전트별 부분 관측값과 마스크
- `lcl_marl/reward.py`: 팀/로컬 보상 조합
- `lcl_marl/conflict_resolution.py`: 유연 화물 요청 조정 및 배차 슬롯 할당
- `lcl_marl/metrics.py`: 진단 지표와 레인별 메트릭
- `lcl_marl/baselines.py`: 탐욕적 휴리스틱 베이스라인과 IPPO/MAPPO/중앙집중식 PPO 지원 설명자
- `lcl_marl/demo.py`: 실행 가능한 CLI 데모

기존 단일 에이전트 시뮬레이터와 웹/서버 코드는 `simulator_v1/`, `server/`, `agents/`에 그대로 남아 있습니다.

## 빠른 시작

새 환경 데모 실행:

```bash
python run.py sim --phase 3 --horizon 24
```

모듈 직접 실행:

```bash
python -m lcl_marl.demo --phase 3 --horizon 24 --json
```

기존 시뮬레이터 실행:

```bash
python run.py legacy-sim
```

## API 예시

```python
from lcl_marl import GreedyHeuristicPolicy, LCLConsolidationEnv, build_phase_config

config = build_phase_config(phase=3, seed=7, horizon=48)
env = LCLConsolidationEnv(config)
policy = GreedyHeuristicPolicy()

observations, infos = env.reset()
done = False

while not done:
    actions = policy.act(observations)
    observations, rewards, dones, infos = env.step(actions)
    done = dones["__all__"]

print(env.get_metrics())
print(env.get_centralized_state())
print(env.get_action_masks())
```

각 에이전트 관측값은 다음을 포함합니다:

- `local_state`
- `visible_shipments`
- `global_summary`
- `other_lanes`
- `action_mask`

환경은 다음과 같은 구조화된 액션을 받을 수 있으며:

```python
{
    "lane_tokyo": {"accept_index": 0, "request_dispatch": False},
    "lane_singapore": {"accept_index": 5, "request_dispatch": True},
}
```

또는 `env.encode_action(...)` / `env.decode_action(...)`를 통한 인코딩된 이산 액션도 받을 수 있습니다.

## 실험 단계

`build_phase_config()`는 구현 단계에 맞춘 순차적 설정을 제공합니다.

1. `phase=1`
   고정 레인 화물만 지원하며, 공유 용량이 넉넉하고, 관측 구성이 단순합니다.
2. `phase=2`
   유연 화물, 제한된 배차 슬롯, 공용 스테이징 병목이 추가됩니다.
3. `phase=3`
   전체 레인 이질성, 더 풍부한 부분 관측, 액션 마스크, 확장된 메트릭이 추가됩니다.

## 베이스라인 지원

이 환경은 다음 베이스라인을 지원하도록 구성되어 있습니다:

- `greedy`: `lcl_marl.baselines.GreedyHeuristicPolicy`에 구현된 규칙 기반 분산 휴리스틱
- `ippo`: 에이전트별 관측값 + 액션 마스크
- `mappo`: 에이전트별 관측값 + `env.get_centralized_state()`
- `centralized_ppo`: 중앙집중식 상태와 joint/discrete 액션 인코딩 헬퍼

실행 중 지원 메타데이터 확인:

```python
support = env.get_baseline_support()
print(support["joint_action_size"])
print(support["baselines"]["mappo"])
```

## 테스트

```bash
python -m pytest -q
```
