# Political-MAD: 토론 에이전트를 위한 judge-free 보상 모듈

진보·보수 페르소나를 가진 두 LLM 에이전트가 정책을 두고 토론하게 하는 프로젝트(TOBIG's 22기 컨퍼런스 NLP팀, "Policy Debate AI")에서, GRPO 강화학습에 쓰는 **보상 함수 모듈**과 그 도구를 담은 리포지토리입니다.

> 이 리포는 팀 저장소 [`kangsoobin/llm_dabate`](https://github.com/kangsoobin/llm_dabate)의 fork입니다.
> 팀(강수빈, 원정선, 김가빈, 김은경, 남재은, 송화원)이 함께 설계한 프로젝트이고, 이 fork의 `jaeeun` 브랜치는 보상 모듈을 독립적으로 쓰고 검증할 수 있게 정리한 버전입니다.

## 어떤 문제를 푸나

멀티에이전트 토론(MAD)에서 LLM은 몇 가지 방식으로 망가집니다. 라운드가 길어지면 페르소나를 잃고 중립으로 돌아가고(neutral drift), 상대에게 동조하고(sycophancy), 상대 논점을 피하고, 같은 말을 반복합니다. GRPO로 이걸 고치려면 각 실패를 점수로 잴 수 있어야 합니다.

가장 쉬운 방법은 LLM Judge에게 점수를 맡기는 것이지만, 그러면 **Judge의 능력이 시스템의 상한**이 되고(Judge Ceiling), 정답이 없는 정치 성향 과제에서는 Judge의 기준 자체가 흔들립니다. 그래서 이 모듈은 실패 유형마다 **Judge 없이도 계산되는 신호**를 하나씩 대응시킵니다.

| 실패 유형 | 보상 컴포넌트 | 계산 방식 |
|---|---|---|
| 중립 회귀, 동조 | `persona` | 페르소나 anchor와의 임베딩 유사도 − 직전보다 상대 쪽으로 가까워진 정도 |
| 논점 회피 | `engagement` | 상대 발언의 핵심 명사 중 이번 발언이 언급한 비율 |
| 에코 챔버 | `diversity` | 1 − max(상대 발언, 자기 과거 발언과의 임베딩 유사도) |
| 표면적 반복 | `novelty` | 자기 과거 발언과의 형태소 Jaccard가 τ를 넘으면 패널티 |
| 근거 없는 주장 | `grounding` | 근거 문서가 있을 때만 NLI 점수 (현재 비활성) |

Judge 점수는 `engagement`, `persona`에 **보조 신호로만** 섞을 수 있습니다(`config/reward.yaml`에서 가중치 0이 기본).

## 설치

```bash
uv sync                      # 보상 모듈 + 테스트 도구만 (GPU 불필요)
uv pip install -e '.[ko]'    # 한국어 형태소 분석(kiwipiepy). 없으면 정규식 토큰화로 대체
uv pip install -e '.[embed]' # persona/diversity용 임베딩 모델
uv pip install -e '.[train]' # SFT·GRPO 학습 (GPU 서버)
```

무거운 의존성(torch, sentence-transformers, kiwipiepy)은 실제로 쓰는 순간에 import합니다. 그래서 테스트와 CLI는 노트북에서도 돕니다.

## 빠른 시작: 보상이 의도대로 점수를 주는지 보기

```bash
uv run mad-reward score tests/data/turns.jsonl
```

```text
#  side   round  novelty  engagement  response
1  left   1      +0.000   +0.429      최저임금 인상은 저임금 노동자의 생계를 지키는…
2  left   2      -1.125   +0.167      최저임금 인상은 저임금 노동자의 생계를 지키는…   ← 자기 발언 반복
3  right  2      +0.000   +0.000      양쪽 다 일리가 있다고 생각합니다…               ← 상대 논점 무시
```

입력은 한 줄에 토론 턴 하나인 JSONL입니다.

```json
{"response": "...", "side": "left", "question": "...", "opponent_response": "...", "own_history": ["..."], "round_num": 2}
```

| 옵션 | 설명 |
|---|---|
| `--components novelty,engagement` | 계산할 컴포넌트 (기본값은 모델 없이 도는 두 개, `all` 가능) |
| `--embedding-model BAAI/bge-m3` | `diversity`에 필요한 임베딩 모델 |
| `--format json` | 한 줄에 하나씩 JSON으로 출력 (다른 도구에 넘길 때) |

### 라이브러리로 쓰기

```python
from rl.rewards import DebateTurnSample, RewardComposer
import yaml

composer = RewardComposer.from_config(yaml.safe_load(open("config/reward.yaml")))
reward_fn = composer.as_trl_reward_fn()   # TRL GRPOTrainer(reward_funcs=[reward_fn])에 그대로 넘긴다
composer.breakdown(DebateTurnSample(response="...", side="left", question="...", opponent_response="..."))
```

새 보상은 `name` 속성과 `__call__(sample) -> float`만 있으면 됩니다(`rl/rewards/base.py`의 `RewardComponent`). 학습 중에는 컴포넌트별 raw 평균이 `reward_components/<name>`으로 로깅돼서, 한 컴포넌트만 오르고 나머지가 무너지는 reward hacking을 일찍 볼 수 있습니다.

## 설계 결정과 이유

- **학습 전에 보상부터 검사한다.** 임베딩 유사도 보상은 미묘한 페르소나 모순을 못 잡을 수 있다는 경고가 있습니다(Abdulhai et al., NeurIPS 2025). 그래서 `rl/check_reward_sanity.py`가 "같은 성향 응답 > 반대 성향 응답", "같은 성향 응답 > 중립 응답"이 실제로 마진 이상 벌어지는지 확인하고, 하나라도 실패하면 exit 1로 학습을 막습니다. 변별력 없는 보상으로 GPU 시간을 쓰지 않으려는 장치입니다.
- **모든 수렴이 동조는 아니다.** 상대 쪽으로 가까워졌더라도 상대 논점을 정면으로 다룬 경우(반박)는 패널티를 `(1 − coverage)`만큼 줄입니다. 논점을 피하면서 가까워진 경우만 강하게 처벌합니다.
- **조용히 무력화되는 대신 실패한다.** persona anchor 파일이 비어 있으면 0점을 주며 넘어가지 않고 생성 시점에 에러를 냅니다.
- **한국어 토큰화.** 어절 단위로 자르면 "최저임금을"과 "최저임금이"가 다른 토큰이 돼 커버리지가 체계적으로 낮게 나옵니다. kiwipiepy 형태소 단위를 쓰고, 없을 때만 정규식으로 대체합니다(안내는 stderr로).

자세한 근거는 [`docs/reward_design_v2.md`](docs/reward_design_v2.md), 발표용 정리는 [`docs/보상설계_발표정리.md`](docs/보상설계_발표정리.md)에 있습니다.

## 결과

- SFT만 한 Left 에이전트는 홀드아웃 토론에서 입장 유지율 0%로 중립에 무너졌고, GRPO를 더한 뒤 89%가 됐습니다 (Kanana-2-30B-A3B, 홀드아웃 질문 10개).
- 양측 GRPO를 4주제 × 8라운드로 비교한 리포트는 [`docs/eval/20260704_sft_vs_grpo_report.md`](docs/eval/20260704_sft_vs_grpo_report.md)에 있습니다. 논점 커버리지는 8라운드 중 7라운드에서 SFT 이상이었고, 페르소나 anchor 유사도는 거의 같았습니다.

## 한계와 다음 계획

- 위 빠른 시작의 3번 턴처럼, **중립·양시론 발언 자체에는 judge-free 컴포넌트가 0점을 줄 뿐 패널티를 주지 않습니다.** 중립 회귀는 `persona`의 anchor 유사도로만 간접적으로 잡습니다. 팀 원안의 중립 표현 패널티를 컴포넌트로 옮기는 것이 다음 작업입니다.
- `engagement`는 핵심 명사가 겹치는지만 봅니다. 상대 논점을 언급만 하고 반박하지 않아도 점수를 받습니다.
- 학습·시뮬레이션 스크립트(`rl/train_grpo.py`, `rl/simulate_vllm.py`)는 GPU 서버 전용이고 CI 범위 밖입니다. CI와 린트는 보상 모듈, CLI, 테스트만 다룹니다.

## 리포 구조

```text
rl/rewards/          보상 라이브러리 (CI·린트 대상)
  base.py            DebateTurnSample, RewardComponent 인터페이스
  v2_redesign.py     judge-free 보상 5종
  v1_pdf.py          팀 원안(Judge 필수) 보상
  judge.py           local / API Judge, 관대한 JSON 파싱
  composer.py        가중합 + TRL reward_fn 어댑터 + 컴포넌트별 로깅
  cli.py             mad-reward 명령
rl/                  GRPO self-play·학습·평가 스크립트 (GPU)
  check_reward_sanity.py   학습 전 보상 변별력 점검 (go/no-go)
  compare_debates.py       SFT vs GRPO 토론 비교 리포트
tests/               pytest (모델 다운로드 없이 가짜 임베더로 로직 검사)
sft/, agents/, debate.py, app.py, fronted/   팀 공통 토론 시스템
docs/RUNBOOK.md      서버에서 SFT·GRPO를 돌리는 순서
```

## 개발

```bash
uv run ruff check && uv run ruff format --check
uv run pytest -q
```

GitHub Actions가 push마다 Python 3.10, 3.12에서 린트, 포맷 검사, 테스트, CLI 실행을 돌립니다.
