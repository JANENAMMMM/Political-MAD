"""
rl/rewards/cli.py
──────────────────
`mad-reward` 명령. 토론 턴(JSONL)을 넣으면 보상 컴포넌트별 점수를 보여준다.

GRPO 학습을 돌리기 전에 "이 보상이 내가 의도한 발언에 높은 점수를 주는가"를
몇 개의 손으로 쓴 예시로 바로 확인하려고 만들었다. 학습 중 로그로 처음 알게 되면
이미 GPU 시간을 쓴 뒤이기 때문이다.

    mad-reward score turns.jsonl                         # judge-free 기본(novelty, engagement)
    mad-reward score turns.jsonl --components all --embedding-model BAAI/bge-m3
    mad-reward score turns.jsonl --format json           # 기계가 읽기 좋은 JSON Lines

JSONL 한 줄 = DebateTurnSample 필드:
    {"response": "...", "side": "left", "question": "...", "opponent_response": "...",
     "own_history": ["..."], "round_num": 2}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import fields

from .base import DebateTurnSample

LIGHT_COMPONENTS = ("novelty", "engagement")
EMBED_COMPONENTS = ("diversity",)
ALL_COMPONENTS = LIGHT_COMPONENTS + EMBED_COMPONENTS

_SAMPLE_FIELDS = {f.name for f in fields(DebateTurnSample)}


def load_samples(path: str) -> list[DebateTurnSample]:
    """JSONL 파일을 읽어 DebateTurnSample 리스트로 바꾼다. 빈 줄은 건너뛴다."""
    samples = []
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: JSON이 아닙니다 ({exc.msg})") from exc
            missing = {"response", "side"} - row.keys()
            if missing:
                raise SystemExit(f"{path}:{lineno}: 필수 필드 없음: {', '.join(sorted(missing))}")
            if row["side"] not in ("left", "right"):
                raise SystemExit(f"{path}:{lineno}: side는 left|right 중 하나여야 합니다 (받은 값: {row['side']!r})")
            unknown = row.keys() - _SAMPLE_FIELDS
            if unknown:
                raise SystemExit(f"{path}:{lineno}: 알 수 없는 필드: {', '.join(sorted(unknown))}")
            row.setdefault("question", "")
            row.setdefault("opponent_response", "")
            samples.append(DebateTurnSample(**row))
    return samples


def build_components(names: Sequence[str], embedding_model: str | None = None) -> list:
    """이름 목록으로 보상 컴포넌트 인스턴스를 만든다. 임베딩이 필요한 것은 모델 이름이 있어야 한다."""
    from .utils import SentenceEmbedder
    from .v2_redesign import DiversityReward, EngagementReward, NoveltyReward

    components = []
    embedder = None
    for name in names:
        if name == "novelty":
            components.append(NoveltyReward())
        elif name == "engagement":
            components.append(EngagementReward())
        elif name == "diversity":
            if embedding_model is None:
                raise SystemExit(
                    "diversity는 임베딩이 필요합니다: --embedding-model BAAI/bge-m3 (pip install '.[embed]')"
                )
            embedder = embedder or SentenceEmbedder(model_id=embedding_model)
            components.append(DiversityReward(embedder))
        else:
            raise SystemExit(f"알 수 없는 컴포넌트: {name!r} (선택지: {', '.join(ALL_COMPONENTS)}, all)")
    return components


def score_samples(samples: Sequence[DebateTurnSample], components: Sequence) -> list[dict[str, float]]:
    return [{c.name: round(float(c(s)), 4) for c in components} for s in samples]


def _format_table(samples: Sequence[DebateTurnSample], rows: Sequence[dict[str, float]]) -> str:
    names = list(rows[0]) if rows else []
    header = ["#", "side", "round", *names, "response"]
    lines = ["\t".join(header)]
    for i, (s, r) in enumerate(zip(samples, rows, strict=True), start=1):
        preview = s.response.replace("\n", " ")
        preview = preview if len(preview) <= 40 else preview[:39] + "…"
        lines.append("\t".join([str(i), s.side, str(s.round_num), *(f"{r[n]:+.3f}" for n in names), preview]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mad-reward", description="토론 턴별 보상 컴포넌트 점수를 계산합니다.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_score = sub.add_parser("score", help="JSONL의 각 턴에 대해 컴포넌트별 점수를 출력")
    p_score.add_argument("path", help="DebateTurnSample JSONL 파일 경로 (- 이면 stdin)")
    p_score.add_argument(
        "--components",
        default=",".join(LIGHT_COMPONENTS),
        help=f"쉼표로 구분 (선택지: {', '.join(ALL_COMPONENTS)}, all). 기본값은 GPU·모델 없이 도는 것만",
    )
    p_score.add_argument("--embedding-model", default=None, help="diversity에 쓸 sentence-transformers 모델")
    p_score.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(argv)

    names = (
        ALL_COMPONENTS
        if args.components == "all"
        else tuple(n.strip() for n in args.components.split(",") if n.strip())
    )
    samples = load_samples(args.path)
    components = build_components(names, args.embedding_model)
    rows = score_samples(samples, components)

    if args.format == "json":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(_format_table(samples, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
