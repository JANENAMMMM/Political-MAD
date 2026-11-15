import pytest

from rl.rewards.base import DebateTurnSample
from rl.rewards.judge import _parse_judge_json, build_judge
from rl.rewards.v2_redesign import (
    DiversityReward,
    EngagementReward,
    GroundingReward,
    NoveltyReward,
    PersonaConsistencyReward,
)


def turn(**kw):
    base = dict(response="", side="left", question="q", opponent_response="", own_history=[])
    base.update(kw)
    return DebateTurnSample(**base)


class TestNovelty:
    def test_no_penalty_below_tau(self):
        assert NoveltyReward(tau=0.5)(turn(response="최저임금 인상", own_history=["복지 확대"])) == 0.0

    def test_full_repeat_is_penalized(self):
        r = NoveltyReward(tau=0.25, gamma=1.5)(turn(response="최저임금 인상", own_history=["최저임금 인상"]))
        assert r == pytest.approx(-1.5 * 0.75)


class TestEngagement:
    def test_zero_without_opponent(self):
        assert EngagementReward()(turn(response="아무 말")) == 0.0

    def test_judge_free_equals_coverage(self):
        r = EngagementReward()(turn(response="세금 문제", opponent_response="세금 복지"))
        assert r == pytest.approx(0.5)

    def test_judge_is_mixed_in_when_weighted(self):
        class StubJudge:
            def score_rebuttal(self, opponent_text, response_text):
                return 5  # -> (5-1)/4 = 1.0

        r = EngagementReward(judge=StubJudge(), judge_weight=0.5)(turn(response="세금", opponent_response="세금 복지"))
        assert r == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)


class TestDiversity:
    def test_echoing_opponent_scores_low(self, fake_embedder):
        d = DiversityReward(fake_embedder)
        echo = d(turn(response="세금 복지 확대", opponent_response="세금 복지 확대"))
        fresh = d(turn(response="노동 시장 유연성", opponent_response="세금 복지 확대"))
        assert echo < fresh


class TestPersona:
    ANCHORS = {"left": ["노동자 복지 확대 분배"], "right": ["시장 성장 규제 완화"]}

    def test_empty_anchor_fails_fast(self, fake_embedder):
        with pytest.raises(ValueError, match="anchor"):
            PersonaConsistencyReward(fake_embedder, {"left": ["x"], "right": [""]})

    def test_own_persona_beats_opposite(self, fake_embedder):
        p = PersonaConsistencyReward(fake_embedder, self.ANCHORS)
        own = p(turn(response="노동자 복지 확대"))
        opposite = p(turn(response="시장 규제 완화"))
        assert own > opposite

    def test_sycophancy_gate_softens_penalty_when_engaging(self, fake_embedder):
        # 직전보다 상대 발언 쪽으로 가까워졌을 때, 상대 논점을 다룬 정도(coverage)만큼 패널티를 덜어 준다.
        sample = turn(
            response="노동자 복지 성장 규제",
            opponent_response="성장 규제",
            own_history=["노동자 복지 분배"],
        )
        gated = PersonaConsistencyReward(fake_embedder, self.ANCHORS, syc_coverage_gate=True)(sample)
        ungated = PersonaConsistencyReward(fake_embedder, self.ANCHORS, syc_coverage_gate=False)(sample)
        assert gated > ungated


def test_grounding_inactive_without_evidence_or_scorer():
    assert GroundingReward()(turn(response="x", evidence="근거")) == 0.0
    assert GroundingReward(lambda e, c: 0.9)(turn(response="x")) == 0.0
    assert GroundingReward(lambda e, c: 0.9)(turn(response="x", evidence="근거")) == 0.9


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"score": 0.3}', {"score": 0.3}),
        ('설명입니다.\n```json\n{"score": 4}\n```', {"score": 4}),
        ("JSON 없음", {}),
        ("{깨진 json}", {}),
    ],
)
def test_parse_judge_json_is_lenient(raw, expected):
    assert _parse_judge_json(raw) == expected


def test_build_judge_none_and_unknown():
    assert build_judge({"backend": "none"}) is None
    with pytest.raises(ValueError):
        build_judge({"backend": "gpt-judge"})
