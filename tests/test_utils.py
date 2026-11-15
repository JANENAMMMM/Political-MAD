import pytest

from rl.rewards.utils import extract_key_points, jaccard_similarity, key_point_coverage, tokenize


def test_tokenize_drops_stopwords_and_single_chars():
    assert tokenize("그리고 최저임금 a 인상") == ["최저임금", "인상"]


def test_jaccard_is_zero_without_history():
    assert jaccard_similarity("최저임금 인상", []) == 0.0


def test_jaccard_identical_text_is_one():
    assert jaccard_similarity("최저임금 인상 필요", ["최저임금 인상 필요"]) == pytest.approx(1.0)


def test_jaccard_partial_overlap():
    # {최저임금, 인상} vs {최저임금, 동결} -> 1/3
    assert jaccard_similarity("최저임금 인상", ["최저임금 동결"]) == pytest.approx(1 / 3)


def test_key_points_ranked_by_frequency():
    kp = extract_key_points("세금 세금 세금 복지 복지 성장", top_n=2)
    assert kp == {"세금", "복지"}


def test_coverage_counts_opponent_points_mentioned():
    assert key_point_coverage("세금 복지", "세금 얘기만 합니다") == pytest.approx(0.5)


def test_coverage_empty_opponent_is_zero():
    assert key_point_coverage("", "아무 말") == 0.0
