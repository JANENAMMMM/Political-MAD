import pytest

from rl.rewards import utils


@pytest.fixture(autouse=True)
def regex_tokenizer(monkeypatch):
    """kiwipiepy 설치 여부와 상관없이 같은 결과가 나오도록 정규식 토큰화로 고정한다."""
    monkeypatch.setattr(utils, "_KIWI", None)
    monkeypatch.setattr(utils, "_KIWI_FAILED", True)


class FakeEmbedder:
    """단어 집합 기반 가짜 임베더. 모델 다운로드 없이 임베딩 보상의 로직만 검사한다."""

    def _vec(self, text):
        return set(utils.tokenize(text))

    def cos_sim(self, a, b):
        if not a or not b:
            return 0.0
        va, vb = self._vec(a), self._vec(b)
        if not va or not vb:
            return 0.0
        return len(va & vb) / (len(va) * len(vb)) ** 0.5

    def encode_mean(self, texts):
        return " ".join(texts)

    def cos_sim_vec(self, text, vec):
        return self.cos_sim(text, vec)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()
