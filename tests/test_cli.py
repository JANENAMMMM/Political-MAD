import json
from pathlib import Path

import pytest

from rl.rewards.cli import load_samples, main

DATA = Path(__file__).parent / "data" / "turns.jsonl"


def test_score_json_output(capsys):
    assert main(["score", str(DATA), "--format", "json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 3
    assert set(rows[0]) == {"novelty", "engagement"}
    # 2번째 턴은 자기 발언을 그대로 반복했으므로 novelty 패널티를 받아야 한다
    assert rows[1]["novelty"] < 0 == rows[0]["novelty"]


def test_table_output_has_header(capsys):
    main(["score", str(DATA)])
    assert capsys.readouterr().out.startswith("#\tside\tround\tnovelty\tengagement")


def test_diversity_requires_embedding_model():
    with pytest.raises(SystemExit, match="embedding-model"):
        main(["score", str(DATA), "--components", "diversity"])


def test_rejects_bad_side(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"response": "x", "side": "center"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="side"):
        load_samples(str(bad))


def test_rejects_unknown_field(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"response": "x", "side": "left", "speaker": "a"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="speaker"):
        load_samples(str(bad))
