import pytest

from rl.rewards.composer import RewardComposer


class Const:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def __call__(self, sample):
        return self.value


def kwargs(n):
    return dict(side=["left"] * n, question=["q"] * n, opponent_response=[""] * n, own_history=[[]] * n)


def test_weighted_sum_and_breakdown():
    c = RewardComposer([(Const("a", 1.0), 0.5), (Const("b", -2.0), 1.0)])
    fn = c.as_trl_reward_fn(log_components=False)
    assert fn(prompts=["p"], completions=["x"], **kwargs(1)) == [pytest.approx(-1.5)]


def test_accepts_conversational_completions():
    c = RewardComposer([(Const("a", 1.0), 1.0)])
    fn = c.as_trl_reward_fn(log_components=False)
    out = fn(prompts=["p"], completions=[[{"role": "assistant", "content": "hi"}]], **kwargs(1))
    assert out == [1.0]


def test_logs_mean_raw_value_per_component():
    logged = {}
    c = RewardComposer([(Const("a", 3.0), 0.1)])
    fn = c.as_trl_reward_fn()
    fn(prompts=["p", "p"], completions=["x", "y"], log_metric=lambda k, v: logged.update({k: v}), **kwargs(2))
    assert logged == {"reward_components/a": 3.0}


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="version"):
        RewardComposer.from_config({"version": "v9", "judge": {"backend": "none"}})
