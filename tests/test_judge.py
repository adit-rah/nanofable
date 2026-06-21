from eval.judge import AXES, parse_judge_json, per_completion_score


class FakeJudge:
    """Implements the JudgeBackend protocol without any model download."""

    def score(self, prefix: str, completion: str) -> dict:
        return {"grammar": 5, "consistency": 4, "completes": 4}


def test_parse_clean_json():
    s = '{"grammar": 5, "consistency": 4, "completes": 3}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 4, "completes": 3}


def test_parse_messy_text_with_prose_and_fence():
    s = 'Sure! Here is my rating:\n```json\n{"grammar": 4, "consistency": 5, "completes": 5}\n```\nThanks.'
    assert parse_judge_json(s) == {"grammar": 4, "consistency": 5, "completes": 5}


def test_parse_clamps_out_of_range():
    s = '{"grammar": 9, "consistency": -2, "completes": 3}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 0, "completes": 3}


def test_parse_missing_axis_defaults_zero():
    s = '{"grammar": 5}'
    assert parse_judge_json(s) == {"grammar": 5, "consistency": 0, "completes": 0}


def test_fake_judge_protocol_shape():
    j = FakeJudge()
    out = j.score("prefix", "completion")
    assert set(out) == set(AXES)
    assert per_completion_score(out) == (5 + 4 + 4) / 3
