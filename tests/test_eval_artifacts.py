import json
import os

from eval.judge import load_judge_prompt

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_judge_prompt_has_placeholders():
    tmpl = load_judge_prompt()
    assert "{prefix}" in tmpl and "{completion}" in tmpl
    # the template must request the three-axis JSON
    for axis in ("grammar", "consistency", "completes"):
        assert axis in tmpl


def test_rubric_exists():
    assert os.path.exists(os.path.join(ROOT, "eval", "rubric.md"))


def test_prefixes_are_exactly_200_when_present():
    path = os.path.join(ROOT, "eval", "prefixes.jsonl")
    if not os.path.exists(path):
        return  # generated later on the data-enabled environment
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 200
    assert all({"id", "prefix", "gold_continuation"} <= set(r) for r in rows)
