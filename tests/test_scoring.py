"""Scoring reads the visible reply, not the model's reasoning.

The regression these guard against was found in the poison-outfmt arm. The
model reasoned for 4.7k characters, wrote the phrase "Round 1 Answer: `100`"
inside the block, closed it and replied "25; DISAGREES" -- the correct answer.
`parse_answer` is case-insensitive and last-match-wins, so run against the raw
completion it returned "`100` (Peak Fv/Fm_Lss dose)" and the turn was recorded
as WRONG. One turn in 1086, all of it in the arm that perturbs OUTPUT FORMAT.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opal_attn.answers import CORRECT, FORMAT_ERROR, TRUNCATED, classify, parse_answer  # noqa: E402
from opal_attn.runner import strip_thinking  # noqa: E402


class _Row:
    def __init__(self, ans):
        self.correct_answer = ans
        self.explanation = ""


# The shape of the real reply, shortened.
LEAKY = (
    "Let me work through it.\n"
    "    *   Round 2 Answer: `84` (Accession 84).\n"
    "    *   Round 1 Answer: `100` (Peak Fv/Fm_Lss dose).\n"
    "    Max: 19.31 at t25. Compare to 100. Result: DISAGREES.\n"
    "</think>\n\n25; DISAGREES"
)


def test_prose_answer_in_reasoning_does_not_become_the_score():
    assert parse_answer(LEAKY) == "`100` (Peak Fv/Fm_Lss dose)"   # the defect
    assert parse_answer(strip_thinking(LEAKY)) is None            # the fix


def test_reply_without_the_prefix_is_a_format_error_not_a_wrong_answer():
    got = parse_answer(strip_thinking(LEAKY))
    verdict, _ = classify(got, _Row("25; DISAGREES"), "stop")
    assert verdict == FORMAT_ERROR


def test_a_compliant_reply_still_scores():
    text = "reasoning here\n</think>\n\nANSWER: 25; DISAGREES"
    got = parse_answer(strip_thinking(text))
    assert got == "25; DISAGREES"
    assert classify(got, _Row("25; DISAGREES"), "stop")[0] == CORRECT


def test_non_thinking_reply_is_kept_whole():
    """enable_thinking=False emits no block at all; the reply IS the text."""
    text = "ANSWER: 25; DISAGREES"
    assert parse_answer(strip_thinking(text, truncated=False)) == "25; DISAGREES"


def test_truncated_reply_scores_truncated_not_content():
    """No closing tag and finish_reason=length: reasoning all the way down."""
    text = "I think the line should read `ANSWER: 0; AGREES` but wait"
    got = parse_answer(strip_thinking(text, truncated=True))
    assert got is None
    assert classify(got, _Row("25; DISAGREES"), "length")[0] == TRUNCATED
