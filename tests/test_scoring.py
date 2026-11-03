import pytest

from neuropsych_bench import Evaluator
from neuropsych_bench.scoring import extract_answer


def score(task, answer_key, response):
    evaluator = Evaluator()
    info = {"task": task, "task_type": "t", "stage": "s", "process": "p"}
    evaluator.evaluate((info, [{"trial_id": "trial_001", "answer_key": answer_key, "model_response": response}]))
    return bool(evaluator.get_trials()["correct"].iloc[0])


@pytest.mark.parametrize("task, key, response, correct", [
    # exact match
    ("borb_orientation", "yes", "{yes}", True),
    ("borb_orientation", "yes", "{No}", False),
    ("borb_orientation", "yes", "yes", False),  # no braces, no answer
    ("left_embedded_figures", "3", "I pick {3}", True),
    ("borb_size_comparison", "same", "{Same}", True),
    # naming, with aliases
    ("mindset_linedrawings", "crocodile", "{alligator}", True),
    ("mindset_linedrawings", "crocodile", "{a big croc}", True),
    ("mindset_linedrawings", "crocodile", "{lizard}", False),
    ("borb_triplet_letters", "EJA", "{EJA}", True),
    # letters in any order
    ("borb_overlapped_letters", "EJA", "{AJE}", True),
    ("borb_overlapped_letters", "EJA", "{AJF}", False),
    # shapes in any order
    ("borb_overlapped_shapes", ["plus", "pentagon", "triangle"], "{triangle, plus, pentagon}", True),
    ("borb_overlapped_shapes", ["plus", "pentagon", "triangle"], "{triangle, plus}", False),
    # shapes in the given order
    ("borb_triplet_shapes", ["plus", "pentagon", "triangle"], "{plus, pentagon, triangle}", True),
    ("borb_triplet_shapes", ["plus", "pentagon", "triangle"], "{triangle, plus, pentagon}", False),
    # refusal or empty response
    ("borb_orientation", "yes", None, False),
])
def test_marking(task, key, response, correct):
    assert score(task, key, response) is correct


def test_extract_answer():
    assert extract_answer("x {A} y {B}") == "a"
    assert extract_answer("{ }") is None
    assert extract_answer(None) is None


def test_unknown_task():
    with pytest.raises(ValueError):
        score("not_a_task", "yes", "{yes}")


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main(["-v", __file__]))
