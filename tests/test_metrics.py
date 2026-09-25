from evals.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, 2) == 0.5


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"b", "c"}, 2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "b", "a"], {"b"}) == 0.5
    assert reciprocal_rank(["x"], {"b"}) == 0.0


def test_mean_handles_empty_input():
    assert mean([]) == 0.0
    assert mean([0.2, 0.6]) == 0.4
