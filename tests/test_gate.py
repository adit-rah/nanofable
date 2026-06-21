from eval.gate import capability_gate, mean_and_ci


def test_capable_requires_both_gates():
    assert capability_gate([5] * 200, val_ppl=10, best_fp16_ppl=10)["capable"] is True
    # ppl fail (100 > 1.5*10)
    assert capability_gate([5] * 200, val_ppl=100, best_fp16_ppl=10)["capable"] is False
    # coherence fail (mean 3 < 4)
    assert capability_gate([3] * 200, val_ppl=10, best_fp16_ppl=10)["capable"] is False


def test_ppl_threshold_is_1p5x_best_fp16():
    g = capability_gate([5] * 200, val_ppl=15.0, best_fp16_ppl=10.0)
    assert g["ppl_threshold"] == 15.0 and g["ppl_pass"] is True
    g2 = capability_gate([5] * 200, val_ppl=15.01, best_fp16_ppl=10.0)
    assert g2["ppl_pass"] is False


def test_ci_straddle_flagged():
    # Mean ~4.0 with spread -> CI straddles 4.0 -> indistinguishable.
    scores = [4.0] * 100 + [3.9, 4.1] * 50
    g = capability_gate(scores, val_ppl=10, best_fp16_ppl=10)
    assert "ci_straddles_4" in g
    assert g["ci_low"] < 4.0 < g["ci_high"]


def test_tight_high_mean_does_not_straddle():
    g = capability_gate([5] * 200, val_ppl=10, best_fp16_ppl=10)
    assert g["ci_straddles_4"] is False


def test_mean_and_ci_basic():
    mean, half = mean_and_ci([4.0] * 10)
    assert mean == 4.0 and half == 0.0  # zero variance -> zero width
