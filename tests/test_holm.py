from vol_gate.holm import holm_bonferroni


def test_hand_computed_five_hypotheses():
    raw = [0.01, 0.02, 0.03, 0.04, 0.005]
    expected = [0.04, 0.06, 0.06, 0.06, 0.025]
    got = holm_bonferroni(raw)
    for a, b in zip(got, expected):
        assert abs(a - b) < 1e-12


def test_monotonic_nondecreasing_by_rank():
    raw = [0.5, 0.001, 0.3, 0.02, 0.1]
    adj = holm_bonferroni(raw)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    vals = [adj[i] for i in order]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_empty_list():
    assert holm_bonferroni([]) == []


def test_single_pvalue_unchanged():
    assert holm_bonferroni([0.03]) == [0.03]
