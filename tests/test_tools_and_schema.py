import pytest

from vol_gate import tools
from vol_gate.finding import SHAPES, build_finding
from vol_gate.ledger import TrialLedger


def test_every_tool_returns_row_ids(conn):
    r = tools.get_vol_series(conn, "SYN-A01", "P", 25, 30, "2025-01-02", "2025-03-27")
    assert len(r.row_ids) > 0
    r2 = tools.cohort_membership(conn, "A")
    assert len(r2.row_ids) == 8


def test_missing_object_raises_not_null(conn):
    with pytest.raises(tools.ObjectNotFoundError):
        tools.get_vol_series(conn, "NO-SUCH-TICKER", "P", 25, 30, "2025-01-02", "2025-03-27")


def test_all_shapes_are_registered():
    # 10 base option-chain shapes + 5 added by the Oct. 2026
    # volatility-store extension (see vol_gate/hypothesis_bank.py)
    assert len(SHAPES) == 15


def test_build_finding_dispatch_hypothesis_shape_uses_ledger(conn):
    ledger = TrialLedger(conn)
    f = build_finding("cohort_compare", conn, ledger,
                       dict(cohort_a="A", cohort_b="B", right="P", delta_bucket=25, tenor_days=30,
                            start_date="2025-01-02", end_date="2025-03-27"))
    assert ledger.size() == 1
    assert any(c.kind == "ledger_adjusted_pvalue" for c in f.claims)


def test_claims_all_have_required_fields(conn):
    ledger = TrialLedger(conn)
    f = build_finding("skew", conn, ledger, dict(ticker="SYN-A01", tenor_days=30, session_date="2025-01-09"))
    for c in f.claims:
        assert isinstance(c.label, str) and c.label
        assert isinstance(c.kind, str) and c.kind
        assert isinstance(c.params, dict)
        assert isinstance(c.value, (int, float))
