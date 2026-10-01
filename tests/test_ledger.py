from vol_gate.ledger import TrialLedger


def test_ledger_persists_and_grows_cumulatively(conn):
    ledger = TrialLedger(conn)
    assert ledger.size() == 0
    e1 = ledger.record_and_adjust("q1", 0.04)
    assert e1.ledger_size_at_report == 1
    assert e1.adjusted_p == 0.04
    e2 = ledger.record_and_adjust("q2", 0.01)
    assert e2.ledger_size_at_report == 2
    # with n=2, Holm multiplies the smaller p by 2
    assert abs(e2.adjusted_p - 0.02) < 1e-12
    assert ledger.size() == 2


def test_same_raw_p_adjusted_harder_as_ledger_grows():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    ledger = TrialLedger(conn)
    first = ledger.record_and_adjust("q1", 0.03)
    for i in range(10):
        ledger.record_and_adjust(f"filler{i}", 0.5)
    later = ledger.record_and_adjust("q_same_raw", 0.03)
    assert later.ledger_size_at_report > first.ledger_size_at_report
    assert later.adjusted_p >= first.adjusted_p
