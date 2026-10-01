"""A small-scale sanity version of the 60/60 and 0/200 gate battery, fast
enough to run in the default test suite; the full-scale numbers are
produced by scripts/run_gate_battery.py and committed under docs/."""
from vol_gate import broken_bank, question_bank
from vol_gate.gate import run_gate
from vol_gate.ledger import TrialLedger


def test_small_correct_set_never_false_blocked(conn):
    ledger = TrialLedger(conn)
    pool = question_bank.build_correct_question_set(conn, ledger, n=25)
    assert len(pool) == 25
    for item in pool:
        result = run_gate(conn, item["finding"])
        assert result.approved, f"false block on {item['question']}: {result.text}"


def test_small_broken_set_always_caught(conn):
    ledger = TrialLedger(conn)
    pool = question_bank.build_correct_question_set(conn, ledger, n=25)
    broken = broken_bank.build_broken_set(conn, ledger, pool, n_fabricated=6, n_silent=6)
    assert len(broken) == 12
    for item in broken:
        result = run_gate(conn, item["finding"])
        assert not result.approved, f"missed a {item['label']} finding: {item['finding'].text}"
        assert "WITHHELD" in result.text
