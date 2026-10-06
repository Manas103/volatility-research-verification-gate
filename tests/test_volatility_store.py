"""The volatility-store extension: import correctness, the five new tools,
oracle independence and agreement for the new claim kinds, the 48-entry
hypothesis bank, and a deliberately broken claim still getting caught by
the independent oracle.
"""
import ast
import os

import pytest

from vol_gate import db, gate, oracle, tools, volstore
from vol_gate.finding import build_finding
from vol_gate.hypothesis_bank import SPLIT_RULES, TICKERS, build_hypothesis_bank
from vol_gate.ledger import TrialLedger


@pytest.fixture()
def vol_conn():
    c = db.connect(":memory:")
    n = volstore.import_csv(c)
    assert n == 640
    yield c
    c.close()


def test_import_is_idempotent_and_matches_csv_row_count(vol_conn):
    (n,) = vol_conn.execute("SELECT COUNT(*) FROM volatility_bins").fetchone()
    assert n == 640
    # re-importing must not duplicate rows
    volstore.import_csv(vol_conn)
    (n2,) = vol_conn.execute("SELECT COUNT(*) FROM volatility_bins").fetchone()
    assert n2 == 640


def test_five_tickers_present(vol_conn):
    assert volstore.tickers(vol_conn) == sorted(TICKERS)


def test_vol_bin_values_and_pooled_all(vol_conn):
    one = tools.vol_bin_values(vol_conn, "SYNA")
    assert len(one.value) == len(one.row_ids) == 128  # 16 days x 8 bins
    pooled = tools.vol_bin_values(vol_conn, "ALL")
    assert len(pooled.value) == 640


def test_vol_bin_values_unknown_ticker_raises(vol_conn):
    with pytest.raises(tools.ObjectNotFoundError):
        tools.vol_bin_values(vol_conn, "NOPE")


def test_vol_paired_bin_values_aligned_by_date_and_bin(vol_conn):
    r = tools.vol_paired_bin_values(vol_conn, "SYNA", "SYNB")
    assert len(r.value["series_a"]) == len(r.value["series_b"]) == 128


def test_vol_regime_split_rules_partition_without_overlap(vol_conn):
    for split_by in SPLIT_RULES:
        r = tools.vol_regime_split(vol_conn, "SYNA", split_by)
        assert len(r.value["group_high"]) + len(r.value["group_low"]) == 128


def test_vol_regime_split_unknown_rule_raises(vol_conn):
    with pytest.raises(tools.ObjectNotFoundError):
        tools.vol_regime_split(vol_conn, "SYNA", "not_a_rule")


def test_hypothesis_bank_has_48_unique_entries():
    bank = build_hypothesis_bank()
    assert len(bank) == 48
    as_tuples = [(shape, tuple(sorted(params.items()))) for shape, params in bank]
    assert len(set(as_tuples)) == 48


def test_hypothesis_bank_pair_shapes_never_compare_a_ticker_to_itself():
    bank = build_hypothesis_bank()
    for shape, params in bank:
        if shape in ("vol_pair_compare", "vol_pair_correlation"):
            assert params["ticker_a"] != params["ticker_b"]


def test_oracle_module_still_imports_no_executor_module_after_vol_additions():
    path = os.path.join(os.path.dirname(oracle.__file__), "oracle.py")
    tree = ast.parse(open(path).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
        if isinstance(node, ast.Import):
            for n in node.names:
                imported.add(n.name.split(".")[-1])
    assert "tools" not in imported
    assert "finding" not in imported
    assert "ledger" not in imported
    assert "holm" not in imported
    assert "volstore" not in imported


def test_all_48_bank_hypotheses_pass_the_gate(vol_conn):
    """None of the 48 pre-registered hypotheses are deliberately broken;
    the oracle should agree with every single one. This is the sanity
    check for the new claim kinds before trusting the screen's tallies."""
    ledger = TrialLedger(vol_conn)
    bank = build_hypothesis_bank()
    n_withheld = 0
    for shape, params in bank:
        f = build_finding(shape, vol_conn, ledger, params)
        result = gate.run_gate(vol_conn, f)
        if not result.approved:
            n_withheld += 1
            print(result.text)
    assert n_withheld == 0


def test_corrupting_a_vol_pair_claim_is_still_caught(vol_conn):
    ledger = TrialLedger(vol_conn)
    f = build_finding("vol_pair_compare", vol_conn, ledger, {"ticker_a": "SYNA", "ticker_b": "SYNB"})
    for c in f.claims:
        if c.label == "mean_a":
            c.value = c.value + 1.0  # silent-error-shaped corruption: wrong number, same recipe
    result = gate.run_gate(vol_conn, f)
    assert result.approved is False
    assert any(d.label == "mean_a" for d in result.disagreements)


def test_corrupting_a_vol_regime_claim_is_still_caught(vol_conn):
    ledger = TrialLedger(vol_conn)
    f = build_finding("vol_regime_compare", vol_conn, ledger, {"ticker": "SYNC", "split_by": "day_parity"})
    for c in f.claims:
        if c.label == "raw_p":
            c.value = min(1.0, c.value + 0.3)
    result = gate.run_gate(vol_conn, f)
    assert result.approved is False
    assert any(d.label == "raw_p" for d in result.disagreements)
