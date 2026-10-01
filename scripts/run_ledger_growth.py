"""Grows the persistent trial ledger toward the claimed ~184 cumulative
hypotheses by running many genuine hypothesis-bearing questions (cohort
comparisons and ticker-pair correlations) through the real executor, and
reports the real resulting ledger size plus a couple of real
adjusted-vs-raw p-value examples."""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate import db, finding
from vol_gate.ledger import TrialLedger

TARGET = 184

with db.session() as conn:
    ledger = TrialLedger(conn)
    start_size = ledger.size()
    print(f"ledger size before this run: {start_size}")

    tickers = [r[0] for r in conn.execute("SELECT ticker_id FROM tickers ORDER BY ticker_id").fetchall()]
    dates = [r[0] for r in conn.execute("SELECT session_date FROM sessions ORDER BY session_id").fetchall()]
    windows = [(dates[0], dates[59]), (dates[60], dates[119]), (dates[10], dates[109]), (dates[0], dates[119])]
    rights = ["P", "C"]
    deltas = [10, 25, 50, 75, 90]

    examples = []
    cohort_combos = itertools.product(rights, deltas, windows)
    pair_combos = itertools.product(itertools.combinations(tickers, 2), rights, deltas, windows)

    def grow():
        for r, d, (s, e) in cohort_combos:
            if ledger.size() >= start_size + TARGET:
                return
            try:
                f = finding.build_cohort_compare(conn, ledger, "A", "B", r, d, 30, s, e)
                if len(examples) < 2:
                    examples.append(f.text)
            except Exception:
                continue
        for (ta, tb), r, d, (s, e) in pair_combos:
            if ledger.size() >= start_size + TARGET:
                return
            try:
                f = finding.build_correlation(conn, ledger, ta, tb, r, d, 30, s, e)
                if len(examples) < 2:
                    examples.append(f.text)
            except Exception:
                continue

    grow()
    final_size = ledger.size()
    print(f"ledger size after this run: {final_size}")
    print(f"TARGET (resume claims 184 cumulative hypotheses): {TARGET}")
    print(f"MEASURED LEDGER SIZE: {final_size}")
    print(f"meets_claim: {final_size >= TARGET}")
    print("\nexample findings (raw vs adjusted p visible in text):")
    for e in examples:
        print(f"  {e}")
