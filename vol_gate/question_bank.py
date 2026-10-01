"""Builds the held-out set of genuinely-correct, genuinely-answerable
plain-English questions used for the 200-case false-block check and for
drawing baseline parameters that broken_bank.py mutates into deliberately
broken findings.

Every question text produced here matches router._PATTERNS exactly (fixed
marker-phrase templates, the same disclosed limitation the portfolio's
other NL-layer repos carry), and every (shape, params) pair is confirmed
answerable by actually building the finding before it is added to the set,
so "200 correct findings" means 200 findings this build actually produced
without error, not 200 assumed-valid specifications.
"""
from __future__ import annotations

import sqlite3

from . import finding
from .ledger import TrialLedger


def get_tickers(conn: sqlite3.Connection, cohort: str | None = None) -> list[str]:
    if cohort:
        rows = conn.execute("SELECT ticker_id FROM tickers WHERE cohort = ? ORDER BY ticker_id", (cohort,))
    else:
        rows = conn.execute("SELECT ticker_id FROM tickers ORDER BY ticker_id")
    return [r[0] for r in rows.fetchall()]


def get_session_dates(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT session_date FROM sessions ORDER BY session_id").fetchall()
    return [r[0] for r in rows]


def get_event_count(conn: sqlite3.Connection, ticker: str) -> int:
    (n,) = conn.execute("SELECT COUNT(*) FROM events WHERE ticker_id = ?", (ticker,)).fetchone()
    return n


_RIGHT_CODE = {"put": "P", "call": "C"}


def _candidates(conn: sqlite3.Connection):
    """Yields (question_text, shape, params) candidates, round-robin across
    all ten shapes, in a fixed deterministic order."""
    tickers = get_tickers(conn)
    dates = get_session_dates(conn)
    windows = [(dates[0], dates[59]), (dates[60], dates[119]), (dates[10], dates[109])]
    rights = ["put", "call"]
    deltas = [10, 25, 50, 75, 90]
    tenors = [30, 60]
    snapshot_dates = [dates[5], dates[40], dates[75], dates[100], dates[115]]

    def median_change():
        for t in tickers:
            for r in rights:
                for d in deltas:
                    for s, e in windows:
                        q = f"Did median {d}-delta {r} implied vol for {t} rise between {s} and {e}?"
                        yield q, "median_change", dict(ticker=t, right=_RIGHT_CODE[r], delta_bucket=d,
                                                        tenor_days=30, start_date=s, end_date=e)

    def cohort_compare():
        for r in rights:
            for d in deltas:
                for s, e in windows:
                    q = (f"Is there a significant difference in {d}-delta {r} implied vol regime between "
                         f"cohort A and cohort B between {s} and {e}?")
                    yield q, "cohort_compare", dict(cohort_a="A", cohort_b="B", right=_RIGHT_CODE[r],
                                                     delta_bucket=d, tenor_days=30, start_date=s, end_date=e)

    def correlation():
        pairs = [(tickers[i], tickers[i + 1]) for i in range(0, len(tickers) - 1, 2)]
        for ta, tb in pairs:
            for r in rights:
                for d in deltas:
                    for s, e in windows:
                        q = (f"What is the correlation between {ta} and {tb} {d}-delta {r} implied vol "
                             f"between {s} and {e}?")
                        yield q, "correlation", dict(ticker_a=ta, ticker_b=tb, right=_RIGHT_CODE[r],
                                                      delta_bucket=d, tenor_days=30, start_date=s, end_date=e)

    def event_window():
        for t in tickers:
            n_events = get_event_count(conn, t)
            for idx in range(min(2, n_events)):
                for r in rights:
                    for d in [10, 25, 50]:
                        for w in [5, 8]:
                            q = (f"How did {t}'s {d}-delta {r} implied vol change around event #{idx} "
                                 f"within {w} sessions?")
                            yield q, "event_window", dict(ticker=t, event_index=idx, right=_RIGHT_CODE[r],
                                                           delta_bucket=d, tenor_days=30, window=w)

    def term_structure():
        for t in tickers:
            for r in rights:
                for d in deltas:
                    for sd in snapshot_dates:
                        q = f"What is {t}'s {d}-delta {r} term structure slope on {sd}?"
                        yield q, "term_structure", dict(ticker=t, right=_RIGHT_CODE[r], delta_bucket=d,
                                                         session_date=sd)

    def skew():
        for t in tickers:
            for tn in tenors:
                for sd in snapshot_dates:
                    q = f"What is {t}'s {tn}-day put skew on {sd}?"
                    yield q, "skew", dict(ticker=t, tenor_days=tn, session_date=sd)

    def percentile_rank():
        for t in tickers:
            for r in rights:
                for d in deltas:
                    for sd in [dates[50], dates[90], dates[115]]:
                        q = (f"Where does {t}'s {d}-delta {r} 30-day implied vol on {sd} rank over the "
                             f"trailing 30 sessions?")
                        yield q, "percentile_rank", dict(ticker=t, right=_RIGHT_CODE[r], delta_bucket=d,
                                                          tenor_days=30, session_date=sd, lookback_sessions=30)

    def dividend_decile():
        for t in tickers:
            q = f"What dividend decile is {t} in?"
            yield q, "dividend_decile", dict(ticker=t)

    def contract_lookup():
        for t in tickers:
            for r in rights:
                for d in deltas:
                    for sd in snapshot_dates:
                        q = f"What is {t}'s {d}-delta {r} 30-day implied vol on {sd}?"
                        yield q, "contract_lookup", dict(ticker=t, right=_RIGHT_CODE[r], delta_bucket=d,
                                                          tenor_days=30, session_date=sd)

    def list_sessions_count():
        for t in tickers:
            for s, e in windows:
                q = f"How many trading sessions does {t} have between {s} and {e}?"
                yield q, "list_sessions_count", dict(ticker=t, start_date=s, end_date=e)

    generators = [median_change(), cohort_compare(), correlation(), event_window(), term_structure(),
                  skew(), percentile_rank(), dividend_decile(), contract_lookup(), list_sessions_count()]
    exhausted = [False] * len(generators)
    while not all(exhausted):
        for i, gen in enumerate(generators):
            if exhausted[i]:
                continue
            try:
                yield next(gen)
            except StopIteration:
                exhausted[i] = True


def build_correct_question_set(conn: sqlite3.Connection, ledger: TrialLedger, n: int = 200) -> list[dict]:
    """Confirms each candidate is genuinely answerable (by actually calling
    build_finding) before adding it to the set. Answerable candidates whose
    shape is hypothesis-bearing also, correctly, grow the trial ledger as a
    side effect, exactly like a real research session would."""
    out = []
    for question_text, shape, params in _candidates(conn):
        if len(out) >= n:
            break
        try:
            f = finding.build_finding(shape, conn, ledger, params)
        except Exception:
            continue
        out.append({"question": question_text, "shape": shape, "params": params, "finding": f})
    return out
