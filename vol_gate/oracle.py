"""The verification gate's oracle: independently recomputes every numeric
claim in a Finding and reports exactly which figure disagrees, if any.

Why this module shares no code with tools.py, stats.py, finding.py, or
ledger.holm_bonferroni: the entire point of the gate is to catch a bug in
THAT side (a wrong date range, an off-by-one window, a stale cached result,
a wrong join, or an outright fabricated number). If the oracle called the
same function to check its own answer, a bug in that function would agree
with itself every time and the gate would catch nothing. So every query
here is written fresh, directly against sqlite3, with its own window-split
logic, its own join, its own Holm-Bonferroni reimplementation
(`_oracle_holm_bonferroni`). tests/test_oracle_independence.py asserts this
module imports none of vol_gate.tools, vol_gate.finding, or
vol_gate.ledger.holm_bonferroni, and that breaking the executor's query
intentionally still gets caught here.
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass

from scipy import stats as _scipy_stats


@dataclass
class Disagreement:
    label: str
    claimed: float
    recomputed: float
    diff: float


@dataclass
class OracleResult:
    ok: bool
    disagreements: list[Disagreement]


def _oracle_holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Independent reimplementation of Holm-Bonferroni; deliberately not
    importing vol_gate.holm.holm_bonferroni."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: pvalues[i])
    out = [0.0] * n
    best_so_far = 0.0
    for k, i in enumerate(indexed):
        rank = k + 1
        candidate = pvalues[i] * (n - rank + 1)
        if candidate > best_so_far:
            best_so_far = candidate
        out[i] = min(1.0, best_so_far)
    return out


def _sessions_for_window(conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int,
                          tenor_days: int, start_date: str, end_date: str) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT s.session_date, q.implied_vol "
        "FROM iv_quotes q, sessions s "
        "WHERE q.session_id = s.session_id AND q.ticker_id = :t AND q.opt_right = :r "
        "AND q.delta_bucket = :d AND q.tenor_days = :tn "
        "AND s.session_date >= :sd AND s.session_date <= :ed "
        "ORDER BY s.session_date ASC",
        {"t": ticker, "r": right, "d": delta_bucket, "tn": tenor_days, "sd": start_date, "ed": end_date},
    ).fetchall()
    return rows


def _recompute_window_median(conn: sqlite3.Connection, params: dict) -> float:
    rows = _sessions_for_window(conn, params["ticker"], params["right"], params["delta_bucket"],
                                 params["tenor_days"], params["start_date"], params["end_date"])
    vals = [v for _, v in rows]
    n = len(vals)
    half = n // 2
    if params["half"] == "first":
        chunk = vals[0:half]
    else:
        chunk = vals[n - half: n]
    return statistics.median(chunk)


def _recompute_window_change(conn: sqlite3.Connection, params: dict) -> float:
    first = _recompute_window_median(conn, {**params, "half": "first"})
    last = _recompute_window_median(conn, {**params, "half": "last"})
    return last - first


def _recompute_iv_level(conn: sqlite3.Connection, params: dict) -> float:
    row = conn.execute(
        "SELECT q.implied_vol FROM iv_quotes q, sessions s "
        "WHERE q.session_id = s.session_id AND q.ticker_id = ? AND q.opt_right = ? "
        "AND q.delta_bucket = ? AND q.tenor_days = ? AND s.session_date = ?",
        (params["ticker"], params["right"], params["delta_bucket"], params["tenor_days"],
         params["session_date"]),
    ).fetchone()
    if row is None:
        raise LookupError("oracle found no matching contract row")
    return row[0]


def _recompute_term_slope(conn: sqlite3.Connection, params: dict) -> float:
    iv30 = _recompute_iv_level(conn, {**params, "tenor_days": 30})
    iv60 = _recompute_iv_level(conn, {**params, "tenor_days": 60})
    return iv60 - iv30


def _recompute_skew(conn: sqlite3.Connection, params: dict) -> float:
    atm = _recompute_iv_level(conn, {"ticker": params["ticker"], "right": "P", "delta_bucket": 50,
                                      "tenor_days": params["tenor_days"], "session_date": params["session_date"]})
    otm = _recompute_iv_level(conn, {"ticker": params["ticker"], "right": "P", "delta_bucket": 10,
                                      "tenor_days": params["tenor_days"], "session_date": params["session_date"]})
    return otm - atm


def _recompute_percentile_rank(conn: sqlite3.Connection, params: dict) -> float:
    rows = conn.execute(
        "SELECT s.session_date, q.implied_vol FROM iv_quotes q, sessions s "
        "WHERE q.session_id = s.session_id AND q.ticker_id = ? AND q.opt_right = ? "
        "AND q.delta_bucket = ? AND q.tenor_days = ? AND s.session_date <= ? "
        "ORDER BY s.session_date DESC LIMIT ?",
        (params["ticker"], params["right"], params["delta_bucket"], params["tenor_days"],
         params["session_date"], params["lookback_sessions"]),
    ).fetchall()
    if not rows or rows[0][0] != params["session_date"]:
        raise LookupError("oracle found no exact-date row for percentile rank")
    current = rows[0][1]
    window = sorted(v for _, v in rows)
    n_leq = sum(1 for v in window if v <= current)
    return 100.0 * n_leq / len(window)


def _recompute_event_window_mean(conn: sqlite3.Connection, params: dict) -> float:
    ev_rows = conn.execute(
        "SELECT session_id FROM events WHERE ticker_id = ? ORDER BY session_id", (params["ticker"],)
    ).fetchall()
    idx = params["event_index"]
    if idx < 0 or idx >= len(ev_rows):
        raise LookupError("oracle found no such event index")
    event_session_id = ev_rows[idx][0]
    window = params["window"]
    if params["side"] == "before":
        lo, hi = event_session_id - window, event_session_id - 1
    else:
        lo, hi = event_session_id + 1, event_session_id + window
    rows = conn.execute(
        "SELECT implied_vol FROM iv_quotes WHERE ticker_id = ? AND opt_right = ? AND delta_bucket = ? "
        "AND tenor_days = ? AND session_id >= ? AND session_id <= ?",
        (params["ticker"], params["right"], params["delta_bucket"], params["tenor_days"], lo, hi),
    ).fetchall()
    if not rows:
        raise LookupError("oracle found no rows in event window")
    return statistics.mean(r[0] for r in rows)


def _recompute_event_window_diff(conn: sqlite3.Connection, params: dict) -> float:
    after = _recompute_event_window_mean(conn, {**params, "side": "after"})
    before = _recompute_event_window_mean(conn, {**params, "side": "before"})
    return after - before


def _recompute_dividend_decile(conn: sqlite3.Connection, params: dict) -> float:
    row = conn.execute("SELECT dividend_decile FROM tickers WHERE ticker_id = ?", (params["ticker"],)).fetchone()
    if row is None:
        raise LookupError("oracle found no such ticker")
    return row[0]


def _recompute_dividend_rank(conn: sqlite3.Connection, params: dict) -> float:
    rows = conn.execute("SELECT ticker_id, dividend_decile FROM tickers").fetchall()
    ordered = sorted(rows, key=lambda r: (-r[1], r[0]))
    for i, (tkr, _) in enumerate(ordered, start=1):
        if tkr == params["ticker"]:
            return i
    raise LookupError("oracle found no such ticker")


def _recompute_session_count(conn: sqlite3.Connection, params: dict) -> float:
    (n,) = conn.execute(
        "SELECT COUNT(DISTINCT s.session_id) FROM sessions s, iv_quotes q "
        "WHERE q.session_id = s.session_id AND q.ticker_id = ? AND s.session_date >= ? AND s.session_date <= ?",
        (params["ticker"], params["start_date"], params["end_date"]),
    ).fetchone()
    return n


def _recompute_cohort_mean_of_medians(conn: sqlite3.Connection, params: dict) -> float:
    tickers = [r[0] for r in conn.execute(
        "SELECT ticker_id FROM tickers WHERE cohort = ? ORDER BY ticker_id", (params["cohort"],)
    ).fetchall()]
    medians = []
    for t in tickers:
        rows = _sessions_for_window(conn, t, params["right"], params["delta_bucket"], params["tenor_days"],
                                     params["start_date"], params["end_date"])
        medians.append(statistics.median(v for _, v in rows))
    return statistics.mean(medians)


def _cohort_medians(conn: sqlite3.Connection, cohort: str, right: str, delta_bucket: int, tenor_days: int,
                     start_date: str, end_date: str) -> list[float]:
    tickers = [r[0] for r in conn.execute(
        "SELECT ticker_id FROM tickers WHERE cohort = ? ORDER BY ticker_id", (cohort,)
    ).fetchall()]
    out = []
    for t in tickers:
        rows = _sessions_for_window(conn, t, right, delta_bucket, tenor_days, start_date, end_date)
        out.append(statistics.median(v for _, v in rows))
    return out


def _recompute_two_sample_pvalue(conn: sqlite3.Connection, params: dict) -> float:
    a = _cohort_medians(conn, params["cohort_a"], params["right"], params["delta_bucket"],
                         params["tenor_days"], params["start_date"], params["end_date"])
    b = _cohort_medians(conn, params["cohort_b"], params["right"], params["delta_bucket"],
                         params["tenor_days"], params["start_date"], params["end_date"])
    result = _scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(result.pvalue)


def _paired_series(conn: sqlite3.Connection, ticker_a: str, ticker_b: str, right: str, delta_bucket: int,
                    tenor_days: int, start_date: str, end_date: str) -> tuple[list[float], list[float]]:
    rows = conn.execute(
        "SELECT qa.session_id, qa.implied_vol, qb.implied_vol "
        "FROM iv_quotes qa, iv_quotes qb, sessions s "
        "WHERE qa.session_id = qb.session_id AND qa.session_id = s.session_id "
        "AND qa.ticker_id = ? AND qb.ticker_id = ? "
        "AND qa.opt_right = ? AND qb.opt_right = qa.opt_right "
        "AND qa.delta_bucket = ? AND qb.delta_bucket = qa.delta_bucket "
        "AND qa.tenor_days = ? AND qb.tenor_days = qa.tenor_days "
        "AND s.session_date >= ? AND s.session_date <= ? "
        "ORDER BY qa.session_id",
        (ticker_a, ticker_b, right, delta_bucket, tenor_days, start_date, end_date),
    ).fetchall()
    return [r[1] for r in rows], [r[2] for r in rows]


def _recompute_correlation_r(conn: sqlite3.Connection, params: dict) -> float:
    a, b = _paired_series(conn, params["ticker_a"], params["ticker_b"], params["right"], params["delta_bucket"],
                           params["tenor_days"], params["start_date"], params["end_date"])
    result = _scipy_stats.pearsonr(a, b)
    return float(result.statistic)


def _recompute_correlation_pvalue(conn: sqlite3.Connection, params: dict) -> float:
    a, b = _paired_series(conn, params["ticker_a"], params["ticker_b"], params["right"], params["delta_bucket"],
                           params["tenor_days"], params["start_date"], params["end_date"])
    result = _scipy_stats.pearsonr(a, b)
    return float(result.pvalue)


def _recompute_ledger_adjusted_pvalue(conn: sqlite3.Connection, params: dict) -> float:
    ledger_id = params["ledger_id"]
    rows = conn.execute(
        "SELECT ledger_id, raw_p FROM trial_ledger WHERE ledger_id <= ? ORDER BY ledger_id", (ledger_id,)
    ).fetchall()
    ids = [r[0] for r in rows]
    raw_ps = [r[1] for r in rows]
    adjusted = _oracle_holm_bonferroni(raw_ps)
    pos = ids.index(ledger_id)
    return adjusted[pos]


_RECOMPUTE = {
    "window_median": _recompute_window_median,
    "window_change": _recompute_window_change,
    "iv_level": _recompute_iv_level,
    "term_slope": _recompute_term_slope,
    "skew": _recompute_skew,
    "percentile_rank": _recompute_percentile_rank,
    "event_window_mean": _recompute_event_window_mean,
    "event_window_diff": _recompute_event_window_diff,
    "dividend_decile": _recompute_dividend_decile,
    "dividend_rank": _recompute_dividend_rank,
    "session_count": _recompute_session_count,
    "cohort_mean_of_medians": _recompute_cohort_mean_of_medians,
    "two_sample_pvalue": _recompute_two_sample_pvalue,
    "correlation_r": _recompute_correlation_r,
    "correlation_pvalue": _recompute_correlation_pvalue,
    "ledger_adjusted_pvalue": _recompute_ledger_adjusted_pvalue,
}


def verify(conn: sqlite3.Connection, finding) -> OracleResult:
    disagreements = []
    for claim in finding.claims:
        fn = _RECOMPUTE.get(claim.kind)
        if fn is None:
            disagreements.append(Disagreement(claim.label, claim.value, float("nan"), float("nan")))
            continue
        try:
            recomputed = fn(conn, claim.params)
        except LookupError:
            disagreements.append(Disagreement(claim.label, claim.value, float("nan"), float("nan")))
            continue
        diff = abs(recomputed - claim.value)
        if diff > claim.tol:
            disagreements.append(Disagreement(claim.label, claim.value, recomputed, diff))
    return OracleResult(ok=(len(disagreements) == 0), disagreements=disagreements)
