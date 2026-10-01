"""The entire read surface: twelve named, typed functions over the
option-chain store. Every tool returns a ToolResult whose `row_ids` is the
exact evidence (iv_quotes.row_id, or ticker/session/event ids where
appropriate) the value came from. No tool accepts free-text SQL; a question
the catalog has no tool for is a question the system correctly cannot
answer.

This module is the executor's only path into the database. gate/oracle.py
deliberately does NOT import anything from this module (see
tests/test_oracle_independence.py): the oracle recomputes every number its
own way, with its own SQL, so a bug shared between "the query that produced
the finding" and "the query that checks the finding" cannot cancel out.
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Any


class ObjectNotFoundError(Exception):
    """Raised when a tool is asked about a ticker, session, or event that
    does not exist, rather than returning a null-shaped value."""


@dataclass
class ToolResult:
    value: Any
    row_ids: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _ticker_exists(conn: sqlite3.Connection, ticker_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM tickers WHERE ticker_id = ?", (ticker_id,)
    ).fetchone() is not None


def _session_ids_in_range(conn: sqlite3.Connection, start_date: str, end_date: str) -> list[int]:
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE session_date BETWEEN ? AND ? ORDER BY session_id",
        (start_date, end_date),
    ).fetchall()
    return [r[0] for r in rows]


def list_sessions(conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str) -> ToolResult:
    if not _ticker_exists(conn, ticker):
        raise ObjectNotFoundError(f"no ticker {ticker!r}")
    rows = conn.execute(
        """SELECT DISTINCT s.session_id, s.session_date
               FROM sessions s JOIN iv_quotes q ON q.session_id = s.session_id
              WHERE q.ticker_id = ? AND s.session_date BETWEEN ? AND ?
              ORDER BY s.session_id""",
        (ticker, start_date, end_date),
    ).fetchall()
    dates = [r[1] for r in rows]
    return ToolResult(value=dates, row_ids=[r[0] for r in rows])


def cohort_membership(conn: sqlite3.Connection, cohort: str) -> ToolResult:
    if cohort not in ("A", "B"):
        raise ObjectNotFoundError(f"no cohort {cohort!r}")
    rows = conn.execute(
        "SELECT ticker_id FROM tickers WHERE cohort = ? ORDER BY ticker_id", (cohort,)
    ).fetchall()
    tickers = [r[0] for r in rows]
    if not tickers:
        raise ObjectNotFoundError(f"cohort {cohort!r} has no members")
    return ToolResult(value=tickers, row_ids=tickers)


def dividend_decile_lookup(conn: sqlite3.Connection, ticker: str) -> ToolResult:
    row = conn.execute(
        "SELECT dividend_decile FROM tickers WHERE ticker_id = ?", (ticker,)
    ).fetchone()
    if row is None:
        raise ObjectNotFoundError(f"no ticker {ticker!r}")
    decile = row[0]
    all_rows = conn.execute(
        "SELECT ticker_id, dividend_decile FROM tickers ORDER BY dividend_decile DESC, ticker_id"
    ).fetchall()
    rank = next(i for i, r in enumerate(all_rows, start=1) if r[0] == ticker)
    return ToolResult(
        value={"decile": decile, "rank": rank, "universe_size": len(all_rows)},
        row_ids=[ticker],
    )


def get_contract(
    conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int, tenor_days: int, session_date: str
) -> ToolResult:
    row = conn.execute(
        """SELECT q.row_id, q.implied_vol
               FROM iv_quotes q JOIN sessions s ON s.session_id = q.session_id
              WHERE q.ticker_id = ? AND q.opt_right = ? AND q.delta_bucket = ?
                AND q.tenor_days = ? AND s.session_date = ?""",
        (ticker, right, delta_bucket, tenor_days, session_date),
    ).fetchone()
    if row is None:
        raise ObjectNotFoundError(
            f"no contract for {ticker} {right}{delta_bucket} {tenor_days}d on {session_date}"
        )
    return ToolResult(value=row[1], row_ids=[row[0]])


def get_vol_series(
    conn: sqlite3.Connection,
    ticker: str,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    start_date: str,
    end_date: str,
) -> ToolResult:
    if not _ticker_exists(conn, ticker):
        raise ObjectNotFoundError(f"no ticker {ticker!r}")
    rows = conn.execute(
        """SELECT q.row_id, s.session_date, q.implied_vol
               FROM iv_quotes q JOIN sessions s ON s.session_id = q.session_id
              WHERE q.ticker_id = ? AND q.opt_right = ? AND q.delta_bucket = ?
                AND q.tenor_days = ? AND s.session_date BETWEEN ? AND ?
              ORDER BY s.session_id""",
        (ticker, right, delta_bucket, tenor_days, start_date, end_date),
    ).fetchall()
    if not rows:
        raise ObjectNotFoundError(
            f"no quotes for {ticker} {right}{delta_bucket} {tenor_days}d in [{start_date}, {end_date}]"
        )
    series = [(r[1], r[2]) for r in rows]
    return ToolResult(value=series, row_ids=[r[0] for r in rows])


def median_iv_change(
    conn: sqlite3.Connection,
    ticker: str,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    start_date: str,
    end_date: str,
) -> ToolResult:
    """Did the median IV rise over the window? Splits the ordered session
    window in half; compares median of the first half to median of the
    second half."""
    series_result = get_vol_series(conn, ticker, right, delta_bucket, tenor_days, start_date, end_date)
    series = series_result.value
    row_ids = series_result.row_ids
    n = len(series)
    if n < 4:
        raise ObjectNotFoundError("window too short to split for a change measurement (need >= 4 sessions)")
    half = n // 2
    first_vals = [v for _, v in series[:half]]
    last_vals = [v for _, v in series[n - half:]]
    median_first = statistics.median(first_vals)
    median_last = statistics.median(last_vals)
    change = median_last - median_first
    return ToolResult(
        value={"median_first": median_first, "median_last": median_last, "change": change, "n_sessions": n},
        row_ids=row_ids,
    )


def term_structure_slope(
    conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int, session_date: str
) -> ToolResult:
    r30 = get_contract(conn, ticker, right, delta_bucket, 30, session_date)
    r60 = get_contract(conn, ticker, right, delta_bucket, 60, session_date)
    slope = r60.value - r30.value
    return ToolResult(
        value={"iv_30": r30.value, "iv_60": r60.value, "slope": slope},
        row_ids=r30.row_ids + r60.row_ids,
    )


def skew_snapshot(conn: sqlite3.Connection, ticker: str, tenor_days: int, session_date: str) -> ToolResult:
    atm = get_contract(conn, ticker, "P", 50, tenor_days, session_date)
    otm_put = get_contract(conn, ticker, "P", 10, tenor_days, session_date)
    skew = otm_put.value - atm.value
    return ToolResult(
        value={"iv_atm": atm.value, "iv_10d_put": otm_put.value, "skew": skew},
        row_ids=atm.row_ids + otm_put.row_ids,
    )


def percentile_rank_iv(
    conn: sqlite3.Connection,
    ticker: str,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    session_date: str,
    lookback_sessions: int,
) -> ToolResult:
    rows = conn.execute(
        """SELECT q.row_id, s.session_date, q.implied_vol
               FROM iv_quotes q JOIN sessions s ON s.session_id = q.session_id
              WHERE q.ticker_id = ? AND q.opt_right = ? AND q.delta_bucket = ?
                AND q.tenor_days = ? AND s.session_date <= ?
              ORDER BY s.session_id DESC LIMIT ?""",
        (ticker, right, delta_bucket, tenor_days, session_date, lookback_sessions),
    ).fetchall()
    if not rows:
        raise ObjectNotFoundError(f"no quotes for {ticker} on or before {session_date}")
    if rows[0][1] != session_date:
        raise ObjectNotFoundError(f"no quote exactly on {session_date} for {ticker}")
    current = rows[0][2]
    window_vals = sorted(r[2] for r in rows)
    n_leq = sum(1 for v in window_vals if v <= current)
    pct = 100.0 * n_leq / len(window_vals)
    return ToolResult(
        value={"current_iv": current, "percentile": pct, "window_size": len(rows)},
        row_ids=[r[0] for r in rows],
    )


def event_window_stats(
    conn: sqlite3.Connection,
    ticker: str,
    event_index: int,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    window: int,
) -> ToolResult:
    events = conn.execute(
        "SELECT event_id, session_id FROM events WHERE ticker_id = ? ORDER BY session_id",
        (ticker,),
    ).fetchall()
    if event_index < 0 or event_index >= len(events):
        raise ObjectNotFoundError(f"no event index {event_index} for {ticker}")
    event_id, event_session_id = events[event_index]

    before_rows = conn.execute(
        """SELECT row_id, implied_vol FROM iv_quotes
              WHERE ticker_id = ? AND opt_right = ? AND delta_bucket = ? AND tenor_days = ?
                AND session_id >= ? AND session_id < ?
              ORDER BY session_id""",
        (ticker, right, delta_bucket, tenor_days, event_session_id - window, event_session_id),
    ).fetchall()
    after_rows = conn.execute(
        """SELECT row_id, implied_vol FROM iv_quotes
              WHERE ticker_id = ? AND opt_right = ? AND delta_bucket = ? AND tenor_days = ?
                AND session_id > ? AND session_id <= ?
              ORDER BY session_id""",
        (ticker, right, delta_bucket, tenor_days, event_session_id, event_session_id + window),
    ).fetchall()
    if not before_rows or not after_rows:
        raise ObjectNotFoundError("insufficient sessions around event for this window")

    mean_before = statistics.mean(r[1] for r in before_rows)
    mean_after = statistics.mean(r[1] for r in after_rows)
    return ToolResult(
        value={"mean_before": mean_before, "mean_after": mean_after, "event_id": event_id},
        row_ids=[r[0] for r in before_rows] + [r[0] for r in after_rows],
    )


def compare_cohorts(
    conn: sqlite3.Connection,
    cohort_a: str,
    cohort_b: str,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    start_date: str,
    end_date: str,
) -> ToolResult:
    """Returns, for each cohort, one median-IV value per ticker over the
    window (the sample unit is a ticker, not a ticker-day), plus every
    iv_quotes row id that went into those medians. No test statistic is
    computed here; that is stats.py's job, kept separate from data
    retrieval."""
    members_a = cohort_membership(conn, cohort_a).value
    members_b = cohort_membership(conn, cohort_b).value

    def per_ticker_medians(tickers: list[str]) -> tuple[list[float], list[int]]:
        meds, row_ids = [], []
        for t in tickers:
            series = get_vol_series(conn, t, right, delta_bucket, tenor_days, start_date, end_date)
            vals = [v for _, v in series.value]
            meds.append(statistics.median(vals))
            row_ids.extend(series.row_ids)
        return meds, row_ids

    meds_a, rows_a = per_ticker_medians(members_a)
    meds_b, rows_b = per_ticker_medians(members_b)
    return ToolResult(
        value={"medians_a": meds_a, "medians_b": meds_b, "tickers_a": members_a, "tickers_b": members_b},
        row_ids=rows_a + rows_b,
    )


def correlation_between_tickers(
    conn: sqlite3.Connection,
    ticker_a: str,
    ticker_b: str,
    right: str,
    delta_bucket: int,
    tenor_days: int,
    start_date: str,
    end_date: str,
) -> ToolResult:
    """Returns the two tickers' daily IV series aligned by session_id over
    the window. Only sessions present for both tickers are kept."""
    rows = conn.execute(
        """SELECT qa.session_id, qa.row_id, qa.implied_vol, qb.row_id, qb.implied_vol
               FROM iv_quotes qa
               JOIN iv_quotes qb
                 ON qb.session_id = qa.session_id AND qb.opt_right = qa.opt_right
                AND qb.delta_bucket = qa.delta_bucket AND qb.tenor_days = qa.tenor_days
               JOIN sessions s ON s.session_id = qa.session_id
              WHERE qa.ticker_id = ? AND qb.ticker_id = ?
                AND qa.opt_right = ? AND qa.delta_bucket = ? AND qa.tenor_days = ?
                AND s.session_date BETWEEN ? AND ?
              ORDER BY qa.session_id""",
        (ticker_a, ticker_b, right, delta_bucket, tenor_days, start_date, end_date),
    ).fetchall()
    if len(rows) < 3:
        raise ObjectNotFoundError("insufficient paired sessions for a correlation")
    series_a = [r[2] for r in rows]
    series_b = [r[4] for r in rows]
    row_ids = [r[1] for r in rows] + [r[3] for r in rows]
    return ToolResult(value={"series_a": series_a, "series_b": series_b, "n": len(rows)}, row_ids=row_ids)


TOOL_CATALOG = {
    "list_sessions": list_sessions,
    "cohort_membership": cohort_membership,
    "dividend_decile_lookup": dividend_decile_lookup,
    "get_contract": get_contract,
    "get_vol_series": get_vol_series,
    "median_iv_change": median_iv_change,
    "term_structure_slope": term_structure_slope,
    "skew_snapshot": skew_snapshot,
    "percentile_rank_iv": percentile_rank_iv,
    "event_window_stats": event_window_stats,
    "compare_cohorts": compare_cohorts,
    "correlation_between_tickers": correlation_between_tickers,
}
