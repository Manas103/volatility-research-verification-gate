"""Generates the two deliberately-broken finding families the gate battery
measures against: fabricated numbers (never came from any real query) and
silent query errors (the query ran, raised nothing, and returned the wrong
thing: wrong date range, off-by-one window, wrong join, stale/cached rows).
Both families must be caught by gate.run_gate via the independent oracle.
"""
from __future__ import annotations

import datetime
import random
import sqlite3
import statistics

from . import finding as F
from . import stats, tools
from .ledger import TrialLedger


def _shift(date_str: str, days: int) -> str:
    d = datetime.date.fromisoformat(date_str)
    return (d + datetime.timedelta(days=days)).isoformat()


# ---------------------------------------------------------------- fabricated

def fabricate_median_change(conn, ticker, right, delta_bucket, tenor_days, start_date, end_date,
                             rng: random.Random) -> F.Finding:
    real = F.build_median_change(conn, ticker, right, delta_bucket, tenor_days, start_date, end_date)
    real_change = next(c.value for c in real.claims if c.label == "change")
    bogus = real_change + rng.choice([1, -1]) * rng.uniform(0.05, 0.2)
    old_sub = f"(change {real_change:+.4f})."
    new_sub = f"(change {bogus:+.4f})."
    text = real.text.replace(old_sub, new_sub, 1)
    claims = [c for c in real.claims if c.label != "change"]
    claims.append(F.NumericClaim("change", bogus, "window_change",
                                  dict(ticker=ticker, right=right, delta_bucket=delta_bucket,
                                       tenor_days=tenor_days, start_date=start_date, end_date=end_date)))
    return F.Finding(question="median_change", text=text, claims=claims, row_ids=real.row_ids)


def fabricate_cohort_raw_p(conn, ledger, cohort_a, cohort_b, right, delta_bucket, tenor_days,
                            start_date, end_date, rng: random.Random) -> F.Finding:
    real = F.build_cohort_compare(conn, ledger, cohort_a, cohort_b, right, delta_bucket, tenor_days,
                                   start_date, end_date)
    real_p = next(c.value for c in real.claims if c.label == "raw_p")
    bogus = rng.uniform(0.0001, 0.001) if real_p > 0.01 else rng.uniform(0.2, 0.4)
    old_sub = f"raw p={real_p:.6f}"
    new_sub = f"raw p={bogus:.6f}"
    text = real.text.replace(old_sub, new_sub, 1)
    claims = [c for c in real.claims if c.label != "raw_p"]
    claims.append(F.NumericClaim("raw_p", bogus, "two_sample_pvalue",
                                  dict(cohort_a=cohort_a, cohort_b=cohort_b, right=right,
                                       delta_bucket=delta_bucket, tenor_days=tenor_days,
                                       start_date=start_date, end_date=end_date)))
    return F.Finding(question="cohort_compare", text=text, claims=claims, row_ids=real.row_ids)


def fabricate_correlation_r(conn, ledger, ticker_a, ticker_b, right, delta_bucket, tenor_days,
                             start_date, end_date, rng: random.Random) -> F.Finding:
    real = F.build_correlation(conn, ledger, ticker_a, ticker_b, right, delta_bucket, tenor_days,
                                start_date, end_date)
    real_r = next(c.value for c in real.claims if c.label == "correlation_r")
    bogus = max(-0.99, min(0.99, real_r + rng.choice([1, -1]) * rng.uniform(0.3, 0.6)))
    old_sub = f"r={real_r:.4f};"
    new_sub = f"r={bogus:.4f};"
    text = real.text.replace(old_sub, new_sub, 1)
    claims = [c for c in real.claims if c.label != "correlation_r"]
    claims.append(F.NumericClaim("correlation_r", bogus, "correlation_r",
                                  dict(ticker_a=ticker_a, ticker_b=ticker_b, right=right,
                                       delta_bucket=delta_bucket, tenor_days=tenor_days,
                                       start_date=start_date, end_date=end_date)))
    return F.Finding(question="correlation", text=text, claims=claims, row_ids=real.row_ids)


# ------------------------------------------------------------ silent errors

def silent_error_date_shift(conn, ticker, right, delta_bucket, tenor_days, start_date, end_date,
                             shift_days: int = 7) -> F.Finding:
    """The query silently runs over [start+shift, end+shift] instead of the
    stated [start, end]; no exception, just the wrong window."""
    actual_start, actual_end = _shift(start_date, shift_days), _shift(end_date, shift_days)
    r = tools.median_iv_change(conn, ticker, right, delta_bucket, tenor_days, actual_start, actual_end)
    v = r.value
    text = (f"Median {delta_bucket}-delta {right} implied vol for {ticker} moved from "
            f"{v['median_first']:.4f} to {v['median_last']:.4f} (change {v['change']:+.4f}) over the "
            f"{v['n_sessions']} sessions from {start_date} to {end_date}.")
    base_params = dict(ticker=ticker, right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                        start_date=start_date, end_date=end_date)
    claims = [
        F.NumericClaim("median_first", v["median_first"], "window_median", dict(half="first", **base_params)),
        F.NumericClaim("median_last", v["median_last"], "window_median", dict(half="last", **base_params)),
        F.NumericClaim("change", v["change"], "window_change", base_params),
    ]
    return F.Finding(question="median_change", text=text, claims=claims, row_ids=r.row_ids)


def silent_error_off_by_one_window(conn, ticker, event_index, right, delta_bucket, tenor_days,
                                    window: int) -> F.Finding:
    """The "after" window silently includes the event session itself and
    drops the window's last session, an off-by-one in the slice
    boundaries; no exception, just one session wrong on each side."""
    events = conn.execute(
        "SELECT event_id, session_id FROM events WHERE ticker_id = ? ORDER BY session_id", (ticker,)
    ).fetchall()
    event_id, event_session_id = events[event_index]
    before_rows = conn.execute(
        "SELECT row_id, implied_vol FROM iv_quotes WHERE ticker_id=? AND opt_right=? AND delta_bucket=? "
        "AND tenor_days=? AND session_id >= ? AND session_id < ?",
        (ticker, right, delta_bucket, tenor_days, event_session_id - window, event_session_id),
    ).fetchall()
    after_rows = conn.execute(
        "SELECT row_id, implied_vol FROM iv_quotes WHERE ticker_id=? AND opt_right=? AND delta_bucket=? "
        "AND tenor_days=? AND session_id >= ? AND session_id < ?",
        (ticker, right, delta_bucket, tenor_days, event_session_id, event_session_id + window - 1),
    ).fetchall()
    mean_before = statistics.mean(r[1] for r in before_rows)
    mean_after = statistics.mean(r[1] for r in after_rows)
    diff = mean_after - mean_before
    base_params = dict(ticker=ticker, event_index=event_index, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, window=window)
    text = (f"Around {ticker}'s event #{event_index}, mean {delta_bucket}-delta {right} IV moved from "
            f"{mean_before:.4f} ({window} sessions before) to {mean_after:.4f} ({window} sessions after), "
            f"a change of {diff:+.4f}.")
    claims = [
        F.NumericClaim("mean_before", mean_before, "event_window_mean", dict(side="before", **base_params)),
        F.NumericClaim("mean_after", mean_after, "event_window_mean", dict(side="after", **base_params)),
        F.NumericClaim("diff", diff, "event_window_diff", base_params),
    ]
    return F.Finding(question="event_window", text=text, claims=claims,
                      row_ids=[r[0] for r in before_rows] + [r[0] for r in after_rows])


def silent_error_wrong_join(conn, ledger, cohort_a, cohort_b, right, delta_bucket, tenor_days,
                             start_date, end_date) -> F.Finding:
    """The membership join silently swaps one cohort-B ticker into cohort
    A's sample; no exception, the query just mixes cohorts."""
    members_a = tools.cohort_membership(conn, cohort_a).value
    members_b = tools.cohort_membership(conn, cohort_b).value
    buggy_a = members_a[:-1] + [members_b[0]]

    def medians(tickers):
        out, rids = [], []
        for t in tickers:
            res = tools.get_vol_series(conn, t, right, delta_bucket, tenor_days, start_date, end_date)
            out.append(statistics.median(v for _, v in res.value))
            rids.extend(res.row_ids)
        return out, rids

    meds_a, rows_a = medians(buggy_a)
    meds_b, rows_b = medians(members_b)
    mean_a, mean_b = statistics.mean(meds_a), statistics.mean(meds_b)
    _, raw_p = stats.two_sample_test(meds_a, meds_b)
    q_text = f"compare_cohorts(silent-join-bug) {cohort_a} vs {cohort_b} {right}{delta_bucket} {tenor_days}d"
    entry = ledger.record_and_adjust(q_text, raw_p)
    text = (f"Cohort {cohort_a} mean-of-median {delta_bucket}-delta {right} IV was {mean_a:.4f} versus "
            f"cohort {cohort_b}'s {mean_b:.4f} over {start_date}..{end_date}; raw p={raw_p:.6f}, "
            f"Holm-adjusted p={entry.adjusted_p:.6f} across {entry.ledger_size_at_report} hypotheses.")
    base_params = dict(cohort_a=cohort_a, cohort_b=cohort_b, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, start_date=start_date, end_date=end_date)
    claims = [
        F.NumericClaim("mean_a", mean_a, "cohort_mean_of_medians",
                        dict(cohort=cohort_a, right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                             start_date=start_date, end_date=end_date)),
        F.NumericClaim("mean_b", mean_b, "cohort_mean_of_medians",
                        dict(cohort=cohort_b, right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                             start_date=start_date, end_date=end_date)),
        F.NumericClaim("raw_p", raw_p, "two_sample_pvalue", base_params),
        F.NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return F.Finding(question="cohort_compare", text=text, claims=claims, row_ids=rows_a + rows_b)


def silent_error_stale_rows(conn, ledger, ticker_a, ticker_b, stale_ticker_b, right, delta_bucket,
                             tenor_days, start_date, end_date) -> F.Finding:
    """Reports a correlation between ticker_a and ticker_b, but the series
    actually pulled for "ticker_b" is a stale cached series from a
    different ticker; no exception, just the wrong rows."""
    r = tools.correlation_between_tickers(conn, ticker_a, stale_ticker_b, right, delta_bucket, tenor_days,
                                           start_date, end_date)
    v = r.value
    corr_r, raw_p = stats.pearson_correlation(v["series_a"], v["series_b"])
    q_text = f"correlation(stale-cache-bug) {ticker_a} vs {ticker_b} {right}{delta_bucket} {tenor_days}d"
    entry = ledger.record_and_adjust(q_text, raw_p)
    text = (f"{ticker_a} and {ticker_b} {delta_bucket}-delta {right} IV correlation over {v['n']} paired "
            f"sessions ({start_date}..{end_date}) is r={corr_r:.4f}; raw p={raw_p:.6f}, "
            f"Holm-adjusted p={entry.adjusted_p:.6f} across {entry.ledger_size_at_report} hypotheses.")
    base_params = dict(ticker_a=ticker_a, ticker_b=ticker_b, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, start_date=start_date, end_date=end_date)
    claims = [
        F.NumericClaim("correlation_r", corr_r, "correlation_r", base_params),
        F.NumericClaim("raw_p", raw_p, "correlation_pvalue", base_params),
        F.NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return F.Finding(question="correlation", text=text, claims=claims, row_ids=r.row_ids)


def build_broken_set(conn: sqlite3.Connection, ledger: TrialLedger, correct_pool: list[dict],
                      n_fabricated: int = 30, n_silent: int = 30, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []

    med_specs = [c for c in correct_pool if c["shape"] == "median_change"]
    cohort_specs = [c for c in correct_pool if c["shape"] == "cohort_compare"]
    corr_specs = [c for c in correct_pool if c["shape"] == "correlation"]
    event_specs = [c for c in correct_pool if c["shape"] == "event_window"]

    i = 0
    while len(out) < n_fabricated and i < 10000:
        kind = i % 3
        try:
            if kind == 0 and med_specs:
                s = med_specs[i % len(med_specs)]["params"]
                f = fabricate_median_change(conn, rng=rng, **s)
            elif kind == 1 and cohort_specs:
                s = cohort_specs[i % len(cohort_specs)]["params"]
                f = fabricate_cohort_raw_p(conn, ledger, rng=rng, **s)
            elif corr_specs:
                s = corr_specs[i % len(corr_specs)]["params"]
                f = fabricate_correlation_r(conn, ledger, rng=rng, **s)
            else:
                i += 1
                continue
            out.append({"label": "fabricated", "finding": f})
        except Exception:
            pass
        i += 1

    j = 0
    while len(out) < n_fabricated + n_silent and j < 10000:
        kind = j % 4
        try:
            if kind == 0 and med_specs:
                s = med_specs[j % len(med_specs)]["params"]
                f = silent_error_date_shift(conn, **s)
            elif kind == 1 and event_specs:
                s = event_specs[j % len(event_specs)]["params"]
                f = silent_error_off_by_one_window(conn, **s)
            elif kind == 2 and cohort_specs:
                s = cohort_specs[j % len(cohort_specs)]["params"]
                f = silent_error_wrong_join(conn, ledger, **s)
            elif corr_specs:
                s = dict(corr_specs[j % len(corr_specs)]["params"])
                all_tickers = [r[0] for r in conn.execute("SELECT ticker_id FROM tickers").fetchall()]
                stale = rng.choice([t for t in all_tickers if t not in (s["ticker_a"], s["ticker_b"])])
                f = silent_error_stale_rows(conn, ledger, s["ticker_a"], s["ticker_b"], stale, s["right"],
                                            s["delta_bucket"], s["tenor_days"], s["start_date"], s["end_date"])
            else:
                j += 1
                continue
            out.append({"label": "silent_error", "finding": f})
        except Exception:
            pass
        j += 1

    return out
