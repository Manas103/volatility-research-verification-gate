"""Builds a Finding: a short written answer citing specific numbers, each
number tagged with enough information (a `kind` and `params`) for
gate/oracle.py to recompute it from scratch through a completely separate
code path. This module is "the executor": it is allowed to call tools.py,
stats.py, and ledger.py freely. gate/oracle.py must never import from here.
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field

from . import stats, tools
from .ledger import TrialLedger


@dataclass
class NumericClaim:
    label: str
    value: float
    kind: str
    params: dict
    tol: float = 1e-6


@dataclass
class Finding:
    question: str
    text: str
    claims: list[NumericClaim] = field(default_factory=list)
    row_ids: list[int] = field(default_factory=list)


def build_median_change(conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int,
                         tenor_days: int, start_date: str, end_date: str) -> Finding:
    r = tools.median_iv_change(conn, ticker, right, delta_bucket, tenor_days, start_date, end_date)
    v = r.value
    direction = "rose" if v["change"] > 0 else ("fell" if v["change"] < 0 else "was unchanged")
    base_params = dict(ticker=ticker, right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                        start_date=start_date, end_date=end_date)
    text = (
        f"Median {delta_bucket}-delta {right} implied vol for {ticker} {direction} over the "
        f"{v['n_sessions']} sessions from {start_date} to {end_date}: "
        f"{v['median_first']:.4f} to {v['median_last']:.4f} (change {v['change']:+.4f})."
    )
    claims = [
        NumericClaim("median_first", v["median_first"], "window_median", dict(half="first", **base_params)),
        NumericClaim("median_last", v["median_last"], "window_median", dict(half="last", **base_params)),
        NumericClaim("change", v["change"], "window_change", base_params),
    ]
    return Finding(question="median_change", text=text, claims=claims, row_ids=r.row_ids)


def build_cohort_compare(conn: sqlite3.Connection, ledger: TrialLedger, cohort_a: str, cohort_b: str,
                          right: str, delta_bucket: int, tenor_days: int, start_date: str,
                          end_date: str) -> Finding:
    r = tools.compare_cohorts(conn, cohort_a, cohort_b, right, delta_bucket, tenor_days, start_date, end_date)
    v = r.value
    mean_a = statistics.mean(v["medians_a"])
    mean_b = statistics.mean(v["medians_b"])
    _, raw_p = stats.two_sample_test(v["medians_a"], v["medians_b"])
    q_text = (f"compare_cohorts {cohort_a} vs {cohort_b} {right}{delta_bucket} {tenor_days}d "
              f"{start_date}..{end_date}")
    entry = ledger.record_and_adjust(q_text, raw_p)
    base_params = dict(cohort_a=cohort_a, cohort_b=cohort_b, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, start_date=start_date, end_date=end_date)
    sig = "a significant" if entry.adjusted_p < 0.05 else "no significant"
    text = (
        f"Cohort {cohort_a} mean-of-median {delta_bucket}-delta {right} IV was {mean_a:.4f} versus "
        f"cohort {cohort_b}'s {mean_b:.4f} over {start_date}..{end_date}; a two-sample test gives "
        f"raw p={raw_p:.6f}, Holm-adjusted p={entry.adjusted_p:.6f} across the {entry.ledger_size_at_report} "
        f"hypotheses proposed so far, {sig} difference at alpha=0.05."
    )
    claims = [
        NumericClaim("mean_a", mean_a, "cohort_mean_of_medians", dict(cohort=cohort_a, **_wo(base_params, "cohort_a", "cohort_b"))),
        NumericClaim("mean_b", mean_b, "cohort_mean_of_medians", dict(cohort=cohort_b, **_wo(base_params, "cohort_a", "cohort_b"))),
        NumericClaim("raw_p", raw_p, "two_sample_pvalue", base_params),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="cohort_compare", text=text, claims=claims, row_ids=r.row_ids)


def build_correlation(conn: sqlite3.Connection, ledger: TrialLedger, ticker_a: str, ticker_b: str,
                       right: str, delta_bucket: int, tenor_days: int, start_date: str,
                       end_date: str) -> Finding:
    r = tools.correlation_between_tickers(conn, ticker_a, ticker_b, right, delta_bucket, tenor_days,
                                           start_date, end_date)
    v = r.value
    corr_r, raw_p = stats.pearson_correlation(v["series_a"], v["series_b"])
    q_text = (f"correlation {ticker_a} vs {ticker_b} {right}{delta_bucket} {tenor_days}d "
              f"{start_date}..{end_date}")
    entry = ledger.record_and_adjust(q_text, raw_p)
    base_params = dict(ticker_a=ticker_a, ticker_b=ticker_b, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, start_date=start_date, end_date=end_date)
    text = (
        f"{ticker_a} and {ticker_b} {delta_bucket}-delta {right} IV correlation over "
        f"{v['n']} paired sessions ({start_date}..{end_date}) is r={corr_r:.4f}; raw p={raw_p:.6f}, "
        f"Holm-adjusted p={entry.adjusted_p:.6f} across {entry.ledger_size_at_report} hypotheses."
    )
    claims = [
        NumericClaim("correlation_r", corr_r, "correlation_r", base_params),
        NumericClaim("raw_p", raw_p, "correlation_pvalue", base_params),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="correlation", text=text, claims=claims, row_ids=r.row_ids)


def build_event_window(conn: sqlite3.Connection, ticker: str, event_index: int, right: str,
                        delta_bucket: int, tenor_days: int, window: int) -> Finding:
    r = tools.event_window_stats(conn, ticker, event_index, right, delta_bucket, tenor_days, window)
    v = r.value
    diff = v["mean_after"] - v["mean_before"]
    base_params = dict(ticker=ticker, event_index=event_index, right=right, delta_bucket=delta_bucket,
                        tenor_days=tenor_days, window=window)
    text = (
        f"Around {ticker}'s event #{event_index}, mean {delta_bucket}-delta {right} IV moved from "
        f"{v['mean_before']:.4f} ({window} sessions before) to {v['mean_after']:.4f} "
        f"({window} sessions after), a change of {diff:+.4f}."
    )
    claims = [
        NumericClaim("mean_before", v["mean_before"], "event_window_mean", dict(side="before", **base_params)),
        NumericClaim("mean_after", v["mean_after"], "event_window_mean", dict(side="after", **base_params)),
        NumericClaim("diff", diff, "event_window_diff", base_params),
    ]
    return Finding(question="event_window", text=text, claims=claims, row_ids=r.row_ids)


def build_term_structure(conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int,
                          session_date: str) -> Finding:
    r = tools.term_structure_slope(conn, ticker, right, delta_bucket, session_date)
    v = r.value
    base_params = dict(ticker=ticker, right=right, delta_bucket=delta_bucket, session_date=session_date)
    text = (
        f"{ticker} {delta_bucket}-delta {right} term structure on {session_date}: 30d {v['iv_30']:.4f}, "
        f"60d {v['iv_60']:.4f}, slope {v['slope']:+.4f}."
    )
    claims = [
        NumericClaim("iv_30", v["iv_30"], "iv_level", dict(tenor_days=30, **base_params)),
        NumericClaim("iv_60", v["iv_60"], "iv_level", dict(tenor_days=60, **base_params)),
        NumericClaim("slope", v["slope"], "term_slope", base_params),
    ]
    return Finding(question="term_structure", text=text, claims=claims, row_ids=r.row_ids)


def build_skew(conn: sqlite3.Connection, ticker: str, tenor_days: int, session_date: str) -> Finding:
    r = tools.skew_snapshot(conn, ticker, tenor_days, session_date)
    v = r.value
    base_params = dict(ticker=ticker, tenor_days=tenor_days, session_date=session_date)
    text = (
        f"{ticker} {tenor_days}d put skew on {session_date}: ATM {v['iv_atm']:.4f}, "
        f"10-delta put {v['iv_10d_put']:.4f}, skew {v['skew']:+.4f}."
    )
    claims = [
        NumericClaim("iv_atm", v["iv_atm"], "iv_level", dict(right="P", delta_bucket=50, **base_params)),
        NumericClaim("iv_10d_put", v["iv_10d_put"], "iv_level", dict(right="P", delta_bucket=10, **base_params)),
        NumericClaim("skew", v["skew"], "skew", base_params),
    ]
    return Finding(question="skew", text=text, claims=claims, row_ids=r.row_ids)


def build_percentile_rank(conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int,
                           tenor_days: int, session_date: str, lookback_sessions: int) -> Finding:
    r = tools.percentile_rank_iv(conn, ticker, right, delta_bucket, tenor_days, session_date, lookback_sessions)
    v = r.value
    base_params = dict(ticker=ticker, right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                        session_date=session_date, lookback_sessions=lookback_sessions)
    text = (
        f"{ticker} {delta_bucket}-delta {right} IV on {session_date} is {v['current_iv']:.4f}, the "
        f"{v['percentile']:.1f}th percentile of its trailing {v['window_size']}-session window."
    )
    claims = [
        NumericClaim("current_iv", v["current_iv"], "iv_level",
                      dict(right=right, delta_bucket=delta_bucket, tenor_days=tenor_days,
                           ticker=ticker, session_date=session_date)),
        NumericClaim("percentile", v["percentile"], "percentile_rank", base_params),
    ]
    return Finding(question="percentile_rank", text=text, claims=claims, row_ids=r.row_ids)


def build_dividend_decile(conn: sqlite3.Connection, ticker: str) -> Finding:
    r = tools.dividend_decile_lookup(conn, ticker)
    v = r.value
    text = (
        f"{ticker} is in dividend decile {v['decile']}, ranked {v['rank']} of {v['universe_size']} "
        f"names by decile."
    )
    claims = [
        NumericClaim("decile", v["decile"], "dividend_decile", dict(ticker=ticker)),
        NumericClaim("rank", v["rank"], "dividend_rank", dict(ticker=ticker)),
    ]
    return Finding(question="dividend_decile", text=text, claims=claims, row_ids=r.row_ids)


def build_contract_lookup(conn: sqlite3.Connection, ticker: str, right: str, delta_bucket: int,
                           tenor_days: int, session_date: str) -> Finding:
    r = tools.get_contract(conn, ticker, right, delta_bucket, tenor_days, session_date)
    text = f"{ticker} {delta_bucket}-delta {right} {tenor_days}d implied vol on {session_date} is {r.value:.4f}."
    claims = [
        NumericClaim("iv", r.value, "iv_level", dict(ticker=ticker, right=right, delta_bucket=delta_bucket,
                                                       tenor_days=tenor_days, session_date=session_date)),
    ]
    return Finding(question="contract_lookup", text=text, claims=claims, row_ids=r.row_ids)


def build_list_sessions_count(conn: sqlite3.Connection, ticker: str, start_date: str, end_date: str) -> Finding:
    r = tools.list_sessions(conn, ticker, start_date, end_date)
    n = len(r.value)
    text = f"{ticker} has {n} trading sessions with quotes between {start_date} and {end_date}."
    claims = [
        NumericClaim("session_count", n, "session_count", dict(ticker=ticker, start_date=start_date,
                                                                 end_date=end_date)),
    ]
    return Finding(question="list_sessions_count", text=text, claims=claims, row_ids=r.row_ids)


def _wo(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def build_vol_pair_compare(conn: sqlite3.Connection, ledger: TrialLedger, ticker_a: str,
                            ticker_b: str) -> Finding:
    """Hypothesis: tickers A and B have different realized-volatility
    levels, against the volatility store imported from
    order-book-signal-research (see vol_gate/volstore.py)."""
    a = tools.vol_bin_values(conn, ticker_a)
    b = tools.vol_bin_values(conn, ticker_b)
    mean_a, mean_b = statistics.mean(a.value), statistics.mean(b.value)
    _, raw_p = stats.two_sample_test(a.value, b.value)
    q_text = f"vol_pair_compare {ticker_a} vs {ticker_b}"
    entry = ledger.record_and_adjust(q_text, raw_p)
    base_params = dict(ticker_a=ticker_a, ticker_b=ticker_b)
    sig = "a significant" if entry.adjusted_p < 0.05 else "no significant"
    text = (
        f"{ticker_a}'s mean 10-minute realized volatility was {mean_a:.4f} versus {ticker_b}'s "
        f"{mean_b:.4f} across all measured bins; a two-sample test gives raw p={raw_p:.6f}, "
        f"Holm-adjusted p={entry.adjusted_p:.6f} across {entry.ledger_size_at_report} hypotheses, "
        f"{sig} difference at alpha=0.05."
    )
    claims = [
        NumericClaim("mean_a", mean_a, "vol_mean", dict(ticker=ticker_a)),
        NumericClaim("mean_b", mean_b, "vol_mean", dict(ticker=ticker_b)),
        NumericClaim("raw_p", raw_p, "vol_pair_pvalue", base_params),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="vol_pair_compare", text=text, claims=claims, row_ids=a.row_ids + b.row_ids)


def build_vol_pair_correlation(conn: sqlite3.Connection, ledger: TrialLedger, ticker_a: str,
                                ticker_b: str) -> Finding:
    """Hypothesis: tickers A and B's realized volatility move together,
    bin for bin (aligned by date and bin_id)."""
    r = tools.vol_paired_bin_values(conn, ticker_a, ticker_b)
    corr_r, raw_p = stats.pearson_correlation(r.value["series_a"], r.value["series_b"])
    q_text = f"vol_pair_correlation {ticker_a} vs {ticker_b}"
    entry = ledger.record_and_adjust(q_text, raw_p)
    base_params = dict(ticker_a=ticker_a, ticker_b=ticker_b)
    text = (
        f"{ticker_a} and {ticker_b} realized-volatility correlation, bin for bin, is r={corr_r:.4f}; "
        f"raw p={raw_p:.6f}, Holm-adjusted p={entry.adjusted_p:.6f} across "
        f"{entry.ledger_size_at_report} hypotheses."
    )
    claims = [
        NumericClaim("correlation_r", corr_r, "vol_pair_correlation_r", base_params),
        NumericClaim("raw_p", raw_p, "vol_pair_correlation_pvalue", base_params),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="vol_pair_correlation", text=text, claims=claims, row_ids=r.row_ids)


def build_vol_activity_correlation(conn: sqlite3.Connection, ledger: TrialLedger, ticker: str) -> Finding:
    """Hypothesis: message count and realized volatility are correlated
    within a bin (the activity-vs-vol relationship the Optiver no-brainer
    frames this project around)."""
    r = tools.vol_activity_series(conn, ticker)
    corr_r, raw_p = stats.pearson_correlation(r.value["msg_count"], r.value["bin_rv"])
    q_text = f"vol_activity_correlation {ticker}"
    entry = ledger.record_and_adjust(q_text, raw_p)
    text = (
        f"{ticker}: message-count-vs-realized-volatility correlation across its bins is r={corr_r:.4f}; "
        f"raw p={raw_p:.6f}, Holm-adjusted p={entry.adjusted_p:.6f} across "
        f"{entry.ledger_size_at_report} hypotheses."
    )
    claims = [
        NumericClaim("correlation_r", corr_r, "vol_activity_correlation_r", dict(ticker=ticker)),
        NumericClaim("raw_p", raw_p, "vol_activity_correlation_pvalue", dict(ticker=ticker)),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="vol_activity_correlation", text=text, claims=claims, row_ids=r.row_ids)


def build_vol_day_trend_correlation(conn: sqlite3.Connection, ledger: TrialLedger, ticker: str) -> Finding:
    """Hypothesis: realized volatility trends across the 16-day sample
    (day_index vs bin_rv)."""
    r = tools.vol_day_trend_series(conn, ticker)
    corr_r, raw_p = stats.pearson_correlation(r.value["day_index"], r.value["bin_rv"])
    q_text = f"vol_day_trend_correlation {ticker}"
    entry = ledger.record_and_adjust(q_text, raw_p)
    text = (
        f"{ticker}: day-index-vs-realized-volatility correlation across the sample is r={corr_r:.4f}; "
        f"raw p={raw_p:.6f}, Holm-adjusted p={entry.adjusted_p:.6f} across "
        f"{entry.ledger_size_at_report} hypotheses."
    )
    claims = [
        NumericClaim("correlation_r", corr_r, "vol_day_trend_correlation_r", dict(ticker=ticker)),
        NumericClaim("raw_p", raw_p, "vol_day_trend_correlation_pvalue", dict(ticker=ticker)),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="vol_day_trend_correlation", text=text, claims=claims, row_ids=r.row_ids)


def build_vol_regime_compare(conn: sqlite3.Connection, ledger: TrialLedger, ticker: str,
                              split_by: str) -> Finding:
    """Hypothesis: realized volatility differs between the two halves of a
    fixed, pre-registered split rule (see tools.vol_regime_split)."""
    r = tools.vol_regime_split(conn, ticker, split_by)
    mean_high = statistics.mean(r.value["group_high"])
    mean_low = statistics.mean(r.value["group_low"])
    _, raw_p = stats.two_sample_test(r.value["group_high"], r.value["group_low"])
    q_text = f"vol_regime_compare {ticker} split_by={split_by}"
    entry = ledger.record_and_adjust(q_text, raw_p)
    base_params = dict(ticker=ticker, split_by=split_by)
    sig = "a significant" if entry.adjusted_p < 0.05 else "no significant"
    text = (
        f"{ticker}: mean realized volatility in the '{split_by}'-high group was {mean_high:.4f} "
        f"versus the low group's {mean_low:.4f}; a two-sample test gives raw p={raw_p:.6f}, "
        f"Holm-adjusted p={entry.adjusted_p:.6f} across {entry.ledger_size_at_report} hypotheses, "
        f"{sig} difference at alpha=0.05."
    )
    claims = [
        NumericClaim("mean_high", mean_high, "vol_regime_mean", dict(group="high", **base_params)),
        NumericClaim("mean_low", mean_low, "vol_regime_mean", dict(group="low", **base_params)),
        NumericClaim("raw_p", raw_p, "vol_regime_pvalue", base_params),
        NumericClaim("adjusted_p", entry.adjusted_p, "ledger_adjusted_pvalue", dict(ledger_id=entry.ledger_id)),
    ]
    return Finding(question="vol_regime_compare", text=text, claims=claims, row_ids=r.row_ids)


SHAPES = {
    "median_change": build_median_change,
    "cohort_compare": build_cohort_compare,
    "correlation": build_correlation,
    "event_window": build_event_window,
    "term_structure": build_term_structure,
    "skew": build_skew,
    "percentile_rank": build_percentile_rank,
    "dividend_decile": build_dividend_decile,
    "contract_lookup": build_contract_lookup,
    "list_sessions_count": build_list_sessions_count,
    "vol_pair_compare": build_vol_pair_compare,
    "vol_pair_correlation": build_vol_pair_correlation,
    "vol_activity_correlation": build_vol_activity_correlation,
    "vol_day_trend_correlation": build_vol_day_trend_correlation,
    "vol_regime_compare": build_vol_regime_compare,
}

HYPOTHESIS_SHAPES = {
    "cohort_compare", "correlation",
    "vol_pair_compare", "vol_pair_correlation", "vol_activity_correlation",
    "vol_day_trend_correlation", "vol_regime_compare",
}


def build_finding(shape: str, conn: sqlite3.Connection, ledger: TrialLedger, params: dict) -> Finding:
    """Single dispatch point used by both LLM-client implementations, so the
    deterministic router and the real claude -p client exercise the exact
    same finding-building code (the thing under test is which shape+params
    each client extracts from the question text, not two different
    execution paths)."""
    builder = SHAPES[shape]
    if shape in HYPOTHESIS_SHAPES:
        return builder(conn, ledger, **params)
    return builder(conn, **params)
