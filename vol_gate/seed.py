"""Seeded synthetic option-chain / implied-vol data generator.

No real market data. 16 tickers (8 cohort A, "high vol regime", 8 cohort B,
"low vol regime"), 120 business-day sessions, 2 tenors (30d, 60d), 5 delta
buckets (10, 25, 50, 75, 90) per right (P, C), one earnings-style event per
ticker every ~30 sessions. The implied-vol process is a per-ticker AR(1)
level with a delta smile, a small term-structure tilt, and an event bump,
all driven by one seeded numpy Generator so the same seed always produces
the same store.
"""
from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import db

DEFAULT_SEED = 20261001
N_SESSIONS = 120
TENORS = (30, 60)
DELTA_BUCKETS = (10, 25, 50, 75, 90)
RIGHTS = ("P", "C")
SECTORS = ("TECH", "ENERGY", "CONSUMER", "INDUSTRIALS")

COHORT_A_TICKERS = [f"SYN-A{i:02d}" for i in range(1, 9)]  # high vol regime
COHORT_B_TICKERS = [f"SYN-B{i:02d}" for i in range(1, 9)]  # low vol regime
ALL_TICKERS = COHORT_A_TICKERS + COHORT_B_TICKERS

START_DATE = datetime.date(2025, 1, 2)


@dataclass
class SeedStats:
    n_tickers: int
    n_sessions: int
    n_events: int
    n_iv_rows: int


def _business_days(start: datetime.date, n: int) -> list[datetime.date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def _smile_factor(delta_bucket: int, right: str) -> float:
    """Puts skew up as delta moves away from 50 (OTM puts bid up); calls
    skew up more mildly. Pure, deterministic function of (delta, right)."""
    dist = abs(delta_bucket - 50) / 50.0
    if right == "P":
        return 1.0 + 0.35 * dist
    return 1.0 + 0.15 * dist


def _tenor_factor(tenor_days: int) -> float:
    return 1.0 if tenor_days == 30 else 1.04


def seed_database(conn: sqlite3.Connection, seed: int = DEFAULT_SEED) -> SeedStats:
    db.init_schema(conn)
    rng = np.random.default_rng(seed)

    cur = conn.cursor()
    cur.execute("DELETE FROM iv_quotes")
    cur.execute("DELETE FROM events")
    cur.execute("DELETE FROM sessions")
    cur.execute("DELETE FROM tickers")

    session_dates = _business_days(START_DATE, N_SESSIONS)
    for i, d in enumerate(session_dates):
        cur.execute(
            "INSERT INTO sessions (session_id, session_date) VALUES (?, ?)",
            (i, d.isoformat()),
        )

    ticker_rows = []
    for idx, tkr in enumerate(ALL_TICKERS):
        cohort = "A" if tkr in COHORT_A_TICKERS else "B"
        decile = int(rng.integers(1, 11))
        sector = SECTORS[idx % len(SECTORS)]
        ticker_rows.append((tkr, cohort, decile, sector))
    cur.executemany(
        "INSERT INTO tickers (ticker_id, cohort, dividend_decile, sector) VALUES (?, ?, ?, ?)",
        ticker_rows,
    )

    base_vol = {}
    for tkr in COHORT_A_TICKERS:
        base_vol[tkr] = rng.normal(0.38, 0.03)
    for tkr in COHORT_B_TICKERS:
        base_vol[tkr] = rng.normal(0.22, 0.02)

    event_id = 0
    event_rows = []
    iv_rows = []
    row_id = 0

    for tkr in ALL_TICKERS:
        # AR(1) level path, 120 sessions
        level = np.zeros(N_SESSIONS)
        level[0] = base_vol[tkr]
        sigma = 0.012
        for s in range(1, N_SESSIONS):
            level[s] = base_vol[tkr] + 0.9 * (level[s - 1] - base_vol[tkr]) + rng.normal(0, sigma)
        level = np.clip(level, 0.05, None)

        # one event every ~30 sessions, offset per ticker so events don't align
        offset = int(rng.integers(5, 25))
        event_sessions = [s for s in range(offset, N_SESSIONS, 30)]
        for s in event_sessions:
            event_rows.append((event_id, tkr, s, "EARNINGS"))
            event_id += 1
        event_bump = np.zeros(N_SESSIONS)
        for s in event_sessions:
            for w in range(-2, 3):
                ss = s + w
                if 0 <= ss < N_SESSIONS:
                    event_bump[ss] += 0.04 * (1 - abs(w) / 3.0)

        for s in range(N_SESSIONS):
            for tenor in TENORS:
                for delta in DELTA_BUCKETS:
                    for right in RIGHTS:
                        iv = (
                            level[s] * _smile_factor(delta, right) * _tenor_factor(tenor)
                            + event_bump[s]
                            + rng.normal(0, 0.004)
                        )
                        iv = float(max(0.02, iv))
                        iv_rows.append((row_id, tkr, s, tenor, delta, right, round(iv, 6)))
                        row_id += 1

    cur.executemany(
        "INSERT INTO events (event_id, ticker_id, session_id, event_type) VALUES (?, ?, ?, ?)",
        event_rows,
    )
    cur.executemany(
        "INSERT INTO iv_quotes (row_id, ticker_id, session_id, tenor_days, delta_bucket, opt_right, implied_vol) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        iv_rows,
    )
    conn.commit()

    return SeedStats(
        n_tickers=len(ticker_rows),
        n_sessions=len(session_dates),
        n_events=len(event_rows),
        n_iv_rows=len(iv_rows),
    )


if __name__ == "__main__":
    with db.session() as conn:
        stats = seed_database(conn)
        print(stats)
