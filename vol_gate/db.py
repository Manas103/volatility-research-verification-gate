"""SQLite schema and connection helper for the synthetic option-chain store.

Four tables, no ORM: tickers, sessions, events, iv_quotes. iv_quotes is the
only table most tools and the oracle read; its integer primary key (row_id)
is the evidence identifier every tool and every verification claim cites.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vol_store.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickers (
    ticker_id TEXT PRIMARY KEY,
    cohort TEXT NOT NULL CHECK (cohort IN ('A', 'B')),
    dividend_decile INTEGER NOT NULL CHECK (dividend_decile BETWEEN 1 AND 10),
    sector TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY,
    session_date TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    ticker_id TEXT NOT NULL REFERENCES tickers(ticker_id),
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    event_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iv_quotes (
    row_id INTEGER PRIMARY KEY,
    ticker_id TEXT NOT NULL REFERENCES tickers(ticker_id),
    session_id INTEGER NOT NULL REFERENCES sessions(session_id),
    tenor_days INTEGER NOT NULL CHECK (tenor_days IN (30, 60)),
    delta_bucket INTEGER NOT NULL CHECK (delta_bucket IN (10, 25, 50, 75, 90)),
    opt_right TEXT NOT NULL CHECK (opt_right IN ('P', 'C')),
    implied_vol REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_iv_lookup
    ON iv_quotes(ticker_id, opt_right, delta_bucket, tenor_days, session_id);
"""


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def row_count(conn: sqlite3.Connection, table: str) -> int:
    (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return n


@contextmanager
def session(db_path: str = DEFAULT_DB_PATH):
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
