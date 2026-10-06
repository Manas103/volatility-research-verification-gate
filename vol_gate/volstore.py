"""The volatility store this extension screens hypotheses against: the
bin-level realized-volatility table measured by the sibling repo
[order-book-signal-research](https://github.com/Manas103/order-book-signal-research)'s
Oct. 2026 extension (`obsr/volatility.py`), not a dataset generated inside
this repo. `data/volatility_bins_from_order_book_signal_research.csv` is a
direct, unedited export of that repo's `build_bin_rv`/`attach_forward_rv`
output (640 rows: 5 synthetic tickers x 16 business days x up to 8
600-second bins per session), committed here because it is the "volatility
store" the resume bullet names and this repo needs it on disk to query.

This module only loads that CSV into its own SQLite table
(`volatility_bins`) and is the sole place that knows the import format;
every hypothesis in `vol_gate/hypothesis_bank.py` reads through
`vol_gate.tools`' volatility functions, never this module directly.
"""
from __future__ import annotations

import csv
import os
import sqlite3

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "volatility_bins_from_order_book_signal_research.csv"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS volatility_bins (
    row_id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    bin_date TEXT NOT NULL,
    day_index INTEGER NOT NULL,
    bin_id INTEGER NOT NULL,
    msg_count INTEGER NOT NULL,
    bin_rv REAL NOT NULL,
    prev_rv REAL NOT NULL,
    fwd_rv REAL
);

CREATE INDEX IF NOT EXISTS idx_volbins_ticker ON volatility_bins(ticker);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def import_csv(conn: sqlite3.Connection, csv_path: str = DEFAULT_CSV_PATH) -> int:
    """Idempotent: clears and reloads the table from the CSV every call, so
    re-running `scripts/init_db.py` never silently duplicates rows."""
    init_schema(conn)
    conn.execute("DELETE FROM volatility_bins")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                r["ticker"], r["date"], int(r["day_index"]), int(r["bin_id"]),
                int(r["msg_count"]), float(r["bin_rv"]), float(r["prev_rv"]),
                None if r["fwd_rv"] == "" else float(r["fwd_rv"]),
            )
            for r in reader
        ]
    conn.executemany(
        "INSERT INTO volatility_bins "
        "(ticker, bin_date, day_index, bin_id, msg_count, bin_rv, prev_rv, fwd_rv) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def tickers(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT ticker FROM volatility_bins ORDER BY ticker").fetchall()
    return [r[0] for r in rows]
