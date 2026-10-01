"""The trial ledger: a persistent, append-only record of every
hypothesis-bearing question (one that produces a p-value) ever answered,
and the Holm-Bonferroni-adjusted p-value actually reported for each, using
however many hypotheses the ledger held at the moment that question was
answered.

Persistent, not per-session: the ledger lives in a SQLite table
(trial_ledger) in the same data file as the option-chain store, survives
process restarts, and is never truncated except by an explicit reset. This
is deliberate: a ledger that forgot earlier hypotheses the moment the
process exited would let a researcher dodge the correction simply by
restarting, defeating the entire point of a cumulative, not batch,
adjustment.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .holm import holm_bonferroni

SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_ledger (
    ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_text TEXT NOT NULL,
    raw_p REAL NOT NULL,
    adjusted_p REAL NOT NULL,
    ledger_size_at_report INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class LedgerEntry:
    ledger_id: int
    question_text: str
    raw_p: float
    adjusted_p: float
    ledger_size_at_report: int


class TrialLedger:
    """Backed by a real sqlite3.Connection. Construct with the same
    connection the option-chain store uses, or a dedicated one; either way
    it persists on disk, not in memory."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def size(self) -> int:
        (n,) = self.conn.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()
        return n

    def record_and_adjust(self, question_text: str, raw_p: float) -> LedgerEntry:
        """Appends one hypothesis to the ledger, then recomputes the
        Holm-Bonferroni adjustment across every raw p-value the ledger now
        holds (the one just appended included), and returns the adjusted
        value for THIS entry. This is the cumulative, not batch, property:
        the n used in the correction grows every time a new hypothesis is
        proposed, so the same raw p-value reported late in a long research
        session is adjusted harder than if it had been the first question
        ever asked.
        """
        cur = self.conn.execute(
            "INSERT INTO trial_ledger (question_text, raw_p, adjusted_p, ledger_size_at_report) "
            "VALUES (?, ?, 0.0, 0)",
            (question_text, raw_p),
        )
        new_id = cur.lastrowid
        self.conn.commit()

        rows = self.conn.execute(
            "SELECT ledger_id, raw_p FROM trial_ledger ORDER BY ledger_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        raw_ps = [r[1] for r in rows]
        adjusted = holm_bonferroni(raw_ps)
        pos = ids.index(new_id)
        adjusted_p_for_new = adjusted[pos]
        n = len(rows)

        self.conn.execute(
            "UPDATE trial_ledger SET adjusted_p = ?, ledger_size_at_report = ? WHERE ledger_id = ?",
            (adjusted_p_for_new, n, new_id),
        )
        self.conn.commit()

        return LedgerEntry(
            ledger_id=new_id,
            question_text=question_text,
            raw_p=raw_p,
            adjusted_p=adjusted_p_for_new,
            ledger_size_at_report=n,
        )

    def get(self, ledger_id: int) -> LedgerEntry:
        row = self.conn.execute(
            "SELECT ledger_id, question_text, raw_p, adjusted_p, ledger_size_at_report "
            "FROM trial_ledger WHERE ledger_id = ?",
            (ledger_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no ledger entry {ledger_id}")
        return LedgerEntry(*row)

    def all_raw_pvalues_up_to(self, ledger_id: int) -> list[float]:
        rows = self.conn.execute(
            "SELECT raw_p FROM trial_ledger WHERE ledger_id <= ? ORDER BY ledger_id", (ledger_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def reset(self) -> None:
        self.conn.execute("DELETE FROM trial_ledger")
        self.conn.commit()
