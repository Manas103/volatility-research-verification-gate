"""Genuine, small live sample: real `claude -p` subprocess calls parsing
plain-English questions into (shape, params), executed through the same
finding builders and verification gate the deterministic router uses."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate import db, gate
from vol_gate.finding import build_finding
from vol_gate.ledger import TrialLedger
from vol_gate.llm_client import ClaudeCLIClient, ClaudeCLIError

QUESTIONS = [
    "What dividend decile is SYN-A01 in?",
    "What dividend decile is SYN-B03 in?",
    "What is SYN-A02's 25-delta put 30-day implied vol on 2025-01-09?",
    "What is SYN-B04's 50-delta call 30-day implied vol on 2025-02-06?",
    "How many trading sessions does SYN-A01 have between 2025-01-02 and 2025-03-27?",
    "What is SYN-A01's 10-delta put term structure slope on 2025-01-09?",
    "What is SYN-B02's 60-day put skew on 2025-02-06?",
    "Where does SYN-A03's 25-delta put 30-day implied vol on 2025-03-14 rank over the trailing 30 sessions?",
    "Did median 25-delta put implied vol for SYN-A01 rise between 2025-01-02 and 2025-03-27?",
    "Did median 10-delta call implied vol for SYN-B05 rise between 2025-01-02 and 2025-03-27?",
    "Is there a significant difference in 25-delta put implied vol regime between cohort A and cohort B between 2025-01-02 and 2025-03-27?",
    "Is there a significant difference in 50-delta call implied vol regime between cohort A and cohort B between 2025-03-27 and 2025-06-18?",
    "What is the correlation between SYN-A01 and SYN-A02 25-delta put implied vol between 2025-01-02 and 2025-03-27?",
    "What is the correlation between SYN-B01 and SYN-B02 50-delta call implied vol between 2025-01-02 and 2025-03-27?",
    "How did SYN-A01's 10-delta put implied vol change around event #0 within 5 sessions?",
    "How did SYN-B02's 25-delta call implied vol change around event #0 within 8 sessions?",
    "What dividend decile is SYN-A05 in?",
    "What is SYN-A06's 75-delta call 60-day implied vol on 2025-02-06?",
    "How many trading sessions does SYN-B07 have between 2025-03-27 and 2025-06-18?",
    "What is SYN-B08's 90-delta put term structure slope on 2025-02-06?",
]

client = ClaudeCLIClient()

with db.session() as conn:
    ledger = TrialLedger(conn)
    n_calls = 0
    n_parsed = 0
    n_gate_approved = 0
    disagreed_with_router = []

    for q in QUESTIONS:
        n_calls += 1
        print(f"\n[{n_calls}] {q}")
        try:
            shape, params = client.parse_question(q)
            n_parsed += 1
            print(f"  claude -p chose: {shape} {params}")
            f = build_finding(shape, conn, ledger, params)
            result = gate.run_gate(conn, f)
            if result.approved:
                n_gate_approved += 1
                print(f"  APPROVED: {result.text}")
            else:
                print(f"  WITHHELD: {result.text}")
        except ClaudeCLIError as e:
            print(f"  CLI ERROR: {e}")
        except Exception as e:
            print(f"  EXECUTION ERROR: {type(e).__name__}: {e}")

    print("\nSUMMARY")
    print(f"REAL CLAUDE CLI SUBPROCESS CALLS MADE: {n_calls}")
    print(f"successfully parsed into a (shape, params) plan: {n_parsed} / {n_calls}")
    print(f"gate-approved findings: {n_gate_approved} / {n_parsed}")
