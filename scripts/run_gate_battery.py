"""The 60 broken-finding / 200 correct-finding gate battery."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate import db, broken_bank, gate, question_bank
from vol_gate.ledger import TrialLedger

with db.session() as conn:
    ledger = TrialLedger(conn)
    correct_pool = question_bank.build_correct_question_set(conn, ledger, n=200)
    print(f"correct findings built: {len(correct_pool)} / 200 requested")

    false_blocks = []
    for item in correct_pool:
        result = gate.run_gate(conn, item["finding"])
        if not result.approved:
            false_blocks.append((item["question"], result.text))
    print(f"\n--- false-block check on {len(correct_pool)} correct findings ---")
    print(f"incorrectly blocked: {len(false_blocks)} / {len(correct_pool)}")
    for q, t in false_blocks:
        print(f"  FALSE BLOCK: {q}\n    {t}")

    broken = broken_bank.build_broken_set(conn, ledger, correct_pool, n_fabricated=30, n_silent=30)
    n_fab = sum(1 for b in broken if b["label"] == "fabricated")
    n_sil = sum(1 for b in broken if b["label"] == "silent_error")
    print(f"\nbroken findings built: {len(broken)} ({n_fab} fabricated, {n_sil} silent_error)")

    caught, missed = 0, []
    for item in broken:
        result = gate.run_gate(conn, item["finding"])
        if not result.approved:
            caught += 1
        else:
            missed.append((item["label"], item["finding"].text))
    print(f"\n--- catch rate on {len(broken)} deliberately broken findings ---")
    print(f"caught (withheld, naming the disagreeing figure): {caught} / {len(broken)}")
    for label, text in missed:
        print(f"  MISSED ({label}): {text}")

    print("\n--- sample withheld responses (first 3) ---")
    shown = 0
    for item in broken:
        result = gate.run_gate(conn, item["finding"])
        if not result.approved and shown < 3:
            print(f"[{item['label']}]\n{result.text}\n")
            shown += 1

    print("SUMMARY")
    print(f"FABRICATION+SILENT-ERROR CATCH RATE: {caught}/{len(broken)}")
    print(f"FALSE BLOCK RATE ON CORRECT FINDINGS: {len(false_blocks)}/{len(correct_pool)}")
