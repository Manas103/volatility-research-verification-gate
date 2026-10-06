import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate import db, seed, volstore

with db.session() as conn:
    stats = seed.seed_database(conn)
    print(stats)
    n_vol_rows = volstore.import_csv(conn)
    print(f"volatility_bins imported: {n_vol_rows} rows (from order-book-signal-research)")
