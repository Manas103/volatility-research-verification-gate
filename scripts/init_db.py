import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vol_gate import db, seed

with db.session() as conn:
    stats = seed.seed_database(conn)
    print(stats)
