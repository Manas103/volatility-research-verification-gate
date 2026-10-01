import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from vol_gate import db, seed


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    seed.seed_database(c, seed=seed.DEFAULT_SEED)
    yield c
    c.close()
