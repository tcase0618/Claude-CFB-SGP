import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def small_db(tmp_path_factory):
    """Two-season, 16-team synthetic league with full analytics (built once per session)."""
    from cfbquant.db import open_db
    from cfbquant.ingest.synthetic import SyntheticLeague, build_synthetic
    import cfbquant.ingest.synthetic as syn
    from cfbquant.cli import analyze
    path = tmp_path_factory.mktemp("db") / "test.db"
    orig = SyntheticLeague.__init__

    def small_init(self, seed=7, n_per_conf=4):
        orig(self, seed=seed, n_per_conf=4)
    syn.SyntheticLeague.__init__ = small_init
    try:
        with open_db(path) as conn:
            build_synthetic(conn, (2024, 2025), seed=3)
            analyze(conn, [2024, 2025], log=lambda *a: None)
    finally:
        syn.SyntheticLeague.__init__ = orig
    return path


@pytest.fixture()
def conn(small_db):
    from cfbquant.db import connect
    c = connect(small_db)
    yield c
    c.close()


@pytest.fixture()
def db_copy(small_db, tmp_path):
    p = tmp_path / "copy.db"
    from cfbquant.db import connect
    src = connect(small_db)
    dst = connect(p)
    src.backup(dst)
    src.close()
    yield dst
    dst.close()
