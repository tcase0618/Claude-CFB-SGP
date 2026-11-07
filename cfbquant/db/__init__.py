"""Database connection + migrations (stdlib sqlite3; schema is Postgres-portable)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# numpy scalars must bind as SQL integers/reals (otherwise sqlite3 silently matches nothing)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.bool_, bool)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    cur = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    if cur is None or cur < SCHEMA_VERSION:
        conn.execute("INSERT OR REPLACE INTO schema_migrations VALUES (?, ?)", (SCHEMA_VERSION, utcnow()))
    conn.commit()


@contextmanager
def open_db(db_path: str | Path):
    conn = connect(db_path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()


def df(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def insert_df(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, replace: bool = True) -> int:
    """Bulk insert a DataFrame; columns must match table columns."""
    if frame is None or frame.empty:
        return 0
    cols = list(frame.columns)
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    sql = f"{verb} INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    rows = [tuple(None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, 'item') else v)
                  for v in r) for r in frame.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    return len(rows)


def is_synthetic(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) FROM data_sources WHERE is_synthetic=1").fetchone()
    real = conn.execute("SELECT COUNT(*) FROM data_sources WHERE is_synthetic=0").fetchone()
    return bool(row[0]) and not bool(real[0])
