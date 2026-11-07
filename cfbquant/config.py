"""Central configuration. Everything is environment-driven; no secrets in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CFBQ_DATA_DIR", ROOT / "data"))


@dataclass(frozen=True)
class Settings:
    # Which database the API/analytics operate on. Real and synthetic data are
    # stored in *separate files* so demo data can never contaminate real data.
    db_path: Path = field(default_factory=lambda: Path(
        os.environ.get("CFBQ_DB", DATA_DIR / "synthetic.db")))
    cfbd_api_key: str | None = field(default_factory=lambda: os.environ.get("CFBD_API_KEY"))
    cfbd_base_url: str = os.environ.get("CFBD_BASE_URL", "https://api.collegefootballdata.com")
    cfbd_min_interval_s: float = float(os.environ.get("CFBD_MIN_INTERVAL_S", "0.35"))
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")
    default_sims: int = int(os.environ.get("CFBQ_SIMS", "10000"))
    backtest_sims: int = int(os.environ.get("CFBQ_BACKTEST_SIMS", "400"))
    seed: int = int(os.environ.get("CFBQ_SEED", "20241109"))


def get_settings() -> Settings:
    return Settings()
