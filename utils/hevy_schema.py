# utils/hevy_schema.py
from dataclasses import dataclass

REQUIRED_SET_COLS = [
    "date",          # python date
    "date_dt",       # pandas datetime64
    "exercise_name",
    "weight_kg",
    "reps",
    "is_warmup",
]

@dataclass
class ProgressionConfig:
    lookback_days: int = 90
    include_warmups: bool = False
    min_sessions: int = 2
    min_sets: int = 5
    mode: str = "double_progression"  # or "linear", "volume"
    rep_floor: int = 6
    rep_ceiling: int = 10
    small_increment_kg: float = 2.5
    large_increment_kg: float = 5.0
    deload_pct: float = 0.9
    plateau_sessions: int = 3
