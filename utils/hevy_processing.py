# utils/hevy_processing.py
import pandas as pd
from datetime import timedelta
from .hevy_schema import REQUIRED_SET_COLS, ProgressionConfig

DATE_CANDIDATES = [
    "date", "workout_date", "start_time", "startTime", "performed_at",
    "performedAt", "workout.start_time", "workoutStart", "timestamp",
]

EXERCISE_CANDIDATES = [
    "exercise_name", "exercise", "name", "exerciseName", "movement",
]

WEIGHT_CANDIDATES = ["weight_kg", "weight", "kg", "load", "weightKg"]
REPS_CANDIDATES = ["reps", "rep_count", "repetitions", "repsCount"]
WARMUP_CANDIDATES = ["is_warmup", "isWarmup", "warmup"]

def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_hevy_sets(sets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a normalized sets DF with REQUIRED_SET_COLS present.
    This is the only place allowed to do column mapping.
    """
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(columns=REQUIRED_SET_COLS)

    df = sets_df.copy()

    date_col = _pick_first(df, DATE_CANDIDATES)
    ex_col = _pick_first(df, EXERCISE_CANDIDATES)
    w_col = _pick_first(df, WEIGHT_CANDIDATES)
    r_col = _pick_first(df, REPS_CANDIDATES)
    wu_col = _pick_first(df, WARMUP_CANDIDATES)

    # date
    if date_col is None:
        df["date_dt"] = pd.NaT
    else:
        df["date_dt"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)

    df["date"] = df["date_dt"].dt.date

    # exercise name
    df["exercise_name"] = df[ex_col].astype(str) if ex_col else None

    # weight and reps
    df["weight_kg"] = pd.to_numeric(df[w_col], errors="coerce") if w_col else 0.0
    df["reps"] = pd.to_numeric(df[r_col], errors="coerce") if r_col else 0

    # warmup
    if wu_col:
        df["is_warmup"] = df[wu_col].astype(bool)
    else:
        df["is_warmup"] = False

    # clean
    df = df.dropna(subset=["date", "exercise_name"])
    df = df[df["exercise_name"].str.len() > 0]
    df["reps"] = df["reps"].fillna(0).astype(int)
    df["weight_kg"] = df["weight_kg"].fillna(0.0).astype(float)

    return df

def build_exercise_library(sets_norm: pd.DataFrame, cfg: ProgressionConfig) -> pd.DataFrame:
    """
    Aggregate per exercise: sessions, sets, avg/max weight, avg reps, total volume, last seen.
    """
    if sets_norm is None or sets_norm.empty:
        return pd.DataFrame(columns=[
            "exercise_name","sessions","sets","avg_weight","max_weight","avg_reps","total_volume","last_seen"
        ])

    df = sets_norm.copy()
    if not cfg.include_warmups:
        df = df[~df["is_warmup"]]

    cutoff = df["date_dt"].max() - timedelta(days=cfg.lookback_days - 1)
    df = df[df["date_dt"] >= cutoff]

    df["volume"] = df["weight_kg"] * df["reps"]

    out = (
        df.groupby("exercise_name", as_index=False)
          .agg(
              sessions=("date", "nunique"),
              sets=("reps", "count"),
              avg_weight=("weight_kg", "mean"),
              max_weight=("weight_kg", "max"),
              avg_reps=("reps", "mean"),
              total_volume=("volume", "sum"),
              last_seen=("date", "max"),
          )
          .sort_values(["sessions","total_volume"], ascending=False)
    )

    return out

def build_exercise_progression(sets_norm: pd.DataFrame, cfg: ProgressionConfig) -> pd.DataFrame:
    """
    Compute start/current/top weight, and volume deltas across the lookback.
    """
    # TODO implement:
    # - sessionize by date + exercise
    # - compute per session top set weight and total volume
    # - compare earliest vs latest in window
    return pd.DataFrame(columns=[
        "exercise_name","sessions","start_weight","current_weight","max_weight","weight_change","volume_change","trend"
    ])

def build_progression_recommendations(prog_df: pd.DataFrame, cfg: ProgressionConfig) -> pd.DataFrame:
    """
    Rule-based suggestions based on progression signals.
    """
    # TODO implement:
    # - if progressing: consider small increase if reps near ceiling
    # - if plateau: add reps or sets, or microload
    # - if regressing: deload
    return pd.DataFrame(columns=[
        "exercise_name","recommendation","next_weight_kg","target_reps","rationale"
    ])
