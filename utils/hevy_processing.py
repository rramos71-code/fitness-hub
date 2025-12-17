# utils/hevy_processing.py
import json
import pandas as pd
from datetime import timedelta
from typing import Callable, Iterable, Mapping, Any
from pandas.api.types import is_datetime64tz_dtype
from .hevy_schema import REQUIRED_SET_COLS, ProgressionConfig

DATE_CANDIDATES = [
    "date", "workout_date", "start_time", "startTime", "performed_at",
    "performedAt", "workout.start_time", "workoutStart", "timestamp",
]

SET_DATETIME_CANDIDATES = [
    "start_time", "startTime", "performed_at", "performedAt",
    "workout.start_time", "workoutStart", "timestamp",
]

SET_DATE_ONLY_CANDIDATES = ["date", "workout_date"]

WORKOUT_ID_CANDIDATES = ["workout_id", "workoutId", "workout.id", "id"]

WORKOUT_DATETIME_CANDIDATES = [
    "start_time", "startTime", "performed_at", "performedAt", "timestamp",
]

WORKOUT_DATE_ONLY_CANDIDATES = ["date", "workout_date"]

EXERCISE_CANDIDATES = [
    "exercise_name", "exercise", "name", "exerciseName", "movement",
]

WEIGHT_CANDIDATES = ["weight_kg", "weight", "kg", "load", "weightKg"]
REPS_CANDIDATES = ["reps", "rep_count", "repetitions", "repsCount"]
WARMUP_CANDIDATES = ["is_warmup", "isWarmup", "warmup"]


def _pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if is_datetime64tz_dtype(dt):
        dt = dt.dt.tz_convert(None)
    return dt


def _extract_exercise_name(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                return _extract_exercise_name(parsed)
            except Exception:
                return stripped or None
        return stripped or None
    if isinstance(value, Mapping):
        for key in ("name", "exercise", "exerciseName", "movement"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                nested = nested.get("name")
            if nested:
                return str(nested)
    return str(value)


def _to_bool(series: pd.Series) -> pd.Series:
    def coerce(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s in {"true", "1", "yes", "y", "warmup", "warm"}
    return series.apply(coerce)


def _first_datetime_from_workouts(
    sets_df: pd.DataFrame,
    workouts_df: pd.DataFrame | None,
    warn: Callable[[str], None],
) -> pd.Series | None:
    if workouts_df is None or workouts_df.empty:
        return None

    set_key = _pick_first(sets_df, WORKOUT_ID_CANDIDATES)
    workout_key = _pick_first(workouts_df, WORKOUT_ID_CANDIDATES)

    if set_key is None or workout_key is None:
        overlap_ids = [c for c in workouts_df.columns if "id" in c.lower() and c in sets_df.columns]
        if overlap_ids:
            set_key = workout_key = overlap_ids[0]

    if set_key is None or workout_key is None:
        warn(
            "Could not join Hevy sets to workouts: no shared workout id column. "
            f"Sets columns: {list(sets_df.columns)}, workouts columns: {list(workouts_df.columns)}"
        )
        return None

    time_candidates = [c for c in WORKOUT_DATETIME_CANDIDATES if c in workouts_df.columns]
    date_candidates = [c for c in WORKOUT_DATE_ONLY_CANDIDATES if c in workouts_df.columns]
    if not time_candidates and not date_candidates:
        warn("Workouts data has no timestamp columns to merge into sets.")
        return None

    workout_meta = workouts_df[[workout_key] + time_candidates + date_candidates].drop_duplicates(subset=[workout_key])
    workout_meta = workout_meta.set_index(workout_key)

    for col in time_candidates:
        mapped = sets_df[set_key].map(workout_meta[col])
        dt = _to_naive_datetime(mapped)
        if dt.notna().any():
            return dt

    for col in date_candidates:
        mapped = sets_df[set_key].map(workout_meta[col])
        dt = _to_naive_datetime(mapped)
        if dt.notna().any():
            return dt

    return None


def normalize_hevy_sets(
    sets_df: pd.DataFrame,
    workouts_df: pd.DataFrame | None = None,
    warn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Return a normalized sets DF with REQUIRED_SET_COLS present.
    This is the only place allowed to do column mapping.
    """
    warn = warn or (lambda msg: None)
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(columns=REQUIRED_SET_COLS)

    df = sets_df.copy()
    debug = {"input_rows": len(df), "input_cols": list(df.columns)}

    date_col = _pick_first(df, SET_DATETIME_CANDIDATES)
    ex_col = _pick_first(df, EXERCISE_CANDIDATES)
    w_col = _pick_first(df, WEIGHT_CANDIDATES)
    r_col = _pick_first(df, REPS_CANDIDATES)
    wu_col = _pick_first(df, WARMUP_CANDIDATES)

    # date priority: set timestamp -> workout timestamp -> set date-only
    date_dt = None
    if date_col:
        date_dt = _to_naive_datetime(df[date_col])
        debug["date_source"] = f"sets:{date_col}"

    if date_dt is None or date_dt.notna().sum() == 0:
        from_workouts = _first_datetime_from_workouts(df, workouts_df, warn)
        if from_workouts is not None and from_workouts.notna().any():
            date_dt = from_workouts
            debug["date_source"] = debug.get("date_source", "") + " workout_timestamp"

    if date_dt is None or date_dt.notna().sum() == 0:
        date_only_col = _pick_first(df, SET_DATE_ONLY_CANDIDATES)
        if date_only_col:
            date_dt = _to_naive_datetime(df[date_only_col])
            debug["date_source"] = debug.get("date_source", "") + f" sets:{date_only_col}"

    if date_dt is None:
        date_dt = pd.Series(pd.NaT, index=df.index)

    df["date_dt"] = date_dt.astype("datetime64[ns]")
    df["date"] = df["date_dt"].dt.date
    debug["parsed_dates"] = int(df["date_dt"].notna().sum())

    if df["date_dt"].notna().sum() == 0:
        warn(
            "Hevy sets have no usable date/timestamp columns. "
            f"Checked {DATE_CANDIDATES}. Available columns: {list(sets_df.columns)}"
        )
        empty = pd.DataFrame(columns=REQUIRED_SET_COLS)
        empty.attrs["normalize_debug"] = debug
        return empty

    # exercise name
    if ex_col:
        df["exercise_name"] = df[ex_col].apply(_extract_exercise_name)
    else:
        df["exercise_name"] = None

    # weight and reps
    df["weight_kg"] = pd.to_numeric(df[w_col], errors="coerce") if w_col else 0.0
    df["reps"] = pd.to_numeric(df[r_col], errors="coerce") if r_col else 0

    # warmup
    if wu_col:
        df["is_warmup"] = _to_bool(df[wu_col])
    else:
        df["is_warmup"] = False

    # clean
    missing_date_mask = df["date_dt"].isna()
    missing_ex_mask = df["exercise_name"].isna() | (df["exercise_name"].astype(str).str.len() == 0)
    debug["missing_date_rows"] = int(missing_date_mask.sum())
    debug["missing_exercise_rows"] = int(missing_ex_mask.sum())

    df = df[~missing_date_mask & ~missing_ex_mask]
    df["reps"] = df["reps"].fillna(0).astype(int)
    df["weight_kg"] = df["weight_kg"].fillna(0.0).astype(float)
    df["is_warmup"] = df["is_warmup"].fillna(False).astype(bool)
    debug["output_rows"] = len(df)

    if df.empty:
        warn(
            "All Hevy sets dropped during normalization. Missing date rows: "
            f"{debug['missing_date_rows']}, missing exercise rows: {debug['missing_exercise_rows']}. "
            f"Available columns: {list(sets_df.columns)}"
        )

    df.attrs["normalize_debug"] = debug
    return df


def build_exercise_library(
    sets_norm: pd.DataFrame,
    cfg: ProgressionConfig,
    warn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """
    Aggregate per exercise: sessions, sets, avg/max weight, avg reps, total volume, last seen.
    """
    warn = warn or (lambda msg: None)
    if sets_norm is None or sets_norm.empty:
        return pd.DataFrame(columns=[
            "exercise_name","sessions","sets","avg_weight","max_weight","avg_reps","total_volume","last_seen"
        ])

    df = sets_norm.copy()
    start_rows = len(df)
    drop_reasons: dict[str, int] = {
        "missing_date": 0,
        "missing_exercise_name": 0,
        "warmup_filter": 0,
        "outside_lookback": 0,
    }

    missing_date_mask = df["date_dt"].isna()
    if missing_date_mask.any():
        drop_reasons["missing_date"] = int(missing_date_mask.sum())
        df = df[~missing_date_mask]

    missing_ex_mask = df["exercise_name"].isna() | (df["exercise_name"].astype(str).str.len() == 0)
    if missing_ex_mask.any():
        drop_reasons["missing_exercise_name"] = int(missing_ex_mask.sum())
        df = df[~missing_ex_mask]

    if not cfg.include_warmups and "is_warmup" in df.columns:
        warmup_mask = df["is_warmup"]
        if warmup_mask.any():
            drop_reasons["warmup_filter"] = int(warmup_mask.sum())
        df = df[~warmup_mask]

    if df["date_dt"].notna().any():
        cutoff = df["date_dt"].max() - timedelta(days=cfg.lookback_days - 1)
        candidate = df[df["date_dt"] >= cutoff]
        if candidate.empty:
            warn(
                "Lookback filter would drop all sets; skipping cutoff. "
                f"Latest date: {df['date_dt'].max()} cutoff: {cutoff}"
            )
        else:
            drop_reasons["outside_lookback"] = int(len(df) - len(candidate))
            df = candidate
    else:
        warn("No valid dates available to apply lookback filter; using all rows.")

    if df.empty:
        reason_str = ", ".join([f"{k}: {v}" for k, v in drop_reasons.items()])
        warn("Exercise library is empty after filtering. " + reason_str)
        return pd.DataFrame(columns=[
            "exercise_name","sessions","sets","avg_weight","max_weight","avg_reps","total_volume","last_seen"
        ])

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

    if out.empty and start_rows > 0:
        reason_str = ", ".join([f"{k}: {v}" for k, v in drop_reasons.items()])
        warn("Exercise library is empty after filtering. " + reason_str)

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
