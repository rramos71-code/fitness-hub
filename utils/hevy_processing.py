from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def _standardize_dates(df: pd.DataFrame, tz_name: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Ensure df has:
      - date (datetime)
      - date_local (datetime localized/converted)
      - date_day (python date)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    tz = ZoneInfo(tz_name)

    # IMPORTANT: include 'workout_date' because Hevy sync writes that column
    date_col = None
    for cand in [
        "date",
        "date_day",
        "workout_date",
        "workout_start_time",
        "workout_start",
        "startTime",
        "start_time",
        "performed_at",
        "performedAt",
        "createdAt",
        "loggedAt",
    ]:
        if cand in df.columns:
            date_col = cand
            break

    if not date_col:
        df["date"] = pd.NaT
        df["date_local"] = pd.NaT
        df["date_day"] = pd.NaT
        return df

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    if getattr(df["date"].dt, "tz", None) is None:
        df["date_local"] = df["date"].dt.tz_localize(
            tz, nonexistent="shift_forward", ambiguous="NaT"
        )
    else:
        df["date_local"] = df["date"].dt.tz_convert(tz)

    df["date_day"] = df["date_local"].dt.date
    return df


def get_hevy_sets_from_session() -> pd.DataFrame:
    """
    Reads Hevy sets from session_state and applies standardization + lookback filtering.
    Expects session_state['hevy_sets_df'] set by the Hevy sync button.
    """
    sets_df = st.session_state.get("hevy_sets_df")
    if sets_df is None or getattr(sets_df, "empty", True):
        return pd.DataFrame()

    sets_df = _standardize_dates(sets_df)

    lookback_days = int(st.session_state.get("hevy_lookback_days", 90))
    if "date_day" in sets_df.columns and sets_df["date_day"].notna().any():
        cutoff = (
            datetime.now(ZoneInfo("Europe/Berlin")).date()
            - timedelta(days=lookback_days)
        )
        sets_df = sets_df[sets_df["date_day"] >= cutoff].copy()

    return sets_df


def build_exercise_library(sets_df: pd.DataFrame, include_warmups: bool = False) -> pd.DataFrame:
    """
    Build an exercise library from Hevy sets.
    Returns columns like:
      - exercise_name
      - last_date
      - last_weight_kg
      - last_reps
      - best_weight_kg
      - total_sets
    """
    if sets_df is None or sets_df.empty:
        return pd.DataFrame()

    df = sets_df.copy()

    # normalize columns that your hevy_client likely provides
    if "exercise" in df.columns and "exercise_name" not in df.columns:
        df["exercise_name"] = df["exercise"]

    if "isWarmup" in df.columns and "is_warmup" not in df.columns:
        df["is_warmup"] = df["isWarmup"]

    if "exercise_name" not in df.columns:
        return pd.DataFrame()

    if "date_day" not in df.columns:
        df = _standardize_dates(df)

    if not include_warmups and "is_warmup" in df.columns:
        df = df[~df["is_warmup"].fillna(False)].copy()

    if "weight" in df.columns and "weight_kg" not in df.columns:
        df["weight_kg"] = pd.to_numeric(df["weight"], errors="coerce")

    if "repetitions" in df.columns and "reps" not in df.columns:
        df["reps"] = pd.to_numeric(df["repetitions"], errors="coerce")

    df = df.sort_values(["exercise_name", "date_day"], ascending=[True, True])

    last_rows = df.groupby("exercise_name", as_index=False).tail(1)

    out = df.groupby("exercise_name", as_index=False).agg(
        total_sets=("exercise_name", "count"),
        best_weight_kg=("weight_kg", "max") if "weight_kg" in df.columns else ("exercise_name", "size"),
        last_date=("date_day", "max"),
    )

    if "weight_kg" in last_rows.columns:
        out = out.merge(
            last_rows[["exercise_name", "weight_kg"]].rename(columns={"weight_kg": "last_weight_kg"}),
            on="exercise_name",
            how="left",
        )

    if "reps" in last_rows.columns:
        out = out.merge(
            last_rows[["exercise_name", "reps"]].rename(columns={"reps": "last_reps"}),
            on="exercise_name",
            how="left",
        )

    out = out.sort_values("total_sets", ascending=False).reset_index(drop=True)
    return out
