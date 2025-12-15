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

def build_exercise_library(
    sets_df,
    lookback_days: int = 90,
    include_warmups: bool = False,
):
    """
    Build a per-exercise summary table from Hevy sets.
    """

    if sets_df is None or sets_df.empty:
        return None

    df = sets_df.copy()

    # Normalize date
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]

    # Optional warmup filter
    if not include_warmups and "isWarmup" in df.columns:
        df = df[~df["isWarmup"]]

    # Required columns guard
    required = {"exercise_name", "weight_kg", "reps"}
    if not required.issubset(df.columns):
        return None

    df["volume"] = df["weight_kg"].fillna(0) * df["reps"].fillna(0)

    exercise_lib = (
        df.groupby("exercise_name")
        .agg(
            sessions=("date", "nunique"),
            sets=("reps", "count"),
            avg_weight=("weight_kg", "mean"),
            max_weight=("weight_kg", "max"),
            avg_reps=("reps", "mean"),
            total_volume=("volume", "sum"),
            last_seen=("date", "max"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
    )

    return exercise_lib
