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

def build_exercise_progression(
    sets_df,
    lookback_days: int = 90,
    include_warmups: bool = False,
):
    """
    Build per-exercise progression metrics from Hevy sets.
    Used to decide load increases / stalls / deloads.
    """

    if sets_df is None or sets_df.empty:
        return None

    df = sets_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]

    if not include_warmups and "isWarmup" in df.columns:
        df = df[~df["isWarmup"]]

    required = {"exercise_name", "weight_kg", "reps", "date"}
    if not required.issubset(df.columns):
        return None

    df["volume"] = df["weight_kg"].fillna(0) * df["reps"].fillna(0)

    # Session-level aggregation
    session = (
        df.groupby(["exercise_name", "date"])
        .agg(
            avg_weight=("weight_kg", "mean"),
            max_weight=("weight_kg", "max"),
            total_reps=("reps", "sum"),
            total_volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["exercise_name", "date"])
    )

    progression_rows = []

    for exercise, g in session.groupby("exercise_name"):
        g = g.sort_values("date")

        if len(g) < 2:
            continue

        first = g.iloc[0]
        last = g.iloc[-1]

        progression_rows.append(
            {
                "exercise_name": exercise,
                "sessions": len(g),
                "start_weight": first["avg_weight"],
                "current_weight": last["avg_weight"],
                "max_weight": g["max_weight"].max(),
                "weight_change": last["avg_weight"] - first["avg_weight"],
                "volume_change_pct": (
                    (last["total_volume"] - first["total_volume"]) / first["total_volume"]
                    if first["total_volume"] > 0
                    else 0
                ),
                "last_session": last["date"],
            }
        )

    return pd.DataFrame(progression_rows).sort_values(
        ["weight_change", "volume_change_pct"],
        ascending=False,
    )

def build_progression_recommendations(
    progression_df,
    *,
    min_sessions: int = 3,
    min_weight_increase: float = 1.25,
    deload_threshold: float = -2.5,
):
    """
    Generate simple load recommendations per exercise
    based on recent progression trends.
    """

    if progression_df is None or progression_df.empty:
        return None

    rows = []

    for _, row in progression_df.iterrows():
        exercise = row["exercise_name"]
        sessions = row["sessions"]
        weight_change = row["weight_change"]
        volume_change = row["volume_change_pct"]

        recommendation = "hold"
        reason = "Insufficient signal"
        confidence = "low"

        if sessions >= min_sessions:
            confidence = "medium"

            if weight_change >= min_weight_increase:
                recommendation = "increase"
                reason = "Consistent load progression observed"
                confidence = "high"

            elif weight_change <= deload_threshold:
                recommendation = "deload"
                reason = "Performance regression detected"

            elif volume_change > 0.15:
                recommendation = "increase"
                reason = "Volume increasing without load increase"

            else:
                recommendation = "hold"
                reason = "Stable performance"

        rows.append(
            {
                "exercise_name": exercise,
                "recommendation": recommendation,
                "confidence": confidence,
                "sessions_observed": sessions,
                "weight_change": round(weight_change, 2),
                "volume_change_pct": round(volume_change * 100, 1),
                "reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["recommendation", "confidence"],
        ascending=[True, False],
    )


def ensure_hevy_date_column(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if "date" in df.columns:
        return df

    for candidate in [
        "performed_at",
        "workout_start",
        "start_time",
        "created_at",
        "timestamp",
    ]:
        if candidate in df.columns:
            df["date"] = pd.to_datetime(df[candidate]).dt.date
            return df

    raise ValueError(
        "Hevy sets dataframe has no recognizable datetime column "
        "to derive 'date'."
    )
