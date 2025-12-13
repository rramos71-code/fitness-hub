import pandas as pd
from zoneinfo import ZoneInfo
import streamlit as st


def canonicalize_hevy_sets(hevy_sets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Hevy sets into a consistent schema.
    """
    empty_schema = pd.DataFrame(
        columns=[
            "date",
            "date_day",
            "workout_name",
            "template_name",
            "exercise_name",
            "set_index",
            "set_type",
            "weight_kg",
            "reps",
            "rir",
            "is_warmup",
            "is_working_set",
        ]
    )

    if hevy_sets_df is None or len(getattr(hevy_sets_df, "index", [])) == 0:
        return empty_schema

    df = pd.DataFrame(hevy_sets_df).copy()

    # Date handling
    tz = ZoneInfo("Europe/Berlin")
    date_col = None
    for cand in ["date", "startTime", "start_time", "performed_at", "performedAt", "createdAt"]:
        if cand in df.columns:
            date_col = cand
            break

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.NaT

    try:
        if df["date"].dt.tz is None:
            df["date_local"] = df["date"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        else:
            df["date_local"] = df["date"].dt.tz_convert(tz)
    except Exception:
        df["date_local"] = df["date"]

    df["date_day"] = df["date_local"].dt.date

    # Names
    df["workout_name"] = df["workout_name"] if "workout_name" in df.columns else df.get("workoutName")
    df["template_name"] = df["template_name"] if "template_name" in df.columns else df.get("templateName")
    df["exercise_name"] = df["exercise_name"] if "exercise_name" in df.columns else df.get("exerciseName")
    df["set_index"] = df["set_index"] if "set_index" in df.columns else df.get("setIndex")
    df["set_type"] = df["set_type"] if "set_type" in df.columns else df.get("setType")

    # Drop missing exercise name
    df = df[df["exercise_name"].notna()].copy()
    if df.empty:
        return empty_schema

    # Numeric conversions
    weight_series = df["weight_kg"] if "weight_kg" in df.columns else df.get("weight")
    df["weight_kg"] = pd.to_numeric(weight_series, errors="coerce")
    reps_series = df["reps"] if "reps" in df.columns else df.get("repetitions")
    df["reps"] = pd.to_numeric(reps_series, errors="coerce")
    rir_series = df["rir"] if "rir" in df.columns else df.get("rpe")
    df["rir"] = pd.to_numeric(rir_series, errors="coerce")

    # Set type normalization
    df["set_type"] = df["set_type"].fillna("work").astype(str).str.lower()
    df["is_warmup"] = df["set_type"].str.contains("warm", na=False) | df["set_type"].isin({"warmup", "warm-up"})
    df["is_working_set"] = (~df["is_warmup"]) & df["reps"].notna() & df["weight_kg"].notna()

    # Defaults
    df["template_name"] = df["template_name"].fillna(df["workout_name"])

    # Final column order
    cols = [
        "date",
        "date_day",
        "workout_name",
        "template_name",
        "exercise_name",
        "set_index",
        "set_type",
        "weight_kg",
        "reps",
        "rir",
        "is_warmup",
        "is_working_set",
    ]

    return df[cols]


def get_hevy_sets_from_session() -> pd.DataFrame:
    """
    Return canonicalized hevy sets from session or empty DataFrame.
    """
    hevy_sets_df = st.session_state.get("hevy_sets_df")
    try:
        return canonicalize_hevy_sets(hevy_sets_df)
    except Exception:
        return pd.DataFrame(
            columns=[
                "date",
                "date_day",
                "workout_name",
                "template_name",
                "exercise_name",
                "set_index",
                "set_type",
                "weight_kg",
                "reps",
                "rir",
                "is_warmup",
                "is_working_set",
            ]
        )
