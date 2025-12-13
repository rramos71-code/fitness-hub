import pandas as pd
from zoneinfo import ZoneInfo
import streamlit as st


def canonicalize_hevy_sets(hevy_sets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Hevy sets into a consistent schema.
    """
    if hevy_sets_df is None or hevy_sets_df.empty:
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

    df = hevy_sets_df.copy()

    # Date handling
    tz = ZoneInfo("Europe/Berlin")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "startTime" in df.columns:
        df["date"] = pd.to_datetime(df["startTime"])
    else:
        df["date"] = pd.NaT

    df["date_day"] = df["date"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT", errors="coerce").dt.date if df["date"].dt.tz is None else df["date"].dt.tz_convert(tz).dt.date

    # Names
    df["workout_name"] = df.get("workout_name") if "workout_name" in df.columns else df.get("workoutName")
    df["template_name"] = df.get("template_name") if "template_name" in df.columns else df.get("templateName")
    if "workout_name" not in df.columns and "workoutName" in df.columns:
        df["workout_name"] = df["workoutName"]
    if df.get("template_name") is None and "templateName" in df.columns:
        df["template_name"] = df["templateName"]
    df["exercise_name"] = df.get("exercise_name") if "exercise_name" in df.columns else df.get("exerciseName")
    df["set_index"] = df.get("set_index") if "set_index" in df.columns else df.get("setIndex")
    df["set_type"] = df.get("set_type") if "set_type" in df.columns else df.get("setType")

    # Drop missing exercise name
    df = df[df["exercise_name"].notna()].copy()

    # Numeric conversions
    df["weight_kg"] = pd.to_numeric(df["weight_kg"] if "weight_kg" in df.columns else df.get("weight"), errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce") if "reps" in df.columns else pd.to_numeric(df.get("repetitions"), errors="coerce")
    df["rir"] = pd.to_numeric(df["rir"], errors="coerce") if "rir" in df.columns else pd.to_numeric(df.get("rpe"), errors="coerce")

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
