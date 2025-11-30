import pandas as pd
import numpy as np

def aggregate_hevy_daily(hevy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw Hevy sets into daily strength metrics.
    Expects at least: start_time, set_type, weight_kg, reps, title.
    """

    if hevy_df is None or hevy_df.empty:
        return pd.DataFrame(columns=[
            "date",
            "gym_sets_total",
            "gym_sets_working",
            "gym_volume_kg",
            "gym_sessions",
        ])

    df = hevy_df.copy()

    # 1) Get a pure date column from start_time
    if "date" not in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["date"] = df["start_time"].dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # 2) Tag working sets (exclude warmups and similar)
    df["set_type"] = df["set_type"].fillna("").astype(str)
    df["is_working_set"] = ~df["set_type"].str.lower().isin(["warmup", "warm up", "deload"])

    # 3) Volume per set for working sets only
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce").fillna(0.0)
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce").fillna(0.0)

    df["set_volume_kg"] = np.where(
        df["is_working_set"],
        df["weight_kg"] * df["reps"],
        0.0,
    )

    # 4) Sessions per day, based on workout title
    session_col = "title" if "title" in df.columns else None
    if session_col:
        sessions_per_day = (
            df.groupby("date")[session_col]
            .nunique()
            .rename("gym_sessions")
        )
    else:
        sessions_per_day = (
            df.groupby("date")
            .size()
            .rename("gym_sessions")
            .clip(lower=1)
        )

    # 5) Aggregate per date
    grouped = df.groupby("date").agg(
        gym_sets_total=("set_type", "count"),
        gym_sets_working=("is_working_set", "sum"),
        gym_volume_kg=("set_volume_kg", "sum"),
    ).reset_index()

    grouped = grouped.merge(
        sessions_per_day.reset_index(),
        on="date",
        how="left",
    )

    return grouped
