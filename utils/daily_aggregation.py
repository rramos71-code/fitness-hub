import pandas as pd
from datetime import datetime


def _ensure_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Make sure df has a 'date' column of type datetime.date."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])

    temp = df.copy()

    if date_col not in temp.columns:
        raise KeyError(f"Expected column '{date_col}' in dataframe, got {temp.columns}")

    temp["date"] = pd.to_datetime(temp[date_col]).dt.date
    return temp


def aggregate_nutrition(nutrition_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a dataframe with at least:
      - 'date'
      - 'calories_kcal', 'protein_g', 'carbs_g', 'fat_g'
    and any other numeric nutrient columns.
    """
    if nutrition_df is None or nutrition_df.empty:
        return pd.DataFrame(columns=["date"])

    df = nutrition_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    group_cols = ["date"]
    agg_cols = [c for c in numeric_cols if c not in group_cols]

    out = (
        df.groupby("date", as_index=False)[agg_cols].sum()
        if agg_cols
        else df[["date"]].drop_duplicates()
    )

    # Make column names explicit
    rename_map = {
        "calories": "calories_kcal",
        "energy_kcal": "calories_kcal",
    }
    out = out.rename(columns=rename_map)

    return out


def aggregate_garmin_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Garmin daily summary.

    Expects columns like (best effort, will only use what exists):
      - 'calendarDate' or 'date'
      - 'steps', 'distance', 'activeKilocalories', 'kilocalories',
        'sleepSeconds', 'averageHeartRate', 'restingHeartRate', etc.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["date"])

    # Infer the date column name
    date_col = "date"
    if "calendarDate" in daily_df.columns:
        date_col = "calendarDate"

    df = _ensure_date_column(daily_df, date_col)

    # Map Garmin raw names to nicer names
    rename_map = {}
    for col in df.columns:
        if col == "steps":
            rename_map[col] = "garmin_steps"
        elif col in ("distance", "distanceInMeters"):
            rename_map[col] = "garmin_distance_m"
        elif col == "activeKilocalories":
            rename_map[col] = "garmin_active_kcal"
        elif col == "kilocalories":
            rename_map[col] = "garmin_total_kcal"
        elif col == "sleepSeconds":
            rename_map[col] = "garmin_sleep_seconds"
        elif col == "averageHeartRate":
            rename_map[col] = "garmin_avg_hr_bpm"
        elif col == "restingHeartRate":
            rename_map[col] = "garmin_resting_hr_bpm"

    df = df.rename(columns=rename_map)

    keep_numeric = df.select_dtypes(include="number").columns.tolist()
    agg_cols = [c for c in keep_numeric if c != "date"]

    out = (
        df.groupby("date", as_index=False)[agg_cols].sum()
        if agg_cols
        else df[["date"]].drop_duplicates()
    )

    # Convert distance to km and sleep to hours where present
    if "garmin_distance_m" in out.columns:
        out["garmin_distance_km"] = out["garmin_distance_m"] / 1000.0

    if "garmin_sleep_seconds" in out.columns:
        out["garmin_sleep_hours"] = out["garmin_sleep_seconds"] / 3600.0

    return out


def aggregate_garmin_activities(activities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Garmin activities per date.

    Expects something like:
      - 'startTimeLocal' or 'startTimeGMT'
      - 'duration' (seconds)
      - 'activityType' or 'activityType.typeId' / 'activityType.typeKey'
      - 'calories'
    All activities are included: runs, walks, rides, kettlebell, bands, "other", etc.
    """
    if activities_df is None or activities_df.empty:
        return pd.DataFrame(columns=["date"])

    df = activities_df.copy()

    # Date
    date_col = "startTimeLocal"
    if date_col not in df.columns:
        if "startTimeGMT" in df.columns:
            date_col = "startTimeGMT"
        elif "startTime" in df.columns:
            date_col = "startTime"

    df = _ensure_date_column(df, date_col)

    # Activity type
    if "activityType.typeKey" in df.columns:
        df["activity_type"] = df["activityType.typeKey"]
    elif "activityType" in df.columns:
        df["activity_type"] = df["activityType"]
    else:
        df["activity_type"] = "unknown"

    # Duration in minutes
    duration_col = None
    for cand in ("duration", "durationInSeconds", "elapsedDuration"):
        if cand in df.columns:
            duration_col = cand
            break

    if duration_col:
        df["duration_min"] = df[duration_col] / 60.0
    else:
        df["duration_min"] = 0.0

    if "calories" not in df.columns:
        df["calories"] = 0.0

    # Simple buckets for cardio / strength / other
    def classify_activity(a_type: str) -> str:
        if not isinstance(a_type, str):
            return "other"
        t = a_type.lower()
        if any(k in t for k in ["run", "walk", "bike", "ride", "elliptical", "cardio"]):
            return "cardio"
        if any(k in t for k in ["strength", "weights", "kettlebell", "resistance", "hiit"]):
            return "strength"
        return "other"

    df["bucket"] = df["activity_type"].map(classify_activity)

    agg = (
        df.groupby(["date", "bucket"])
        .agg(
            garmin_activity_minutes=("duration_min", "sum"),
            garmin_activity_calories=("calories", "sum"),
        )
        .reset_index()
    )

    # Pivot buckets to columns
    out = agg.pivot_table(
        index="date",
        columns="bucket",
        values=["garmin_activity_minutes", "garmin_activity_calories"],
        fill_value=0,
    )

    out.columns = ["_".join(col).strip() for col in out.columns.to_flat_index()]
    out = out.reset_index()

    # Total minutes and calories across all buckets
    minute_cols = [c for c in out.columns if c.startswith("garmin_activity_minutes_")]
    cal_cols = [c for c in out.columns if c.startswith("garmin_activity_calories_")]

    out["garmin_total_activity_minutes"] = out[minute_cols].sum(axis=1)
    out["garmin_activity_calories_kcal"] = out[cal_cols].sum(axis=1)

    return out


def aggregate_hevy_sets(sets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Hevy sets per date.

    Expects a flattened sets dataframe with at least:
      - 'date'
      - 'set_index'
      - 'weight_kg'
      - 'reps'
      - 'exercise_name'
    """
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(columns=["date"])

    df = sets_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["volume_kg"] = df["weight_kg"].fillna(0) * df["reps"].fillna(0)

    agg = (
        df.groupby("date")
        .agg(
            hevy_total_sets=("set_index", "count"),
            hevy_total_reps=("reps", "sum"),
            hevy_total_volume_kg=("volume_kg", "sum"),
        )
        .reset_index()
    )

    # Average load per working set
    agg["hevy_avg_load_kg"] = agg["hevy_total_volume_kg"] / agg["hevy_total_sets"].replace(0, pd.NA)

    return agg


def add_metadata(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived date and training flags."""
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["date"])

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["weekday"] = df["date"].apply(lambda d: d.weekday())  # 0 Monday
    df["year"] = df["date"].apply(lambda d: d.year)
    df["month"] = df["date"].apply(lambda d: d.month)
    df["iso_year"] = df["date"].apply(lambda d: d.isocalendar()[0])
    df["iso_week"] = df["date"].apply(lambda d: d.isocalendar()[1])

    # Simple flags
    df["training_day_flag"] = (
        (df.get("garmin_total_activity_minutes", 0) > 0)
        | (df.get("hevy_total_sets", 0) > 0)
    )

    df["strength_day_flag"] = (
        (df.get("garmin_activity_minutes_strength", 0) > 0)
        | (df.get("hevy_total_sets", 0) > 0)
    )

    df["cardio_day_flag"] = df.get("garmin_activity_minutes_cardio", 0) > 0

    # Energy balance if both sides present
    if "calories_kcal" in df.columns and "garmin_total_kcal" in df.columns:
        df["energy_balance_kcal"] = df["calories_kcal"] - df["garmin_total_kcal"]

    return df


def build_daily_dataset(
    nutrition_df: pd.DataFrame,
    garmin_daily_df: pd.DataFrame,
    garmin_activities_df: pd.DataFrame,
    hevy_sets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build unified daily dataframe.
    """
    nut = aggregate_nutrition(nutrition_df)
    g_daily = aggregate_garmin_daily(garmin_daily_df)
    g_act = aggregate_garmin_activities(garmin_activities_df)
    hevy = aggregate_hevy_sets(hevy_sets_df)

    dfs = [nut, g_daily, g_act, hevy]

    # Outer join on date so missing pieces are allowed
    out = None
    for part in dfs:
        if part is None or part.empty:
            continue
        if out is None:
            out = part.copy()
        else:
            out = out.merge(part, on="date", how="outer")

    if out is None:
        return pd.DataFrame(columns=["date"])

    out = out.sort_values("date").reset_index(drop=True)

    # Replace NaNs with 0 for numeric metrics
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].fillna(0)

    out = add_metadata(out)
    return out
