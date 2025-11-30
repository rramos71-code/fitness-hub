import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

# ------------- Helpers and global keys -------------

DAILY_SUMMARY_KEY = "daily_summary"
ACTIVITIES_KEY = "activities"
SETTINGS_KEY = "settings"


def init_session_state():
    if SETTINGS_KEY not in st.session_state:
        st.session_state[SETTINGS_KEY] = {
            "goal_type": "recomp",
            "target_calories": 2200,
            "target_protein": 120,
            "trip_start_chile": None,
            "trip_end_chile": None,
            "sleep_target_hours": 7.0,
        }
    if ACTIVITIES_KEY not in st.session_state:
        st.session_state[ACTIVITIES_KEY] = pd.DataFrame()
    if DAILY_SUMMARY_KEY not in st.session_state:
        st.session_state[DAILY_SUMMARY_KEY] = pd.DataFrame()


# ------------- Settings page -------------

def settings_page():
    st.header("Settings")

    settings = st.session_state[SETTINGS_KEY]

    st.subheader("Goal")
    goal = st.selectbox(
        "Primary goal",
        ["recomp", "lean_bulk", "cut"],
        index=["recomp", "lean_bulk", "cut"].index(settings["goal_type"]),
    )
    settings["goal_type"] = goal

    col1, col2 = st.columns(2)
    with col1:
        settings["target_calories"] = st.number_input(
            "Target daily calories",
            value=float(settings["target_calories"]),
            step=50.0,
        )
    with col2:
        settings["target_protein"] = st.number_input(
            "Target daily protein (g)",
            value=float(settings["target_protein"]),
            step=5.0,
        )

    st.subheader("Travel and sleep")
    c1, c2, c3 = st.columns(3)
    with c1:
        settings["trip_start_chile"] = st.date_input(
            "Chile trip start",
            value=settings["trip_start_chile"],
        )
    with c2:
        settings["trip_end_chile"] = st.date_input(
            "Chile trip end",
            value=settings["trip_end_chile"],
        )
    with c3:
        settings["sleep_target_hours"] = st.number_input(
            "Target sleep hours",
            value=float(settings["sleep_target_hours"]),
            step=0.5,
        )

    st.session_state[SETTINGS_KEY] = settings
    st.success("Settings saved in session (for now).")


# ------------- Data import helpers -------------

def parse_garmin_activities(file) -> pd.DataFrame:
    """
    Parse Garmin activities CSV export.

    Goal: normalize different Garmin export formats into:
      date, start_time, duration_minutes, total_calories_burned, raw_type, source

    It tries several common column name variants for:
      - start time
      - duration
      - calories
      - activity type
    """

    df = pd.read_csv(file)

    # Helper to pick the first existing column from a list of candidates
    def pick(col_candidates):
        for c in col_candidates:
            if c in df.columns:
                return c
        return None

    # 1) Start time
    start_col = pick([
        "Start Time",
        "Start time",
        "Start",
        "Activity Start Date",
        "StartTime",
        "startTime",
        "start_time",
    ])
    if start_col is None:
        raise ValueError(
            f"Could not find a start time column in Garmin file. "
            f"Available columns: {list(df.columns)}"
        )

    df["start_time"] = pd.to_datetime(df[start_col], errors="coerce")
    df["date"] = df["start_time"].dt.date

    # 2) Duration in minutes
    # Common patterns:
    #   - "Duration" in seconds
    #   - "Elapsed Time" in seconds
    #   - "Duration" as "hh:mm:ss" string
    duration_col = pick([
        "Duration",
        "Elapsed Time",
        "ElapsedTime",
        "duration",
        "elapsed_time",
    ])

    duration_minutes = None
    if duration_col is not None:
        # Try numeric seconds first
        if pd.api.types.is_numeric_dtype(df[duration_col]):
            duration_minutes = df[duration_col] / 60.0
        else:
            # Try to parse "hh:mm:ss" or "mm:ss"
            def parse_duration(val):
                if pd.isna(val):
                    return np.nan
                s = str(val)
                parts = s.split(":")
                try:
                    if len(parts) == 3:
                        h, m, sec = map(float, parts)
                        return (h * 3600 + m * 60 + sec) / 60.0
                    if len(parts) == 2:
                        m, sec = map(float, parts)
                        return (m * 60 + sec) / 60.0
                except Exception:
                    return np.nan
                return np.nan

            duration_minutes = df[duration_col].apply(parse_duration)
    else:
        duration_minutes = np.nan

    df["duration_minutes"] = duration_minutes

    # 3) Calories burned
    calories_col = pick([
        "Calories",
        "calories",
        "Calories Burned",
        "Energy Expenditure",
        "Total Calories",
    ])
    if calories_col is not None:
        df["total_calories_burned"] = df[calories_col]
    else:
        df["total_calories_burned"] = 0.0

    # 4) Activity type
    raw_type_col = pick([
        "Activity Type",
        "Activity type",
        "Type",
        "Sport",
        "Activity",
        "activity_type",
    ])
    if raw_type_col is not None:
        df["raw_type"] = df[raw_type_col]
    else:
        df["raw_type"] = "unknown"

    df["source"] = "garmin"

    return df[["date", "start_time", "duration_minutes",
               "total_calories_burned", "raw_type", "source"]]


def parse_hevy_workouts(file) -> pd.DataFrame:
    """
    Parse Hevy CSV export.

    Input format (per set):
      title, start_time, end_time, description, exercise_title,
      superset_id, exercise_notes, set_index, set_type,
      weight_kg, reps, distance_km, duration_seconds, rpe

    We convert this into one row per workout session with:
      date, start_time, duration_minutes, total_calories_burned, raw_type, source
    """

    df = pd.read_csv(file)

    # Parse start and end datetime
    # Example: "28 Nov 2025, 07:05"
    df["start_dt"] = pd.to_datetime(df["start_time"], format="%d %b %Y, %H:%M")
    df["end_dt"] = pd.to_datetime(df["end_time"], format="%d %b %Y, %H:%M")

    # Workout date from start time
    df["date"] = df["start_dt"].dt.date

    # Duration of the whole workout in minutes
    # Same start/end for all sets in a workout, so we can compute once per row and then aggregate
    df["duration_minutes"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60.0

    # Group by workout, not by set
    grouped = (
        df.groupby(["title", "start_dt", "end_dt"], as_index=False)
        .agg(
            date=("date", "first"),
            duration_minutes=("duration_minutes", "max"),
        )
    )

    # Normalized columns expected by the app
    grouped = grouped.rename(columns={"start_dt": "start_time"})
    grouped["total_calories_burned"] = 0.0  # We will use Garmin for calories
    grouped["raw_type"] = "strength"
    grouped["source"] = "hevy"

    return grouped[["date", "start_time", "duration_minutes",
                    "total_calories_burned", "raw_type", "source"]]


def parse_mfp_nutrition(file) -> pd.DataFrame:
    """
    MVP: parse MyFitnessPal CSV export.

    Output columns:
      date, calories_in, protein_g, carbs_g, fats_g
    """
    df = pd.read_csv(file)

    # Adjust these based on the actual export columns
    date_col = "Date" if "Date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col]).dt.date

    calories_col = "Calories" if "Calories" in df.columns else "calories"
    df["calories_in"] = df[calories_col]

    for macro, alt in [("Protein", "protein"), ("Carbs", "carbs"), ("Fat", "fat")]:
        if macro in df.columns:
            df[f"{macro.lower()}_g"] = df[macro]
        elif alt in df.columns:
            df[f"{macro.lower()}_g"] = df[alt]
        else:
            df[f"{macro.lower()}_g"] = np.nan

    return df[["date", "calories_in", "protein_g", "carbs_g", "fat_g"]]


# ------------- Classifier and daily summary -------------

def classify_sessions(activities: pd.DataFrame,
                      settings: dict) -> pd.DataFrame:
    """
    Simple rule based classifier:
      - if source is hevy -> gym
      - if raw_type contains "Run" or GPS flag -> cardio (GPS not yet used)
      - if raw_type contains "Strength" and source is garmin and no hevy that day -> functional
      - if date within Chile trip and no hevy that day -> home
    """

    if activities.empty:
        return activities

    df = activities.copy()

    # mark days with hevy
    hevy_days = df.loc[df["source"] == "hevy", "date"].unique()

    def classify_row(row):
        d = row["date"]
        raw_type = str(row["raw_type"]).lower()
        src = row["source"]

        in_chile = False
        if settings.get("trip_start_chile") and settings.get("trip_end_chile"):
            in_chile = settings["trip_start_chile"] <= d <= settings["trip_end_chile"]

        if src == "hevy":
            return "gym"
        if "run" in raw_type or "cycling" in raw_type:
            return "cardio"
        if src == "garmin" and "strength" in raw_type:
            if d in hevy_days:
                return "gym"
            if in_chile:
                return "home"
            return "functional"
        # fallback
        if in_chile and d not in hevy_days:
            return "home"
        return "functional"

    df["session_type"] = df.apply(classify_row, axis=1)

    return df


def build_daily_summary(activities: pd.DataFrame,
                        nutrition: pd.DataFrame,
                        settings: dict) -> pd.DataFrame:
    """
    Build one row per day:
      - calories_in, calories_out
      - training_minutes by type
      - simple readiness flag based on sleep proxy and load
    For MVP, sleep and HRV will be added later when we parse Garmin wellness exports.
    """

    if activities.empty and nutrition.empty:
        return pd.DataFrame()

    # calories out and training minutes
    if not activities.empty:
        agg = activities.groupby("date").agg(
            calories_out=("total_calories_burned", "sum"),
            training_minutes_total=("duration_minutes", "sum"),
        )
        # minutes per type
        per_type = activities.pivot_table(
            index="date",
            columns="session_type",
            values="duration_minutes",
            aggfunc="sum",
            fill_value=0.0,
        )
        per_type.columns = [f"training_minutes_{c}" for c in per_type.columns]
        daily = agg.join(per_type, how="left")
    else:
        daily = pd.DataFrame()

    # calories in
    if not nutrition.empty:
        nut = nutrition.groupby("date").agg(
            calories_in=("calories_in", "sum"),
            protein_g=("protein_g", "sum"),
            carbs_g=("carbs_g", "sum"),
            fat_g=("fat_g", "sum"),
        )
        daily = nut if daily.empty else daily.join(nut, how="outer")

    daily = daily.sort_index().reset_index()

    # location mode based on Chile trip
    trip_start = settings.get("trip_start_chile")
    trip_end = settings.get("trip_end_chile")

    def location_mode(d):
        if trip_start and trip_end:
            if trip_start <= d <= trip_end:
                return "Chile"
        return "Germany"

    daily["location_mode"] = daily["date"].apply(location_mode)

    # placeholder sleep and readiness
    # later, add sleep and HRV once Garmin wellness exports are parsed
    daily["sleep_hours"] = np.nan
    daily["readiness_flag"] = "unknown"

    return daily


# ------------- Data import page -------------

def data_import_page():
    st.header("Data import")

    st.markdown("Upload your latest exports from Garmin, Hevy and MyFitnessPal.")

    garmin_file = st.file_uploader("Garmin activities export (CSV)", type="csv")
    hevy_file = st.file_uploader("Hevy workouts export (CSV)", type="csv")
    mfp_file = st.file_uploader("MyFitnessPal nutrition export (CSV)", type="csv")

    if st.button("Process files"):
        activities_list = []

        if garmin_file is not None:
            garmin_df = parse_garmin_activities(garmin_file)
            activities_list.append(garmin_df)

        if hevy_file is not None:
            hevy_df = parse_hevy_workouts(hevy_file)
            activities_list.append(hevy_df)

        if activities_list:
            activities = pd.concat(activities_list, ignore_index=True)
        else:
            activities = pd.DataFrame()

        nutrition = pd.DataFrame()
        if mfp_file is not None:
            nutrition = parse_mfp_nutrition(mfp_file)

        # classify and build daily summary
        settings = st.session_state[SETTINGS_KEY]
        activities = classify_sessions(activities, settings)
        daily = build_daily_summary(activities, nutrition, settings)

        st.session_state[ACTIVITIES_KEY] = activities
        st.session_state[DAILY_SUMMARY_KEY] = daily

        st.success("Data processed and stored in session.")
        if not activities.empty:
            st.write("Sample activities:")
            st.dataframe(activities.head())
        if not daily.empty:
            st.write("Sample daily summary:")
            st.dataframe(daily.head())

    st.info("Later we can replace file uploads with API based connectors for Garmin and Hevy.")


# ------------- Daily hub page -------------

def daily_hub_page():
    st.header("Daily hub")

    daily = st.session_state[DAILY_SUMMARY_KEY]
    if daily.empty:
        st.warning("No daily summary yet. Please import data first.")
        return

    st.dataframe(daily)

    # simple filters
    with st.expander("Filter"):
        location = st.multiselect(
            "Location",
            options=daily["location_mode"].unique().tolist(),
            default=daily["location_mode"].unique().tolist(),
        )
        start_date = st.date_input(
            "Start date",
            value=daily["date"].min(),
        )
        end_date = st.date_input(
            "End date",
            value=daily["date"].max(),
        )

    mask = (
        daily["location_mode"].isin(location)
        & (daily["date"] >= start_date)
        & (daily["date"] <= end_date)
    )
    st.write("Filtered view:")
    st.dataframe(daily.loc[mask])


# ------------- Dashboard page -------------

def dashboard_page():
    st.header("Dashboard")

    daily = st.session_state[DAILY_SUMMARY_KEY]
    if daily.empty:
        st.warning("No data to show yet. Please import data first.")
        return

    # simple weekly summary for now
    st.subheader("Calories in vs out")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    import plotly.express as px

    melt_cols = []
    if "calories_in" in df.columns:
        melt_cols.append("calories_in")
    if "calories_out" in df.columns:
        melt_cols.append("calories_out")

    if melt_cols:
        m = df.melt(id_vars="date", value_vars=melt_cols,
                    var_name="type", value_name="calories")
        fig = px.line(m, x="date", y="calories", color="type")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training minutes by type")

    training_cols = [c for c in df.columns if c.startswith("training_minutes_")]
    if training_cols:
        tm = df[["date"] + training_cols].copy()
        tm = tm.melt(id_vars="date", value_vars=training_cols,
                     var_name="type", value_name="minutes")
        fig2 = px.bar(tm, x="date", y="minutes", color="type")
        st.plotly_chart(fig2, use_container_width=True)

    # very simple weekly text summary
    st.subheader("Weekly summary")

    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly = df.groupby("week").agg(
        avg_calories_in=("calories_in", "mean"),
        avg_calories_out=("calories_out", "mean"),
        sessions=("date", "count"),
    ).reset_index()

    st.dataframe(weekly)

    if not weekly.empty:
        last = weekly.iloc[-1]
        st.markdown(
            f"""
            - Week starting {last['week']}:  
              - average calories in: {last['avg_calories_in']:.0f}  
              - average calories out: {last['avg_calories_out']:.0f}  
              - days with data: {int(last['sessions'])}
            """
        )


# ------------- Placeholder for future API connectors -------------

def future_api_connectors_info():
    st.header("API connectors (future)")

    st.markdown("""
    This MVP uses CSV uploads.

    Later we can add:
    - A Hevy connector that uses the official Hevy public API with a token stored in `st.secrets`.
    - A Garmin connector using `python-garminconnect` or a Strava based bridge.
    - An optional MyFitnessPal connector using a community library if you are comfortable with that.

    The rest of the app logic will remain the same because data ingestion is separated from analysis.
    """)


# ------------- Main -------------

def main():
    st.set_page_config(page_title="Rodrigo Fitness Hub", layout="wide")
    init_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Settings", "Data import", "Daily hub", "Dashboard", "Future APIs"],
    )

    if page == "Settings":
        settings_page()
    elif page == "Data import":
        data_import_page()
    elif page == "Daily hub":
        daily_hub_page()
    elif page == "Dashboard":
        dashboard_page()
    else:
        future_api_connectors_info()


if __name__ == "__main__":
    main()