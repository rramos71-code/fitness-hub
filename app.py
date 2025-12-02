import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


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


# ------------- Hevy API integration -------------

def fetch_hevy_activities_from_api(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch workouts from the Hevy public API and normalize them into the
    activities schema used by the app:

      date, start_time, duration_minutes, total_calories_burned, raw_type, source

    Requires in .streamlit/secrets.toml:

      [hevy]
      api_key = "YOUR_HEVY_API_KEY"
    """
    if "hevy" not in st.secrets or "api_key" not in st.secrets["hevy"]:
        raise RuntimeError("Hevy API key not configured in .streamlit/secrets.toml")

    api_key = st.secrets["hevy"]["api_key"]

    # Base URL according to current Hevy public API docs
    base_url = "https://api.hevyapp.com/v1/workouts"

    headers = {
        "accept": "application/json",
        "api-key": api_key,
    }

    since_dt = datetime.utcnow() - timedelta(days=days_back)
    since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    all_workouts = []
    page = 1
    page_size = 10  # Hevy API restriction

    while True:
        params = {
            "page": page,
            "pageSize": page_size,
            "since": since_iso,
        }

        resp = requests.get(base_url, headers=headers, params=params, timeout=15)

        if resp.status_code != 200:
            # surface the first part of the response body to help debugging
            raise RuntimeError(
                f"Hevy API error {resp.status_code}: {resp.text[:400]}"
            )

        data = resp.json()
        workouts = data.get("workouts", [])

        if not workouts:
            break

        all_workouts.extend(workouts)

        if len(workouts) < page_size:
            break

        page += 1

    if not all_workouts:
        return pd.DataFrame(
            columns=[
                "date",
                "start_time",
                "duration_minutes",
                "total_calories_burned",
                "raw_type",
                "source",
            ]
        )

    rows = []
    for w in all_workouts:
        # Adjust keys here if Hevy changes the payload
        start = pd.to_datetime(w.get("start_time"), utc=True, errors="coerce")
        end = pd.to_datetime(w.get("end_time"), utc=True, errors="coerce")

        if pd.isna(start) or pd.isna(end):
            continue

        dur_min = (end - start).total_seconds() / 60.0

        rows.append(
            {
                "date": start.date(),
                "start_time": start,
                "duration_minutes": dur_min,
                "total_calories_burned": 0.0,   # we use Hevy mainly for strength details
                "raw_type": "strength_training",
                "source": "hevy",
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["start_time", "source"])
    return df


def merge_activities_into_session(new_acts: pd.DataFrame):
    """Merge newly fetched activities into the main activities + daily summary."""
    if new_acts is None or new_acts.empty:
        return

    settings = st.session_state[SETTINGS_KEY]
    new_acts = classify_sessions(new_acts, settings)

    existing = st.session_state.get(ACTIVITIES_KEY, pd.DataFrame())
    if existing is None or existing.empty:
        combined = new_acts.copy()
    else:
        combined = pd.concat([existing, new_acts], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "start_time", "raw_type", "source"],
            keep="last",
        )

    daily = build_daily_summary(combined, pd.DataFrame(), settings)

    st.session_state[ACTIVITIES_KEY] = combined
    st.session_state[DAILY_SUMMARY_KEY] = daily


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


# ------------- Classifier and daily summary -------------

def classify_sessions(activities: pd.DataFrame,
                      settings: dict) -> pd.DataFrame:
    """
    Simple rule based classifier:
      - source hevy -> gym
      - run / cycle / ride -> cardio (for future non-Hevy sources)
      - otherwise functional or home depending on Chile trip dates
    """

    if activities.empty:
        return activities

    df = activities.copy()
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
        if "run" in raw_type or "cycle" in raw_type or "ride" in raw_type:
            return "cardio"

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
      - location mode (Germany or Chile)
    """

    if activities.empty and nutrition.empty:
        return pd.DataFrame()

    if not activities.empty:
        agg = activities.groupby("date").agg(
            calories_out=("total_calories_burned", "sum"),
            training_minutes_total=("duration_minutes", "sum"),
        )
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

    if not nutrition.empty:
        nut = nutrition.groupby("date").agg(
            calories_in=("calories_in", "sum"),
            protein_g=("protein_g", "sum"),
            carbs_g=("carbs_g", "sum"),
            fat_g=("fat_g", "sum"),
        )
        daily = nut if daily.empty else daily.join(nut, how="outer")

    daily = daily.sort_index().reset_index()

    trip_start = settings.get("trip_start_chile")
    trip_end = settings.get("trip_end_chile")

    def location_mode(d):
        if trip_start and trip_end:
            if trip_start <= d <= trip_end:
                return "Chile"
        return "Germany"

    daily["location_mode"] = daily["date"].apply(location_mode)
    daily["sleep_hours"] = np.nan
    daily["readiness_flag"] = "unknown"

    return daily


# ------------- Hevy import page -------------

def hevy_import_page():
    st.header("Hevy activities (API)")

    days_back = st.number_input(
        "Days to sync from Hevy",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
    )

    if st.button("Sync from Hevy"):
        try:
            hevy_df = fetch_hevy_activities_from_api(days_back)
            merge_activities_into_session(hevy_df)

            st.success(f"Fetched {len(hevy_df)} workouts from Hevy.")
            if not hevy_df.empty:
                st.caption("Latest Hevy workouts (normalized)")
                st.dataframe(hevy_df.head())
        except Exception as e:
            st.error(f"Error fetching Hevy activities: {e}")


# ------------- Daily hub page -------------

def daily_hub_page():
    st.header("Daily hub")

    daily = st.session_state[DAILY_SUMMARY_KEY]
    if daily.empty:
        st.warning("No daily summary yet. Please sync from Hevy first.")
        return

    st.dataframe(daily)

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
        st.warning("No data to show yet. Please sync from Hevy first.")
        return

    st.subheader("Training minutes by type")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    import plotly.express as px

    training_cols = [c for c in df.columns if c.startswith("training_minutes_")]
    if training_cols:
        tm = df[["date"] + training_cols].copy()
        tm = tm.melt(
            id_vars="date",
            value_vars=training_cols,
            var_name="type",
            value_name="minutes",
        )
        fig2 = px.bar(tm, x="date", y="minutes", color="type")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Weekly summary")

    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly = df.groupby("week").agg(
        avg_calories_out=("calories_out", "mean"),
        sessions=("date", "count"),
    ).reset_index()

    st.dataframe(weekly)

    if not weekly.empty:
        last = weekly.iloc[-1]
        avg_out = last.get("avg_calories_out", np.nan)
        st.markdown(
            f"""
            - Week starting {last['week']}:  
              - average calories out: {avg_out:.0f}  
              - days with data: {int(last['sessions'])}
            """
        )


# ------------- Main -------------

def main():
    st.set_page_config(page_title="Rodrigo Fitness Hub", layout="wide")
    init_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Settings", "Hevy import", "Daily hub", "Dashboard"],
    )

    if page == "Settings":
        settings_page()
    elif page == "Hevy import":
        hevy_import_page()
    elif page == "Daily hub":
        daily_hub_page()
    elif page == "Dashboard":
        dashboard_page()


if __name__ == "__main__":
    main()
