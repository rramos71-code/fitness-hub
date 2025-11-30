import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime, date, timedelta
from data_model import aggregate_hevy_daily


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

def get_strava_secrets():
    """
    Read Strava secrets from st.secrets.

    Expected structure in .streamlit/secrets.toml:

      [strava]
      client_id = "187872"
      client_secret = "..."
      access_token = "..."
    """
    try:
        section = st.secrets["strava"]
    except Exception:
        raise ValueError(
            f"Strava block [strava] not found in secrets. "
            f"Available sections: {list(st.secrets.keys())}"
        )

    required = ["client_id", "client_secret", "access_token"]
    missing = [k for k in required if k not in section]

    if missing:
        raise ValueError(
            f"Strava secrets missing keys: {missing}. "
            f"Present keys: {list(section.keys())}"
        )

    client_id = str(section["client_id"])
    client_secret = section["client_secret"]
    access_token = section["access_token"]

    return client_id, client_secret, access_token

def fetch_hevy_activities_from_api(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch workouts from the Hevy public API and normalize them into the
    activities schema used by the app:

      date, start_time, duration_minutes, total_calories_burned, raw_type, source
    """
    # Check secrets
    if "hevy" not in st.secrets or "api_key" not in st.secrets["hevy"]:
        raise RuntimeError("Hevy API key not configured in .streamlit/secrets.toml")

    api_key = st.secrets["hevy"]["api_key"]

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
            raise RuntimeError(
                f"Hevy API error {resp.status_code}: {resp.text[:400]}"
            )

        data = resp.json()

        # Hevy returns workouts under "workouts"
        workouts = data.get("workouts", [])

        if not workouts:
            break

        all_workouts.extend(workouts)

        # stop if we got fewer than page_size, so we are at the last page
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
        # adjust keys if Hevy uses different field names
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
                "total_calories_burned": 0.0,
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

    # classify sessions using existing rules
    settings = st.session_state[SETTINGS_KEY]
    new_acts = classify_sessions(new_acts, settings)

    # existing activities in session
    existing = st.session_state.get(ACTIVITIES_KEY, pd.DataFrame())
    if existing is None or existing.empty:
        combined = new_acts.copy()
    else:
        combined = pd.concat([existing, new_acts], ignore_index=True)

        # de-duplicate on basic identity keys
        combined = combined.drop_duplicates(
            subset=["date", "start_time", "raw_type", "source"],
            keep="last",
        )

    # rebuild daily summary (no nutrition yet, so pass empty df for now)
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


# ------------- Data import helpers -------------

def fetch_strava_activities(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch activities from Strava API.

    Output columns:
      date, start_time, duration_minutes, total_calories_burned, raw_type, source
    """
    if "strava" not in st.secrets:
        raise ValueError("Strava secrets not configured in .streamlit/secrets.toml")

    client_id = st.secrets["strava"].get("client_id")
    access_token = st.secrets["strava"].get("access_token")

    if not access_token:
        raise ValueError("Strava access_token missing in secrets")

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    after_dt = datetime.utcnow() - timedelta(days=days_back)
    after_ts = int(after_dt.timestamp())

    all_acts = []
    page = 1
    per_page = 200

    while True:
        params = {
            "after": after_ts,
            "page": page,
            "per_page": per_page,
        }
        resp = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params=params,
            timeout=20,
        )
        if resp.status_code != 200:
            raise ValueError(
                f"Strava API error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        if not data:
            break

        all_acts.extend(data)
        if len(data) < per_page:
            break
        page += 1
        # small pause to be gentle with the API
        time.sleep(0.1)

    if not all_acts:
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

    df = pd.json_normalize(all_acts)

    # start time and date
    if "start_date_local" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_date_local"], errors="coerce")
    else:
        df["start_time"] = pd.to_datetime(df.get("start_date"), errors="coerce")

    df["date"] = df["start_time"].dt.date

    # duration in minutes
    if "moving_time" in df.columns:
        df["duration_minutes"] = df["moving_time"] / 60.0
    elif "elapsed_time" in df.columns:
        df["duration_minutes"] = df["elapsed_time"] / 60.0
    else:
        df["duration_minutes"] = np.nan

    # calories if available
    if "calories" in df.columns:
        df["total_calories_burned"] = df["calories"]
    else:
        df["total_calories_burned"] = 0.0

    # activity type
    if "sport_type" in df.columns:
        df["raw_type"] = df["sport_type"].astype(str)
    elif "type" in df.columns:
        df["raw_type"] = df["type"].astype(str)
    else:
        df["raw_type"] = "unknown"

    df["source"] = "strava"

    return df[[
        "date",
        "start_time",
        "duration_minutes",
        "total_calories_burned",
        "raw_type",
        "source",
    ]]


def parse_hevy_workouts(file) -> pd.DataFrame:
    """
    Parse Hevy CSV export per workout session.

    Input format (per set):
      title, start_time, end_time, description, exercise_title,
      superset_id, exercise_notes, set_index, set_type,
      weight_kg, reps, distance_km, duration_seconds, rpe
    """

    df = pd.read_csv(file)

    df["start_dt"] = pd.to_datetime(df["start_time"], format="%d %b %Y, %H:%M")
    df["end_dt"] = pd.to_datetime(df["end_time"], format="%d %b %Y, %H:%M")

    df["date"] = df["start_dt"].dt.date
    df["duration_minutes"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60.0

    grouped = (
        df.groupby(["title", "start_dt", "end_dt"], as_index=False)
        .agg(
            date=("date", "first"),
            duration_minutes=("duration_minutes", "max"),
        )
    )

    grouped = grouped.rename(columns={"start_dt": "start_time"})
    grouped["total_calories_burned"] = 0.0
    grouped["raw_type"] = "strength"
    grouped["source"] = "hevy"

    return grouped[[
        "date",
        "start_time",
        "duration_minutes",
        "total_calories_burned",
        "raw_type",
        "source",
    ]]


# placeholder that we will replace with FatSecret later
def parse_mfp_nutrition(file) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=["date", "calories_in", "protein_g", "carbs_g", "fat_g"]
    )
    return df


# ------------- Classifier and daily summary -------------

def classify_sessions(activities: pd.DataFrame,
                      settings: dict) -> pd.DataFrame:
    """
    Simple rule based classifier:
      - source hevy -> gym
      - run or cycling -> cardio
      - strava or garmin strength type -> functional or gym
      - during Chile trip with no hevy -> home
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
        if src in ("strava", "garmin") and (
            "strength" in raw_type or "weight" in raw_type or "workout" in raw_type
        ):
            if d in hevy_days:
                return "gym"
            if in_chile:
                return "home"
            return "functional"

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


# ------------- Data import page -------------

def data_import_page():
    st.header("Data import")

    st.markdown("Upload exports and/or sync via APIs.")

    # ---------------- Strava (already implemented) ----------------
    st.subheader("Strava activities (API)")
    days_strava = st.number_input(
        "Days to sync from Strava",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="days_strava_api",
    )

    strava_activities = pd.DataFrame()
    if st.button("Sync from Strava"):
        from strava_api import fetch_strava_activities  # wherever you put it

        try:
            strava_activities = fetch_strava_activities(days_back=days_strava)
            st.success(f"Fetched {len(strava_activities)} activities from Strava.")
            st.dataframe(strava_activities.head())
        except Exception as e:
            st.error(f"Error fetching Strava activities: {e}")

    st.markdown("---")

    # ---------------- Hevy via API ----------------
    st.subheader("Hevy activities (API)")
    days_hevy = st.number_input(
        "Days to sync from Hevy",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="days_hevy_api",
    )

    hevy_activities_api = pd.DataFrame()
    if st.button("Sync from Hevy"):
        try:
            hevy_activities_api = fetch_hevy_activities_from_api(days_back=days_hevy)
            st.success(f"Fetched {len(hevy_activities_api)} workouts from Hevy.")
            if not hevy_activities_api.empty:
                st.dataframe(hevy_activities_api.head())
        except Exception as e:
            st.error(f"Error fetching Hevy activities: {e}")

    st.markdown("---")

    # ---------------- Optional CSV uploads (Garmin + Hevy CSV) ----------------
    st.subheader("CSV uploads (optional)")

    st.markdown("Garmin activities export (CSV or summarizedActivities.json)")
    garmin_file = st.file_uploader(
        "Garmin activities file",
        type=["csv", "json"],
        key="garmin_file_uploader",
    )

#    hevy_file = st.file_uploader(
#        "Hevy workouts export (CSV)",
#        type="csv",
#        key="hevy_csv_file_uploader",
#    )

    mfp_file = None  # we skip MyFitnessPal for now

    if st.button("Process CSV files"):
        activities_list = []

        if garmin_file is not None:
            garmin_df = parse_garmin_activities(garmin_file)
            activities_list.append(garmin_df)

#        if hevy_file is not None:
#            hevy_df_csv = parse_hevy_workouts(hevy_file)
#            activities_list.append(hevy_df_csv)

        # combine CSV-based activities
        csv_activities = (
            pd.concat(activities_list, ignore_index=True) if activities_list else pd.DataFrame()
        )

        if not csv_activities.empty:
            st.success(f"Processed {len(csv_activities)} activities from CSV.")
            st.dataframe(csv_activities.head())
        else:
            st.info("No CSV activities processed.")

        # you can also build nutrition here later when FatSecret is wired

        # merge all sources for the main app state
        all_activities = []

        if not strava_activities.empty:
            all_activities.append(strava_activities)
        if not hevy_activities_api.empty:
            all_activities.append(hevy_activities_api)
        if not csv_activities.empty:
            all_activities.append(csv_activities)

        if all_activities:
            activities = pd.concat(all_activities, ignore_index=True)
        else:
            activities = pd.DataFrame()

        # at this point nutrition is still empty
        nutrition = pd.DataFrame()

        settings = st.session_state[SETTINGS_KEY]
        activities = classify_sessions(activities, settings)
        daily = build_daily_summary(activities, nutrition, settings)

        st.session_state[ACTIVITIES_KEY] = activities
        st.session_state[DAILY_SUMMARY_KEY] = daily

        st.success("All data processed and stored in session.")
        if not activities.empty:
            st.write("Sample activities:")
            st.dataframe(activities.head())
        if not daily.empty:
            st.write("Sample daily summary:")
            st.dataframe(daily.head())


# ------------- Daily hub page -------------

def daily_hub_page():
    st.header("Daily hub")

    daily = st.session_state[DAILY_SUMMARY_KEY]
    if daily.empty:
        st.warning("No daily summary yet. Please import data first.")
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
        st.warning("No data to show yet. Please import data first.")
        return

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
        m = df.melt(
            id_vars="date",
            value_vars=melt_cols,
            var_name="type",
            value_name="calories",
        )
        fig = px.line(m, x="date", y="calories", color="type")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training minutes by type")

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
        avg_calories_in=("calories_in", "mean")
        if "calories_in" in df.columns
        else ("calories_out", "mean"),
        avg_calories_out=("calories_out", "mean"),
        sessions=("date", "count"),
    ).reset_index()

    st.dataframe(weekly)

    if not weekly.empty:
        last = weekly.iloc[-1]
        avg_in = last.get("avg_calories_in", np.nan)
        avg_out = last.get("avg_calories_out", np.nan)
        st.markdown(
            f"""
            - Week starting {last['week']}:  
              - average calories in: {avg_in:.0f}  
              - average calories out: {avg_out:.0f}  
              - days with data: {int(last['sessions'])}
            """
        )



# ------------- Hevy API page -------------

def hevy_api_page():
    st.header("Hevy activities (API)")

    days_back = st.number_input(
        "Days to sync from Hevy",
        min_value=1,
        max_value=90,
        value=30,
        step=1,
    )

    if st.button("Sync from Hevy"):
        try:
            # 1) Pull from Hevy API
            hevy_df = fetch_hevy_activities_from_api(days_back)

            # 2) Classify sessions (gym / functional / etc.)
            settings = st.session_state[SETTINGS_KEY]
            hevy_df = classify_sessions(hevy_df, settings)

            # 3) Merge with any existing activities (e.g. Strava or imports)
            current = st.session_state.get(ACTIVITIES_KEY, pd.DataFrame())
            if current is not None and not current.empty:
                all_activities = pd.concat([current, hevy_df], ignore_index=True)
                # avoid duplicates if you sync multiple times
                all_activities = all_activities.drop_duplicates(
                    subset=["start_time", "source"]
                )
            else:
                all_activities = hevy_df

            st.session_state[ACTIVITIES_KEY] = all_activities

            # 4) Rebuild daily summary (nutrition empty for now, FatSecret later)
            nutrition = pd.DataFrame()
            daily = build_daily_summary(all_activities, nutrition, settings)
            st.session_state[DAILY_SUMMARY_KEY] = daily

            # 5) Show feedback
            st.success(f"Fetched {len(hevy_df)} workouts from Hevy.")
            st.caption("Latest Hevy workouts (normalized)")
            st.dataframe(hevy_df.head())

        except Exception as e:
            st.error(f"Error fetching Hevy activities: {e}")
            st.exception(e)


# ------------- Future API info page -------------

def future_api_connectors_info():
    st.header("API connectors (future)")

    st.markdown(
        """
        This MVP uses:
        - Strava API for activities
        - Hevy CSV for strength session details
        - Nutrition layer will be wired to FatSecret API next

        Because ingestion is separated from analysis we can swap sources later
        without changing the dashboards.
        """
    )


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
