import streamlit as st
from datetime import date

from clients.hevy_client import HevyClient
from clients.garmin_client import GarminClient
from clients.googlefit_client import GoogleFitClient

from googlefit_client import GoogleFitClient

st.title("Fitness Hub — Core Integrations Test")

from zoneinfo import ZoneInfo
from datetime import datetime

def aggregate_googlefit_macros(raw: dict, tz_name: str = "Europe/Berlin"):
    """
    Turn the raw Google Fit aggregate JSON into a daily macros DataFrame.

    - Uses the bucket start time as the calendar date
    - Only counts nutrition coming from Lose It (originDataSourceId contains 'com.fitnow.loseit')
    """
    import pandas as pd

    tz = ZoneInfo(tz_name)
    rows = []

    for b in raw.get("bucket", []):
        # bucket start -> local date
        start_ms = int(b.get("startTimeMillis", "0") or "0")
        if start_ms == 0:
            continue

        day_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=tz)
        day = day_dt.date()

        calories = protein = carbs = fat = 0.0

        for ds in b.get("dataset", []):
            for p in ds.get("point", []):
                origin = p.get("originDataSourceId", "")

                # only Lose It entries
                if "com.fitnow.loseit" not in origin:
                    continue

                for val in p.get("value", []):
                    for mv in val.get("mapVal", []):
                        key = mv.get("key")
                        v = mv.get("value", {})
                        raw_value = v.get("fpVal") if "fpVal" in v else v.get("intVal")
                        if raw_value is None:
                            continue
                        value = float(raw_value)

                        if key in ("calories", "nutrition.calories"):
                            calories += value
                        elif key in ("protein", "nutrition.protein"):
                            protein += value
                        elif key in ("carbs.total", "nutrition.carbs.total"):
                            carbs += value
                        elif key in ("fat.total", "nutrition.fat.total"):
                            fat += value

        if any(x > 0 for x in (calories, protein, carbs, fat)):
            rows.append(
                {
                    "date": day,
                    "calories_kcal": calories,
                    "protein_g": protein,
                    "carbs_g": carbs,
                    "fat_g": fat,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"])

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ---------------- Hevy Connect ----------------

st.header("Hevy Connection")
if st.button("Test Hevy"):
    try:
        client = HevyClient()
        workouts = client.get_workouts()
        st.success(f"Hevy connection OK, {len(workouts)} workouts retrieved")
        # Show a small sample so the response is not huge
        st.json(workouts[:2])
    except Exception as e:
        st.error(str(e))

# ---------------- Garmin Connect ----------------

st.header("Garmin Connection")
if st.button("Test Garmin"):
    try:
        client = GarminClient()
        data = client.get_daily_summary(date.today())
        st.success("Garmin connection OK")
        st.json(data)
    except Exception as e:
        st.error(str(e))

# ---------------- Google Fit ----------------

from clients.googlefit_client import GoogleFitClient

st.header("Google Fit Connection")

from google_fit_client import GoogleFitClient

# inside your Streamlit layout
days_back = st.number_input("Days back", min_value=1, max_value=30, value=7, step=1)

if st.button("Test Google Fit nutrition"):
    try:
        client = GoogleFitClient()
        # use the same aggregate call as the debug button
        raw = client.debug_aggregate_raw(days_back=days_back)
        df = aggregate_googlefit_macros(raw)

        if df.empty:
            st.info("Google Fit returned no nutrition entries for the selected period.")
        else:
            st.success("Google Fit connection OK - daily calories and macros aggregated")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading Google Fit nutrition: {e}")


# Debug button
if st.button("Debug raw Google Fit aggregate response"):
    client = GoogleFitClient()
    raw = client.debug_aggregate_raw(days_back=days_back)
    st.json(raw)


