import streamlit as st
from datetime import date

from clients.hevy_client import HevyClient
from clients.garmin_client import GarminClient
from clients.googlefit_client import GoogleFitClient

from utils.daily_aggregation import build_daily_dataset

from zoneinfo import ZoneInfo
from datetime import datetime, timezone

st.title("Fitness Hub — Core Integrations Test")

def aggregate_googlefit_macros(raw: dict, tz_name: str = "Europe/Berlin"):
    """
    Turn raw Google Fit aggregate JSON into a daily macros DataFrame.

    Logic:
    - Iterate over all buckets and all points.
    - Use endTimeNanos to decide which local calendar date a point belongs to.
    - Aggregate calories, protein, carbs and fat per date.
    - No filtering by app (LoseIt, Cronometer, etc) so any app that writes
      com.google.nutrition.summary is included.
    """
    import pandas as pd

    tz = ZoneInfo(tz_name)
    per_day = {}  # date -> dict with totals

    for b in raw.get("bucket", []):
        for ds in b.get("dataset", []):
            for p in ds.get("point", []):
                origin = p.get("originDataSourceId", "")

                # (we no longer filter on origin, so Cronometer, LoseIt, manual Fit input etc are all included)

                # Decide which day this point belongs to
                end_ns = int(p.get("endTimeNanos", "0") or "0")
                if end_ns == 0:
                    # Fallback: bucket start if endTimeNanos is missing
                    start_ms = int(b.get("startTimeMillis", "0") or "0")
                    if start_ms == 0:
                        continue
                    dt_local = datetime.fromtimestamp(start_ms / 1000.0, tz=tz)
                else:
                    dt_utc = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
                    dt_local = dt_utc.astimezone(tz)

                day = dt_local.date()

                if day not in per_day:
                    per_day[day] = {
                        "calories_kcal": 0.0,
                        "protein_g": 0.0,
                        "carbs_g": 0.0,
                        "fat_g": 0.0,
                    }

                # Sum macro values
                for val in p.get("value", []):
                    for mv in val.get("mapVal", []):
                        key = mv.get("key")
                        v = mv.get("value", {})
                        raw_value = v.get("fpVal") if "fpVal" in v else v.get("intVal")
                        if raw_value is None:
                            continue
                        value = float(raw_value)

                        if key in ("calories", "nutrition.calories"):
                            per_day[day]["calories_kcal"] += value
                        elif key in ("protein", "nutrition.protein"):
                            per_day[day]["protein_g"] += value
                        elif key in ("carbs.total", "nutrition.carbs.total"):
                            per_day[day]["carbs_g"] += value
                        elif key in ("fat.total", "nutrition.fat.total"):
                            per_day[day]["fat_g"] += value

    if not per_day:
        return pd.DataFrame(columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"])

    rows = []
    for day, totals in per_day.items():
        rows.append({"date": day, **totals})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

# ---------------- Hevy Connect ----------------

st.header("Hevy Connection")
# After Hevy sync
if st.button("Sync Hevy workouts"):
    workouts_df, sets_df = hevy_client.sync_workouts()
    st.session_state["hevy_sets_df"] = sets_df
    st.dataframe(sets_df.head())
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
# After Garmin sync
if st.button("Test Garmin"):
    daily_df, activities_df = garmin_client.fetch_daily_and_activities()
    st.session_state["garmin_daily_df"] = daily_df
    st.session_state["garmin_activities_df"] = activities_df
    st.write("Daily")
    st.dataframe(daily_df.head())
    st.write("Activities")
    st.dataframe(activities_df.head())
# ---------------- Google Fit ----------------

from clients.googlefit_client import GoogleFitClient

st.header("Google Fit Connection")

# inside your Streamlit layout
days_back = st.number_input("Days back", min_value=1, max_value=30, value=7, step=1)

# After Google Fit sync
if st.button("Test Google Fit nutrition"):
    df = gf_client.aggregate_daily_macros(days_back=st.session_state.get("gf_days_back", 7))
    st.session_state["googlefit_nutrition_df"] = df
    st.dataframe(df)

# Debug button
if st.button("Debug raw Google Fit aggregate response"):
    client = GoogleFitClient()
    raw = client.debug_aggregate_raw(days_back=days_back)
    st.json(raw)

# ... inside main() after other tabs

daily_tab = st.sidebar.checkbox("Show daily dataset", value=True)  # or use st.tabs if you prefer

if daily_tab:
    st.subheader("Daily overview")

    nutrition_df = st.session_state.get("googlefit_nutrition_df")
    garmin_daily_df = st.session_state.get("garmin_daily_df")
    garmin_activities_df = st.session_state.get("garmin_activities_df")
    hevy_sets_df = st.session_state.get("hevy_sets_df")

    if (
        (nutrition_df is None or nutrition_df.empty)
        and (garmin_daily_df is None or garmin_daily_df.empty)
        and (garmin_activities_df is None or garmin_activities_df.empty)
        and (hevy_sets_df is None or hevy_sets_df.empty)
    ):
        st.info("Load data from at least one source (Google Fit, Garmin, Hevy) to see the unified daily dataset.")
    else:
        daily_df = build_daily_dataset(
            nutrition_df=nutrition_df,
            garmin_daily_df=garmin_daily_df,
            garmin_activities_df=garmin_activities_df,
            hevy_sets_df=hevy_sets_df,
        )
        st.dataframe(daily_df)

        # Quick sanity checks
        st.caption("Rows per source:")
        st.write(
            {
                "nutrition_rows": 0 if nutrition_df is None else len(nutrition_df),
                "garmin_daily_rows": 0 if garmin_daily_df is None else len(garmin_daily_df),
                "garmin_activities_rows": 0 if garmin_activities_df is None else len(garmin_activities_df),
                "hevy_sets_rows": 0 if hevy_sets_df is None else len(hevy_sets_df),
                "daily_rows": len(daily_df),
            }
        )
