import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# ---- Local modules ----
from clients.hevy_client import HevyClient
from clients.garmin_client import GarminClient
from clients.googlefit_client import GoogleFitClient
from utils.daily_aggregation import build_daily_dataset

# =========================================================
# Client initialisation (safe, via session_state)
# =========================================================

def init_clients():
    # -------- Hevy --------
    if "hevy_client" not in st.session_state and "hevy_client_error" not in st.session_state:
        try:
            st.session_state["hevy_client"] = HevyClient()
        except Exception as e:
            st.session_state["hevy_client"] = None
            st.session_state["hevy_client_error"] = str(e)

    # -------- Garmin --------
    if "garmin_client" not in st.session_state and "garmin_client_error" not in st.session_state:
        try:
            st.session_state["garmin_client"] = GarminClient()
        except Exception as e:
            st.session_state["garmin_client"] = None
            st.session_state["garmin_client_error"] = str(e)

    # -------- Google Fit --------
    if "gf_client" not in st.session_state and "gf_client_error" not in st.session_state:
        try:
            st.session_state["gf_client"] = GoogleFitClient()
        except Exception as e:
            st.session_state["gf_client"] = None
            st.session_state["gf_client_error"] = str(e)


init_clients()

hevy_client = st.session_state.get("hevy_client")
garmin_client = st.session_state.get("garmin_client")
gf_client = st.session_state.get("gf_client")

hevy_err = st.session_state.get("hevy_client_error")
garmin_err = st.session_state.get("garmin_client_error")
gf_err = st.session_state.get("gf_client_error")

# =========================================================
# (Optional) helper if you still want raw → df conversion
# =========================================================

def aggregate_googlefit_macros(raw: dict, tz_name: str = "Europe/Berlin"):
    """
    Turn raw Google Fit aggregate JSON into a daily macros DataFrame.
    (not used by main flow now, but kept for debugging if needed)
    """
    tz = ZoneInfo(tz_name)
    per_day = {}  # date -> dict with totals

    for b in raw.get("bucket", []):
        for ds in b.get("dataset", []):
            for p in ds.get("point", []):
                end_ns = int(p.get("endTimeNanos", "0") or "0")
                if end_ns == 0:
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
        return pd.DataFrame(
            columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"]
        )

    rows = [{"date": d, **totals} for d, totals in per_day.items()]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# =========================================================
# UI
# =========================================================

st.title("Fitness Hub — Core Integrations Test")

# ----------------------- Hevy ----------------------------

st.header("Hevy Connection")

if hevy_err:
    st.warning(f"Hevy client not available: {hevy_err}")

if st.button("Sync Hevy workouts"):
    if hevy_client is None:
        st.error("Hevy client not initialised. Check your HEVY_API_KEY configuration.")
    else:
        try:
            workouts_df, sets_df = hevy_client.sync_workouts()
            st.session_state["hevy_sets_df"] = sets_df

            st.success(f"Hevy connection OK, {len(workouts_df)} workouts retrieved")
            st.subheader("Hevy sets (sample)")
            st.dataframe(sets_df.head())
        except Exception as e:
            st.error(f"Hevy error: {e}")

# ----------------------- Garmin --------------------------

st.header("Garmin Connection")

if garmin_err:
    st.warning(f"Garmin client not available: {garmin_err}")

if st.button("Test Garmin"):
    if garmin_client is None:
        st.error("Garmin client not initialised. Check GARMIN_EMAIL / GARMIN_PASSWORD.")
    else:
        try:
            daily_df, activities_df = garmin_client.fetch_daily_and_activities()
            st.session_state["garmin_daily_df"] = daily_df
            st.session_state["garmin_activities_df"] = activities_df

            st.write("Daily")
            st.dataframe(daily_df.head())

            st.write("Activities")
            st.dataframe(activities_df.head())
        except Exception as e:
            st.error(f"Garmin error: {e}")

# ----------------------- Google Fit ----------------------

st.header("Google Fit Connection")

if gf_err:
    st.warning(f"Google Fit client not available: {gf_err}")

days_back = st.number_input(
    "Days back", min_value=1, max_value=30, value=7, step=1
)

if st.button("Test Google Fit nutrition"):
    if gf_client is None:
        st.error("Google Fit client not initialised. Check your Google Fit secrets.")
    else:
        try:
            df = gf_client.aggregate_daily_macros(days_back=days_back)
            st.session_state["googlefit_nutrition_df"] = df

            if df.empty:
                st.info("Google Fit returned no nutrition entries for the selected period.")
            else:
                st.dataframe(df)
        except Exception as e:
            st.error(f"Google Fit error: {e}")

if st.button("Debug raw Google Fit aggregate response"):
    if gf_client is None:
        st.error("Google Fit client not initialised. Check your Google Fit secrets.")
    else:
        try:
            raw = gf_client.debug_aggregate_raw(days_back=days_back)
            st.json(raw)
        except Exception as e:
            st.error(f"Google Fit debug error: {e}")

# -------------------- Unified daily view -----------------

def _ensure_date_column(df, candidates):
    """
    Make sure the dataframe has a 'date' column.

    - If 'date' already exists, leave it.
    - Otherwise, try each candidate column name, convert to datetime,
      and take the .date() part.
    - If nothing works, return an empty DataFrame to avoid KeyError
      in the daily aggregation.
    """
    if df is None or getattr(df, "empty", True):
        return df

    if "date" in df.columns:
        # normalise to date only, just in case it’s datetime
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    for col in candidates:
        if col in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df[col]).dt.date
            return df

    # No usable date column found; better to drop this df than to crash
    return pd.DataFrame()

# =================== Unified daily view ==================
st.subheader("Daily overview (unified dataset)")

nutrition_df = st.session_state.get("googlefit_nutrition_df")
garmin_daily_df = st.session_state.get("garmin_daily_df")
garmin_activities_df = st.session_state.get("garmin_activities_df")
hevy_sets_df = st.session_state.get("hevy_sets_df")

# Standardise 'date' column for all sources
nutrition_df = _ensure_date_column(nutrition_df, ["date"])
garmin_daily_df = _ensure_date_column(garmin_daily_df, ["calendarDate"])
garmin_activities_df = _ensure_date_column(
    garmin_activities_df,
    ["startTimeLocal", "startTimeGMT", "start_time"],
)
hevy_sets_df = _ensure_date_column(
    hevy_sets_df,
    ["performed_at", "start_time"],
)

if (
    (nutrition_df is None or nutrition_df.empty)
    and (garmin_daily_df is None or garmin_daily_df.empty)
    and (garmin_activities_df is None or garmin_activities_df.empty)
    and (hevy_sets_df is None or hevy_sets_df.empty)
):
    st.info(
        "Load data from at least one source (Google Fit, Garmin, Hevy) "
        "to see the unified daily dataset."
    )
else:
    try:
        daily_df = build_daily_dataset(
            nutrition_df=nutrition_df,
            garmin_daily_df=garmin_daily_df,
            garmin_activities_df=garmin_activities_df,
            hevy_sets_df=hevy_sets_df,
        )
        st.dataframe(daily_df)

        st.caption("Rows per source:")
        st.write(
            {
                "nutrition_rows": 0 if nutrition_df is None else len(nutrition_df),
                "garmin_daily_rows": 0 if garmin_daily_df is None else len(garmin_daily_df),
                "garmin_activities_rows": 0
                if garmin_activities_df is None
                else len(garmin_activities_df),
                "hevy_sets_rows": 0 if hevy_sets_df is None else len(hevy_sets_df),
                "daily_rows": len(daily_df),
            }
        )
    except Exception as e:
        st.error(f"Daily aggregation error: {e}")
