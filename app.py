import streamlit as st
from datetime import date

from clients.hevy_client import HevyClient
from clients.garmin_client import GarminClient
from clients.googlefit_client import GoogleFitClient

st.title("Fitness Hub — Core Integrations Test")

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

col1, col2 = st.columns(2)
with col1:
    days_back = st.number_input("Days back", min_value=1, max_value=90, value=7, step=1)

with col2:
    if st.button("Test Google Fit nutrition"):
        try:
            gf = GoogleFitClient()
            daily_macros = gf.get_daily_macros(days_back=days_back)

            if daily_macros.empty:
                st.info("Google Fit returned no nutrition entries for the selected period.")
            else:
                st.success("Google Fit connection OK - daily calories and macros aggregated")
                st.dataframe(daily_macros)
        except Exception as e:
            st.error(str(e))

# Debug button
if st.button("Debug raw Google Fit aggregate response"):
    try:
        gf = GoogleFitClient()
        raw = gf.debug_aggregate_raw(days_back=7)
        st.json(raw)
    except Exception as e:
        st.error(str(e))

