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

st.header("Google Fit Connection")

try:
    google_client = GoogleFitClient()
    service = google_client.authorize()
except Exception as e:
    service = None
    st.error(str(e))

if service and st.button("Test Google Fit nutrition"):
    try:
        data = google_client.get_nutrition(service)
        st.success("Google Fit connection OK - nutrition dataset loaded")
        st.json(data)
    except Exception as e:
        st.error(str(e))