import os
import streamlit as st

from hevy_api.client import HevyClient
from garminconnect import Garmin, GarminConnectAuthenticationError


# ---------- Hevy helpers ----------

def get_hevy_client() -> HevyClient:
    """
    Create a HevyClient using HEVY_API_KEY
    from Streamlit secrets or environment variables.
    """
    api_key = None

    # 1. Streamlit secrets
    if "HEVY_API_KEY" in st.secrets:
        api_key = st.secrets["HEVY_API_KEY"]

    # 2. Fallback to environment variable
    if not api_key:
        api_key = os.getenv("HEVY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Hevy API key not found. Set HEVY_API_KEY in Streamlit secrets "
            "or as an environment variable."
        )

    os.environ["HEVY_API_KEY"] = api_key
    return HevyClient()


def sync_hevy():
    client = get_hevy_client()
    response = client.get_workouts()
    workouts = response.workouts or []

    st.success(f"Hevy: retrieved {len(workouts)} workouts.")

    if workouts:
        last = workouts[-1]
        st.subheader("Hevy - most recent workout")
        if hasattr(last, "model_dump"):
            st.json(last.model_dump())
        elif hasattr(last, "dict"):
            st.json(last.dict())
        else:
            st.write(last)


# ---------- Garmin helpers ----------

def get_garmin_client() -> Garmin:
    """
    Create a Garmin client using credentials from secrets.
    """
    email = st.secrets.get("GARMIN_EMAIL")
    password = st.secrets.get("GARMIN_PASSWORD")

    if not email or not password:
        raise RuntimeError(
            "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD in Streamlit secrets."
        )

    client = Garmin(email, password)
    return client


def sync_garmin():
    try:
        client = get_garmin_client()
        client.login()

        # Get first 10 activities (index, limit)
        activities = client.get_activities(0, 10)  # returns list of dicts

        st.success(f"Garmin: retrieved {len(activities)} activities.")

        if activities:
            last = activities[0]  # newest first
            st.subheader("Garmin - most recent activity")
            st.json(last)

    except GarminConnectAuthenticationError:
        st.error("Garmin authentication failed. Check email or password in secrets.")
    except Exception as e:
        st.error(f"Error communicating with Garmin: {e}")


# ---------- Main app ----------

def main():
    st.set_page_config(
        page_title="Fitness Hub - Hevy + Garmin",
        page_icon="💪",
        layout="centered",
    )

    st.title("Fitness Hub - Hevy + Garmin connection")

    st.write(
        "The app uses secrets for both Hevy and Garmin. "
        "Use the buttons below to test each connection and fetch a small preview."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sync Hevy workouts"):
            with st.spinner("Syncing from Hevy"):
                sync_hevy()

    with col2:
        if st.button("Sync Garmin activities"):
            with st.spinner("Syncing from Garmin"):
                sync_garmin()


if __name__ == "__main__":
    main()
