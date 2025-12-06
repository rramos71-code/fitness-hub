from datetime import date
import os

import streamlit as st
from garminconnect import Garmin, GarminConnectAuthenticationError


class GarminClient:
    def __init__(self):
        email = None
        password = None

        # 1. Try top level secrets
        try:
            if "GARMIN_EMAIL" in st.secrets:
                email = st.secrets["GARMIN_EMAIL"]
            if "GARMIN_PASSWORD" in st.secrets:
                password = st.secrets["GARMIN_PASSWORD"]
        except Exception:
            pass

        # 2. Try nested section [fitness_hub]
        if not email or not password:
            try:
                fh = st.secrets.get("fitness_hub", {})
                email = email or fh.get("GARMIN_EMAIL")
                password = password or fh.get("GARMIN_PASSWORD")
            except Exception:
                pass

        # 3. Fallback to environment variables
        email = email or os.getenv("GARMIN_EMAIL")
        password = password or os.getenv("GARMIN_PASSWORD")

        if not email or not password:
            raise RuntimeError(
                "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD in Streamlit secrets."
            )

        self.email = email
        self.password = password

    def _login(self) -> Garmin:
        client = Garmin(self.email, self.password)
        client.login()
        return client

    def get_today_summary(self):
        """Convenience method: summary for today."""
        return self.get_daily_summary(date.today())

    def get_daily_summary(self, day: date):
        """This mirrors the name you used in the old app."""
        client = self._login()
        try:
            stats = client.get_stats(day)
        finally:
            try:
                client.logout()
            except Exception:
                pass
        return stats
