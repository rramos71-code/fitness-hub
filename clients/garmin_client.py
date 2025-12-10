from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from garminconnect import Garmin, GarminConnectAuthenticationError


@dataclass
class GarminCredentials:
    email: str
    password: str


class GarminClient:
    """
    Thin wrapper around garminconnect.Garmin that:

    - Reads credentials from Streamlit secrets or env variables
    - Provides a helper to fetch daily summaries + activities
    - Parses calories / steps / distance out of the raw stats dict
    """

    def __init__(self):
        self.creds = self._load_credentials()

    # ---------- internal helpers ----------

    def _load_credentials(self) -> GarminCredentials:
        email = None
        password = None

        # 1) top-level secrets
        try:
            if "GARMIN_EMAIL" in st.secrets:
                email = st.secrets["GARMIN_EMAIL"]
            if "GARMIN_PASSWORD" in st.secrets:
                password = st.secrets["GARMIN_PASSWORD"]
        except Exception:
            pass

        # 2) nested section [fitness_hub]
        if not email or not password:
            try:
                fh = st.secrets.get("fitness_hub", {})
                email = email or fh.get("GARMIN_EMAIL")
                password = password or fh.get("GARMIN_PASSWORD")
            except Exception:
                pass

        # 3) env variables
        email = email or os.getenv("GARMIN_EMAIL")
        password = password or os.getenv("GARMIN_PASSWORD")

        if not email or not password:
            raise RuntimeError(
                "Garmin credentials not found. "
                "Set GARMIN_EMAIL and GARMIN_PASSWORD in Streamlit secrets or env."
            )

        return GarminCredentials(email=email, password=password)

    def _login(self) -> Garmin:
        client = Garmin(self.creds.email, self.creds.password)
        client.login()
        return client

    # ---------- public methods ----------

    def get_daily_summary(self, day: date) -> dict:
        """One-off helper if you ever want a single day."""
        client = self._login()
        try:
            return client.get_stats(day.isoformat()) or {}
        finally:
            try:
                client.logout()
            except Exception:
                pass

    def fetch_daily_and_activities(self, days_back: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point used by the app.

        Returns:
            daily_df:    columns = [date, steps, calories, distance_km, raw_stats]
            activities_df: one row per recorded activity
        """
        client = self._login()
        try:
            today = date.today()
            start = today - timedelta(days=days_back - 1)

            # -------- daily summaries --------
            daily_rows = []
            for i in range(days_back):
                d = start + timedelta(days=i)
                d_str = d.isoformat()

                stats = client.get_stats(d_str) or {}
                raw_stats = json.dumps(stats)  # keep full blob for debugging

                # Garmin’s keys can vary a bit between accounts, so we check a few options
                def _get_number(keys, default=None):
                    for k in keys:
                        if k in stats and stats[k] not in (None, "", "null"):
                            try:
                                return float(stats[k])
                            except Exception:
                                pass
                    return default

                calories = _get_number(["totalKilocalories", "activeKilocalories"])
                steps = _get_number(["steps", "totalSteps", "stepCount"])
                # distance often comes in meters
                distance_m = _get_number(["distance", "totalDistanceMeters"])
                distance_km = None if distance_m is None else distance_m / 1000.0

                daily_rows.append(
                    {
                        "date": d,
                        "steps": None if steps is None else int(steps),
                        "calories": calories,
                        "distance_km": distance_km,
                        "raw_stats": raw_stats,
                    }
                )

            daily_df = pd.DataFrame(daily_rows)

            # -------- activities --------
            # grab recent activities (0..99 is plenty for our use case)
            activities = client.get_activities(0, 100) or []
            act_rows = []
            for a in activities:
                act_rows.append(
                    {
                        "activityId": a.get("activityId"),
                        "activityName": a.get("activityName"),
                        "startTimeLocal": a.get("startTimeLocal"),
                        "startTimeGMT": a.get("startTimeGMT"),
                        "activityType": (a.get("activityType") or {}).get("typeKey"),
                        "distance": a.get("distance"),
                        "duration": a.get("duration"),
                        "calories": a.get("calories"),
                    }
                )
            activities_df = pd.DataFrame(act_rows)

        finally:
            try:
                client.logout()
            except Exception:
                pass

        return daily_df, activities_df
