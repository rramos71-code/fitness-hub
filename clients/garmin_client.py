from datetime import date, timedelta
import os
from typing import Tuple, List, Dict, Any

import pandas as pd
import streamlit as st
from garminconnect import Garmin


class GarminClient:
    """
    Wrapper for garminconnect.Garmin.

    - Reads GARMIN_EMAIL / GARMIN_PASSWORD from secrets or env.
    - Exposes:
        * get_daily_summary(day)                -> raw stats dict
        * fetch_daily_and_activities(days_back) -> (daily_df, activities_df)
    """

    def __init__(self) -> None:
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

        # 3. Env vars
        email = email or os.getenv("GARMIN_EMAIL")
        password = password or os.getenv("GARMIN_PASSWORD")

        if not email or not password:
            raise RuntimeError(
                "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD "
                "in Streamlit secrets or environment variables."
            )

        self.email = email
        self.password = password

    # ------------------------------------------------------------------
    def _login(self) -> Garmin:
        client = Garmin(self.email, self.password)
        client.login()
        return client

    # ------------------------------------------------------------------
    # Existing convenience methods
    # ------------------------------------------------------------------
    def get_today_summary(self):
        return self.get_daily_summary(date.today())

    def get_daily_summary(self, day: date) -> Dict[str, Any]:
        client = self._login()
        try:
            day_str = day.isoformat()
            stats = client.get_stats(day_str)
        finally:
            try:
                client.logout()
            except Exception:
                pass
        return stats

    # ------------------------------------------------------------------
    # New method expected by app.py
    # ------------------------------------------------------------------
    def fetch_daily_and_activities(
        self, days_back: int = 14
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch a simple daily summary and activities list for the last `days_back` days.

        Returns:
            daily_df      : one row per day (date, steps, calories & distance if available)
            activities_df : one row per activity from Garmin's get_activities()

        The schema is intentionally simple and robust; you can always extend
        it later once you know exactly which fields you want for analytics.
        """
        client = self._login()
        try:
            # --- daily summaries ---
            today = date.today()
            start = today - timedelta(days=days_back - 1)

            daily_rows: List[Dict[str, Any]] = []

            for i in range(days_back):
                day = start + timedelta(days=i)
                stats = client.get_stats(day.isoformat()) or {}

                # stats shape can vary; we try to be defensive
                steps = None
                calories = None
                distance_km = None

                # Common locations for the values in Garmin's JSON
                try:
                    steps = stats.get("steps", {}).get("total", None)
                except Exception:
                    pass

                try:
                    calories = stats.get("calories", {}).get("total", None)
                except Exception:
                    pass

                try:
                    distance_km = stats.get("distance", {}).get("total", None)
                except Exception:
                    pass

                daily_rows.append(
                    {
                        "date": day,
                        "steps": steps,
                        "calories": calories,
                        "distance_km": distance_km,
                        "raw_stats": stats,
                    }
                )

            daily_df = (
                pd.DataFrame(daily_rows)
                if daily_rows
                else pd.DataFrame(columns=["date", "steps", "calories", "distance_km", "raw_stats"])
            )

            # --- activities list ---
            # Garmin API: get_activities(start, limit)
            # We'll just grab recent activities; limit is a bit arbitrary.
            try:
                activities = client.get_activities(0, 100) or []
            except Exception:
                activities = []

            activities_df = (
                pd.DataFrame(activities) if activities else pd.DataFrame()
            )

        finally:
            try:
                client.logout()
            except Exception:
                pass

        return daily_df, activities_df
