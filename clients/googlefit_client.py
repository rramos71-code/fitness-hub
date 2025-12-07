import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]


class GoogleFitClient:
    def __init__(self):
        token_json = st.secrets.get("GOOGLE_FIT_TOKEN_JSON")
        client_id = st.secrets.get("GOOGLE_FIT_CLIENT_ID")
        client_secret = st.secrets.get("GOOGLE_FIT_CLIENT_SECRET")

        if not token_json or not client_id or not client_secret:
            raise RuntimeError(
                "Google Fit secrets missing. Set GOOGLE_FIT_CLIENT_ID, "
                "GOOGLE_FIT_CLIENT_SECRET and GOOGLE_FIT_TOKEN_JSON in secrets."
            )

        info = json.loads(token_json)
        info.setdefault("client_id", client_id)
        info.setdefault("client_secret", client_secret)

        creds = Credentials.from_authorized_user_info(info, scopes=SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise RuntimeError("Stored Google Fit credentials are invalid and cannot be refreshed.")

        self.service = build("fitness", "v1", credentials=creds)

    def get_daily_macros(self, days_back: int = 14) -> pd.DataFrame:
        """
        Aggregate daily calories and macros via Google Fit aggregate API.

        - Uses com.google.nutrition as base type
        - Buckets by 1 day in Europe/Berlin
        - Sums all points in each bucket
        - Only counts entries coming from Lose It (originDataSourceId contains 'com.fitnow.loseit')
        """
        tz = ZoneInfo("Europe/Berlin")

        # Use local time boundaries so buckets align with what you see in the app
        end_local = datetime.now(tz)
        start_local = end_local - timedelta(days=days_back)

        end_ms = int(end_local.timestamp() * 1000)
        start_ms = int(start_local.timestamp() * 1000)

        body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.nutrition"}
            ],
            "bucketByTime": {
                "durationMillis": 24 * 60 * 60 * 1000,
                "timeZoneId": "Europe/Berlin",
            },
            "startTimeMillis": start_ms,
            "endTimeMillis": end_ms,
        }

        response = (
            self.service.users()
            .dataset()
            .aggregate(userId="me", body=body)
            .execute()
        )

        buckets = response.get("bucket", [])
        rows = []

        for b in buckets:
            start_time_ms = int(b.get("startTimeMillis", "0") or "0")
            if start_time_ms == 0:
                continue

            # convert bucket start to local calendar date
            day_dt = datetime.fromtimestamp(start_time_ms / 1000.0, tz=tz)
            day = day_dt.date()

            calories = protein = carbs = fat = 0.0

            for ds in b.get("dataset", []):
                for p in ds.get("point", []):
                    origin = p.get("originDataSourceId", "")

                    # keep only Lose It entries
                    if "com.fitnow.loseit" not in origin:
                        continue

                    for val in p.get("value", []):
                        for mv in val.get("mapVal", []):
                            key = mv.get("key")
                            v = mv.get("value", {})
                            raw = v.get("fpVal") if "fpVal" in v else v.get("intVal")
                            if raw is None:
                                continue
                            value = float(raw)

                            if key in ("calories", "nutrition.calories"):
                                calories += value
                            elif key in ("protein", "nutrition.protein"):
                                protein += value
                            elif key in ("carbs.total", "nutrition.carbs.total"):
                                carbs += value
                            elif key in ("fat.total", "nutrition.fat.total"):
                                fat += value

            # only add rows that actually have some macro data
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
            return pd.DataFrame(
                columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"]
            )

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df

    def debug_aggregate_raw(self, days_back: int = 7) -> dict:
        """Return raw aggregate response for debugging in the Streamlit app."""
        tz = ZoneInfo("Europe/Berlin")
        end_local = datetime.now(tz)
        start_local = end_local - timedelta(days=days_back)

        end_ms = int(end_local.timestamp() * 1000)
        start_ms = int(start_local.timestamp() * 1000)

        body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.nutrition"}
            ],
            "bucketByTime": {
                "durationMillis": 24 * 60 * 60 * 1000,
                "timeZoneId": "Europe/Berlin",
            },
            "startTimeMillis": start_ms,
            "endTimeMillis": end_ms,
        }

        response = (
            self.service.users()
            .dataset()
            .aggregate(userId="me", body=body)
            .execute()
        )
        return response
