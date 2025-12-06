import json
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from zoneinfo import ZoneInfo


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

    # Raw dataset, like you already tested
    def get_nutrition_dataset(self):
        data_source = "derived:com.google.nutrition.summary:com.google.android.gms:merge_nutrition"
        dataset = f"0-{10**18}"
        request = self.service.users().dataSources().datasets().get(
            userId="me",
            dataSourceId=data_source,
            datasetId=dataset,
        )
        return request.execute()

    # New helper: aggregate daily calories and macros
    def get_daily_macros(self, days_back: int = 14) -> pd.DataFrame:
        """
        Aggregate daily calories and macros using the Google Fit aggregate API.
        Uses base type com.google.nutrition (required by API) and parses the
        returned com.google.nutrition.summary points with keys like
        'calories', 'protein', 'carbs.total', 'fat.total'.
        """
        tz = ZoneInfo("Europe/Berlin")  # your local TZ

        # End = now in local time, start = N days back in local time
        end_local = datetime.now(tz=tz)
        start_local = end_local - timedelta(days=days_back)

        end_ms = int(end_local.timestamp() * 1000)
        start_ms = int(start_local.timestamp() * 1000)

        body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.nutrition"}
            ],
            "bucketByTime": {
                "durationMillis": 24 * 60 * 60 * 1000,
                "timeZoneId": "Europe/Berlin",  # <- key line
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
            # Extract start time of the bucket in ms
            start_time_ms = int(b.get("startTimeMillis", "0") or "0")
            if start_time_ms == 0:
                continue

            # Convert bucket start into your local timezone (Europe/Berlin)
            day_dt = datetime.fromtimestamp(start_time_ms / 1000.0, tz=tz)
            day = day_dt.date()

            # Default values
            calories = protein = carbs = fat = 0.0

            # Read all datasets in the bucket
            for dataset in b.get("dataset", []):
                for point in dataset.get("point", []):
                    for val in point.get("value", []):
                        # Nutrition is encoded as mapVal
                        for m in val.get("mapVal", []):
                            key = m.get("key")
                            v = m.get("value", {}).get("fpVal", 0)

                            if key == "calories":
                                calories = v
                            elif key == "protein":
                                protein = v
                            elif key == "carbs.total":
                                carbs = v
                            elif key == "fat.total":
                                fat = v

            # Append even if calories/macros are zero
            rows.append({
                "date": str(day),
                "calories_kcal": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
            })

        if not rows:
            return pd.DataFrame(
                columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"]
            )

        return pd.DataFrame(rows).sort_values("date")

    def debug_aggregate_raw(self, days_back: int = 3):
        """
        Call the aggregate API and return the raw response
        so we can inspect the structure (data types, keys, etc.).
        """
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=days_back)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)

        body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.nutrition"}
            ],
            "bucketByTime": {"durationMillis": 24 * 60 * 60 * 1000},
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
