import json
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

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
        end = datetime.utcnow().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=days_back)
        end_ms = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)

        body = {
            "aggregateBy": [
                {"dataTypeName": "com.google.nutrition"}  # <- important
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

        buckets = response.get("bucket", [])
        rows = []

        for b in buckets:
            datasets = b.get("dataset", [])
            if not datasets:
                continue

            start_time_ms = int(b.get("startTimeMillis", "0") or "0")
            if start_time_ms == 0:
                continue

            day_dt = datetime.fromtimestamp(start_time_ms / 1000.0, tz=timezone.utc)
            day = day_dt.date()

            calories = protein = carbs = fat = 0.0

            for ds in datasets:
                points = ds.get("point", [])
                for p in points:
                    vals = p.get("value", [])
                    if not vals:
                        continue
                    map_vals = vals[0].get("mapVal", [])
                    for mv in map_vals:
                        key = mv.get("key")
                        v = mv.get("value", {})
                        val = v.get("fpVal") if "fpVal" in v else v.get("intVal")
                        if val is None:
                            continue
                        val = float(val)

                        # Your keys
                        if key in ("calories", "nutrition.calories"):
                            calories += val
                        elif key in ("protein", "nutrition.protein"):
                            protein += val
                        elif key in ("carbs.total", "nutrition.carbs.total"):
                            carbs += val
                        elif key in ("fat.total", "nutrition.fat.total"):
                            fat += val

            if any(x > 0 for x in [calories, protein, carbs, fat]):
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
