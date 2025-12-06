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
        raw = self.get_nutrition_dataset()
        points = raw.get("point", [])

        rows = []

        for p in points:
            # timestamps are nanoseconds since epoch
            start_ns = int(p.get("startTimeNanos", "0"))
            if start_ns == 0:
                continue

            # convert to UTC date
            ts = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
            date = ts.date()

            vals = p.get("value", [])
            if not vals:
                continue

            map_vals = vals[0].get("mapVal", [])

            data = {
                "date": date,
                "calories_kcal": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
            }

            for mv in map_vals:
                key = mv.get("key")
                v = mv.get("value", {})
                val = v.get("fpVal") if "fpVal" in v else v.get("intVal")
                if val is None:
                    continue

                if key == "nutrition.calories":
                    data["calories_kcal"] = float(val)
                elif key == "nutrition.protein":
                    data["protein_g"] = float(val)
                elif key == "nutrition.carbs.total":
                    data["carbs_g"] = float(val)
                elif key == "nutrition.fat.total":
                    data["fat_g"] = float(val)

            rows.append(data)

        if not rows:
            return pd.DataFrame(columns=["date", "calories_kcal", "protein_g", "carbs_g", "fat_g"])

        df = pd.DataFrame(rows)

        # Aggregate per day
        agg = (
            df.groupby("date")
            .agg(
                calories_kcal=("calories_kcal", "sum"),
                protein_g=("protein_g", "sum"),
                carbs_g=("carbs_g", "sum"),
                fat_g=("fat_g", "sum"),
            )
            .reset_index()
            .sort_values("date")
        )

        # Filter to last N days
        if days_back is not None:
            cutoff = datetime.utcnow().date() - timedelta(days=days_back)
            agg = agg[agg["date"] >= cutoff]

        return agg
