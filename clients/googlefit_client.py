import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any

import pandas as pd
import streamlit as st
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]


class GoogleFitClient:
    """
    Wrapper around Google Fit REST API for nutrition.

    - Uses a pre-authorised token JSON from Streamlit secrets.
    - Exposes:
        * get_daily_macros(days_back)
        * aggregate_daily_macros(days_back)  # thin alias used by app.py
        * debug_aggregate_raw(days_back)
    """

    def __init__(self) -> None:
        self.creds = self._build_credentials()
        self.service = build("fitness", "v1", credentials=self.creds)

    def _build_credentials(self) -> Credentials:
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
                try:
                    creds.refresh(Request())
                    # Cache the refreshed token in session_state for the current run.
                    st.session_state["google_fit_token_json"] = creds.to_json()
                except RefreshError as exc:
                    raise RuntimeError(
                        "Google Fit refresh failed. Re-authenticate and update GOOGLE_FIT_TOKEN_JSON."
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(
                        "Google Fit refresh failed due to an unexpected error."
                    ) from exc
            else:
                raise RuntimeError(
                    "Stored Google Fit credentials are invalid or missing refresh_token; re-authenticate and replace GOOGLE_FIT_TOKEN_JSON."
                )

        return creds

    # ------------------------------------------------------------------
    def _aggregate_request(self, days_back: int) -> Dict[str, Any]:
        tz = ZoneInfo("Europe/Berlin")

        end_local = datetime.now(tz)
        start_local = end_local - timedelta(days=days_back)

        end_ms = int(end_local.timestamp() * 1000)
        start_ms = int(start_local.timestamp() * 1000)

        body = {
            "aggregateBy": [{"dataTypeName": "com.google.nutrition"}],
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

    # ------------------------------------------------------------------
    # Main logic: convert aggregate response into daily macros
    # ------------------------------------------------------------------
    def get_daily_macros(self, days_back: int = 14) -> pd.DataFrame:
        tz = ZoneInfo("Europe/Berlin")
        response = self._aggregate_request(days_back)
        buckets = response.get("bucket", [])

        rows = []

        for b in buckets:
            start_ms = int(b.get("startTimeMillis", "0") or "0")
            if start_ms == 0:
                continue

            day_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=tz)
            day = day_dt.date()

            calories = protein = carbs = fat = 0.0

            for ds in b.get("dataset", []):
                for p in ds.get("point", []):
                    origin = p.get("originDataSourceId", "")

                    # We no longer filter to LoseIt only: any app that writes
                    # com.google.nutrition.summary is accepted (Cronometer, Lose It, manual input,…)

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

    # ------------------------------------------------------------------
    # Alias used by app.py
    # ------------------------------------------------------------------
    def aggregate_daily_macros(self, days_back: int = 14) -> pd.DataFrame:
        """
        Thin wrapper used by the Streamlit UI.
        """
        return self.get_daily_macros(days_back=days_back)

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------
    def debug_aggregate_raw(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Return raw aggregate response for debugging.
        """
        return self._aggregate_request(days_back)
