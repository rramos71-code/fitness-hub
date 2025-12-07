import os
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from hevy_api.client import HevyClient as SDKClient


class HevyClient:
    """
    Thin wrapper around the official Hevy SDK client.

    - Reads HEVY_API_KEY from Streamlit secrets or env.
    - Exposes:
        * get_workouts()  -> raw workouts list from SDK
        * sync_workouts() -> (workouts_df, sets_df) pandas DataFrames
    """

    def __init__(self) -> None:
        api_key = None

        # 1. Try top level secrets
        try:
            if "HEVY_API_KEY" in st.secrets:
                api_key = st.secrets["HEVY_API_KEY"]
        except Exception:
            pass

        # 2. Try nested section [fitness_hub]
        if not api_key:
            try:
                fh = st.secrets.get("fitness_hub", {})
                if "HEVY_API_KEY" in fh:
                    api_key = fh["HEVY_API_KEY"]
            except Exception:
                pass

        # 3. Fallback to environment variable
        if not api_key:
            api_key = os.getenv("HEVY_API_KEY")

        if not api_key:
            raise RuntimeError(
                "HEVY_API_KEY not found in Streamlit secrets or environment variables."
            )

        # The SDK reads HEVY_API_KEY from the environment
        os.environ["HEVY_API_KEY"] = api_key
        self.client: SDKClient = SDKClient()

    # ------------------------------------------------------------------
    # Old method (used already in your app earlier)
    # ------------------------------------------------------------------
    def get_workouts(self) -> List[Dict[str, Any]]:
        """Return the raw workouts list from the SDK."""
        resp = self.client.get_workouts()
        # SDK typically returns an object with .workouts; fall back to resp itself.
        workouts = getattr(resp, "workouts", resp)
        return workouts or []

    # ------------------------------------------------------------------
    # New method expected by app.py
    # ------------------------------------------------------------------
    def sync_workouts(self):
        """
        Fetch workouts from Hevy and return two DataFrames:

        - workouts_df: one row per workout
        - sets_df: one row per set (flattened from exercises/sets)

        This is intentionally generic so it works even if Hevy slightly
        changes field names. Missing fields will just become NaN.
        """
        workouts = self.get_workouts()

        if not workouts:
            return pd.DataFrame(), pd.DataFrame()

        # 1) workouts_df: simple json_normalize over the workout objects
        workouts_df = pd.json_normalize(workouts)

        # 2) sets_df: flatten exercises / sets if present
        set_rows = []

        for w in workouts:
            # workouts might be dicts or dataclass-like; handle both.
            if not isinstance(w, dict):
                # fall back to __dict__ if needed
                w = getattr(w, "__dict__", {})

            workout_id = (
                w.get("id")
                or w.get("workout_id")
                or w.get("_id")
            )
            workout_date = (
                w.get("start_time")
                or w.get("startTime")
                or w.get("performed_at")
            )

            exercises = w.get("exercises", []) or w.get("workout_exercises", [])

            for ex in exercises:
                if not isinstance(ex, dict):
                    ex = getattr(ex, "__dict__", {})
                ex_name = ex.get("name") or ex.get("exercise_name")

                sets = ex.get("sets", []) or ex.get("exercise_sets", [])
                for s in sets:
                    if not isinstance(s, dict):
                        s = getattr(s, "__dict__", {})
                    row = {
                        "workout_id": workout_id,
                        "workout_date": workout_date,
                        "exercise_name": ex_name,
                    }
                    # include all set fields as-is (kg, reps, rpe, etc.)
                    row.update(s)
                    set_rows.append(row)

        if set_rows:
            sets_df = pd.DataFrame(set_rows)
        else:
            sets_df = pd.DataFrame(columns=["workout_id", "workout_date", "exercise_name"])

        return workouts_df, sets_df
