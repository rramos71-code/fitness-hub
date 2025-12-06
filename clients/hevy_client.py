import os
import streamlit as st
from hevy_api.client import HevyClient as SDKClient


class HevyClient:
    def __init__(self):
        api_key = None

        # 1. Try top level secrets
        try:
            if "HEVY_API_KEY" in st.secrets:
                api_key = st.secrets["HEVY_API_KEY"]
        except Exception:
            pass

        # 2. Try nested section [fitness_hub] if you used that before
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
        self.client = SDKClient()

    def get_workouts(self):
        resp = self.client.get_workouts()
        return resp.workouts or []
