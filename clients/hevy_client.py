from hevy_api.client import HevyClient as SDKClient
import os

class HevyClient:
    def __init__(self):
        api_key = os.getenv("HEVY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "HEVY_API_KEY not found. Set it in Streamlit Secrets or environment variables."
            )
        os.environ["HEVY_API_KEY"] = api_key
        self.client = SDKClient()

    def get_workouts(self):
        response = self.client.get_workouts()
        return response.workouts or []
