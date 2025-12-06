import requests
from utils.secrets import get_secret

BASE_URL = "https://hevy.com/api/v1"

class HevyClient:
    def __init__(self):
        self.api_key = get_secret("hevy_api_key")

    def _headers(self):
        return {"x-api-key": self.api_key}

    def get_user(self):
        url = f"{BASE_URL}/me"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def get_workouts(self):
        # Step 1: get current user's ID
        user = self.get_user()
        user_id = user["id"]

        # Step 2: fetch workouts for that user
        url = f"{BASE_URL}/users/{user_id}/workouts"
        resp = requests.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()
