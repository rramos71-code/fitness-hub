import requests
from utils.secrets import get_secret

BASE_URL = "https://hevy.com/api/v1"

class HevyClient:
    def __init__(self):
        self.api_key = get_secret("hevy_api_key")

    def get_workouts(self):
        headers = {"x-api-key": self.api_key}
        url = f"{BASE_URL}/workouts"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
