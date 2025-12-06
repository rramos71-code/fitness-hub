from garminconnect import Garmin
from utils.secrets import get_secret
from datetime import date

class GarminClient:
    def __init__(self):
        self.username = get_secret("garmin_user")
        self.password = get_secret("garmin_password")
        self.client = Garmin(self.username, self.password)
        self.client.login()

    def get_daily_summary(self, date_input: date):
        return self.client.get_stats(date_input)
