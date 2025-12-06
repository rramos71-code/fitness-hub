import streamlit as st
from googleapiclient.discovery import build
import google_auth_oauthlib.flow
import os

from utils.secrets import get_secret

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]


class GoogleFitClient:
    def __init__(self):
        # Look in [fitness_hub] first, then top level, then env
        client_id = get_secret("google_fit_client_id") or st.secrets.get("GOOGLE_FIT_CLIENT_ID")
        client_secret = get_secret("google_fit_client_secret") or st.secrets.get("GOOGLE_FIT_CLIENT_SECRET")

        client_id = client_id or os.getenv("GOOGLE_FIT_CLIENT_ID")
        client_secret = client_secret or os.getenv("GOOGLE_FIT_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise RuntimeError(
                "Google Fit client id or secret missing. "
                "Set google_fit_client_id and google_fit_client_secret under [fitness_hub] "
                "or GOOGLE_FIT_CLIENT_ID and GOOGLE_FIT_CLIENT_SECRET at top level."
            )

        self.client_id = client_id
        self.client_secret = client_secret

    def authorize(self):
        # Installed app style config (shows code you copy paste)
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            {
                "installed": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            scopes=SCOPES,
        )

        auth_url, _ = flow.authorization_url(prompt="consent")

        st.write("1. Open this link and authorize Google Fit:")
        st.write(auth_url)
        code = st.text_input("2. Paste the authorization code here", key="google_fit_code")

        if not code:
            return None

        try:
            flow.fetch_token(code=code)
        except Exception as e:
            st.error(f"Error exchanging code for token: {e}")
            return None

        credentials = flow.credentials
        service = build("fitness", "v1", credentials=credentials)
        return service

    def get_nutrition(self, service):
        data_source = "derived:com.google.nutrition.summary:com.google.android.gms:merge_nutrition"
        dataset = "0-{}".format(10**18)
        request = service.users().dataSources().datasets().get(
            userId="me",
            dataSourceId=data_source,
            datasetId=dataset,
        )
        return request.execute()
