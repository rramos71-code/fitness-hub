import os

import streamlit as st
from googleapiclient.discovery import build
import google_auth_oauthlib.flow

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]


class GoogleFitClient:
    def __init__(self):
        # Only use top-level secrets or env, to avoid old values under [fitness_hub]
        client_id = st.secrets.get("GOOGLE_FIT_CLIENT_ID") or os.getenv("GOOGLE_FIT_CLIENT_ID")
        client_secret = st.secrets.get("GOOGLE_FIT_CLIENT_SECRET") or os.getenv("GOOGLE_FIT_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise RuntimeError(
                "Google Fit client id or secret missing. "
                "Set GOOGLE_FIT_CLIENT_ID and GOOGLE_FIT_CLIENT_SECRET in Streamlit secrets."
            )

        self.client_id = client_id
        self.client_secret = client_secret

    def authorize(self):
        # Explicit desktop / installed config
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

        # Explicitly set redirect_uri so Google gets it
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

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
