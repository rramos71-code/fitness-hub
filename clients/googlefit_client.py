import google_auth_oauthlib.flow
from googleapiclient.discovery import build
import streamlit as st
from utils.secrets import get_secret

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]

class GoogleFitClient:
    def __init__(self):
        self.client_id = get_secret("google_fit_client_id")
        self.client_secret = get_secret("google_fit_client_secret")

    def authorize(self):
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            {
                "installed": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=SCOPES,
        )

        auth_url, _ = flow.authorization_url(prompt="consent")
        st.write("Open this link to authorize Google Fit:")
        st.write(auth_url)

        code = st.text_input("Paste authorization code:")
        if not code:
            return None

        flow.fetch_token(code=code)
        credentials = flow.credentials
        service = build("fitness", "v1", credentials=credentials)
        return service

    def get_nutrition(self, service):
        data_source = "derived:com.google.nutrition.summary:com.google.android.gms:merge_nutrition"
        dataset = "0-{}".format(10**18)
        request = service.users().dataSources().datasets().get(
            userId="me",
            dataSourceId=data_source,
            datasetId=dataset
        )
        return request.execute()
