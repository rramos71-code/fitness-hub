import os
import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/fitness.nutrition.read"]


class GoogleFitClient:
    def __init__(self):
        self.client_id = st.secrets["GOOGLE_FIT_CLIENT_ID"]
        self.client_secret = st.secrets["GOOGLE_FIT_CLIENT_SECRET"]
        self.redirect_uri = st.secrets.get("GOOGLE_FIT_REDIRECT_URI", "http://localhost")

    def authorize(self):
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=SCOPES,
        )

        flow.redirect_uri = self.redirect_uri

        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        st.write("### 1. Click to authorize Google Fit:")
        st.link_button("Authorize Google Fit", auth_url)

        st.write("""
        ### 2. After logging in, Google will try to redirect to `http://localhost`  
        This will fail to load, but the **URL will contain `?code=XYZ`**.  
        Copy that code and paste it below.
        """)

        code = st.text_input("Paste the `code` value here:")

        if not code:
            return None

        try:
            flow.fetch_token(code=code)
        except Exception as e:
            st.error(f"Error exchanging code for token: {e}")
            return None

        credentials = flow.credentials
        return build("fitness", "v1", credentials=credentials)

    def get_nutrition(self, service):
        datasource = "derived:com.google.nutrition.summary:com.google.android.gms:merge_nutrition"
        dataset = f"0-{10**18}"

        request = service.users().dataSources().datasets().get(
            userId="me",
            dataSourceId=datasource,
            datasetId=dataset
        )
        return request.execute()
