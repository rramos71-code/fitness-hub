import os
import streamlit as st
from hevy_api.client import HevyClient


def get_hevy_client(api_key: str) -> HevyClient:
    """
    Create a HevyClient using the given API key.

    The hevy_api library reads the key from HEVY_API_KEY,
    so we set that environment variable just for this process.
    """
    os.environ["HEVY_API_KEY"] = api_key
    return HevyClient()


def main():
    st.set_page_config(
        page_title="Fitness Hub - Hevy API",
        page_icon="💪",
        layout="centered",
    )

    st.title("Fitness Hub - Hevy API connection")

    st.write(
        "Paste your Hevy API key and press Test connection. "
        "If it works, we will fetch your workouts and show a small preview."
    )

    api_key = st.text_input(
        "Hevy API key",
        type="password",
        help="For Hevy Pro users. You get it in Hevy settings, under Developer.",
    )

    if api_key and st.button("Test connection"):
        with st.spinner("Contacting Hevy API"):
            try:
                client = get_hevy_client(api_key)

                response = client.get_workouts()
                workouts = response.workouts or []

                st.success(f"Connection successful. Retrieved {len(workouts)} workouts.")

                if workouts:
                    last = workouts[-1]
                    st.subheader("Most recent workout")

                    # Try to convert model to JSON if possible
                    if hasattr(last, "model_dump"):
                        st.json(last.model_dump())
                    elif hasattr(last, "dict"):
                        st.json(last.dict())
                    else:
                        st.write(last)

            except Exception as e:
                st.error(f"Error communicating with Hevy API: {e}")
    else:
        st.info("Enter your API key above to test the connection.")


if __name__ == "__main__":
    main()
