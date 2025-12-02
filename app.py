import os
import streamlit as st
from hevy_api.client import HevyClient


def get_hevy_client() -> HevyClient:
    """
    Create a HevyClient using the HEVY_API_KEY from
    Streamlit secrets or environment variables.
    """
    api_key = None

    # 1. Streamlit secrets (Cloud and local secrets.toml)
    if "HEVY_API_KEY" in st.secrets:
        api_key = st.secrets["HEVY_API_KEY"]

    # 2. Fallback to environment variable if needed
    if not api_key:
        api_key = os.getenv("HEVY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Hevy API key not found. Set HEVY_API_KEY in Streamlit secrets "
            "or as an environment variable."
        )

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
        "This app reads the Hevy API key from Streamlit secrets and tests the connection "
        "automatically."
    )

    if st.button("Sync workouts"):
        with st.spinner("Contacting Hevy API"):
            try:
                client = get_hevy_client()
                response = client.get_workouts()
                workouts = response.workouts or []

                st.success(f"Connection successful. Retrieved {len(workouts)} workouts.")

                if workouts:
                    last = workouts[-1]
                    st.subheader("Most recent workout")

                    if hasattr(last, "model_dump"):
                        st.json(last.model_dump())
                    elif hasattr(last, "dict"):
                        st.json(last.dict())
                    else:
                        st.write(last)

            except Exception as e:
                st.error(f"Error communicating with Hevy API: {e}")
    else:
        st.info("Press Sync workouts to test the connection using the stored API key.")


if __name__ == "__main__":
    main()
