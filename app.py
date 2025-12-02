import streamlit as st
import pandas as pd

ACTIVITIES_KEY = "activities"


def init_session_state():
    if ACTIVITIES_KEY not in st.session_state:
        st.session_state[ACTIVITIES_KEY] = pd.DataFrame()


def load_hevy_csv(uploaded_file: "UploadedFile") -> pd.DataFrame:
    """
    Minimal loader for a Hevy CSV export.
    No aggregation or transformation yet.
    """
    df = pd.read_csv(uploaded_file)
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    st.set_page_config(
        page_title="Fitness Hub - Hevy sync",
        page_icon="💪",
        layout="wide",
    )

    init_session_state()

    st.title("Fitness Hub - Hevy sync")

    st.write(
        "Upload a Hevy CSV export. The file will be parsed and stored in session state. "
        "No aggregation or dashboard yet, this is only to validate the connection."
    )

    uploaded_file = st.file_uploader("Upload Hevy CSV export", type=["csv"])

    if uploaded_file is not None:
        df = load_hevy_csv(uploaded_file)
        st.session_state[ACTIVITIES_KEY] = df
        st.success(f"Loaded {len(df)} rows from Hevy and stored them in session state.")

    # Simple visual check that data is actually stored
    if not st.session_state[ACTIVITIES_KEY].empty:
        st.subheader("Current synced Hevy data (preview)")
        st.dataframe(st.session_state[ACTIVITIES_KEY].head())
        st.caption(f"Total rows stored: {len(st.session_state[ACTIVITIES_KEY])}")


if __name__ == "__main__":
    main()
