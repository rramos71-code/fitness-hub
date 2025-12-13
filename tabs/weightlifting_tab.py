import streamlit as st
import pandas as pd

from utils.hevy_processing import get_hevy_sets_from_session
from analytics.weightlifting import (
    build_exercise_library,
    build_progression_recommendations,
    build_notifications,
)


def render_weightlifting_tab() -> None:
    """
    Render the weightlifting tab using Hevy sets already present in session state.
    """
    st.header("Weightlifting")

    with st.expander("Controls", expanded=False):
        lookback_days = st.number_input("Lookback days (library)", min_value=14, max_value=365, value=90, step=7)
        show_warmups = st.checkbox("Include warmups", value=False)

    sets_df = get_hevy_sets_from_session()
    if sets_df is None or sets_df.empty:
        st.info("Sync Hevy workouts first.")
        return

    if not show_warmups:
        sets_df = sets_df[sets_df["is_working_set"]]

    if sets_df is None or sets_df.empty:
        st.info("No working sets yet. Try including warmups or sync again.")
        return

    # Summary metrics
    today = pd.Timestamp.utcnow().date()
    seven_start = today - pd.Timedelta(days=7)
    twenty8_start = today - pd.Timedelta(days=28)

    recent7 = sets_df[sets_df["date_day"] >= seven_start]
    recent28 = sets_df[sets_df["date_day"] >= twenty8_start]
    recent7_volume = (recent7["weight_kg"] * recent7["reps"]).sum()
    recent28_volume = (recent28["weight_kg"] * recent28["reps"]).sum()

    sessions_7d = recent7["date_day"].nunique()
    exercises_count = sets_df["exercise_name"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions (7d)", f"{sessions_7d}")
    c2.metric("Volume 7d (kg)", f"{recent7_volume:,.0f}")
    c3.metric("Volume 28d (kg)", f"{recent28_volume:,.0f}")
    c4.metric("Exercises", f"{exercises_count}")

    # Build tables
    library_df = build_exercise_library(sets_df, lookback_days=lookback_days)
    goal_type = st.session_state.get("goal_config", {}).get("goal_type", "recomp")
    progress_df = build_progression_recommendations(sets_df, goal_type=goal_type)
    notifications = build_notifications(progress_df)

    st.subheader("Notifications")
    if notifications["increase_now"]:
        st.success(f"Increase now: {', '.join(notifications['increase_now'])}")
    if notifications["hold"]:
        st.info(f"Hold: {', '.join(notifications['hold'])}")
    if notifications["decrease"]:
        st.warning(f"Decrease: {', '.join(notifications['decrease'])}")
    if notifications["low_confidence"]:
        st.warning(f"Low confidence: {', '.join(notifications['low_confidence'])}")
    if not any(notifications.values()):
        st.info("No recommendations yet.")

    st.subheader("Exercise library")
    st.dataframe(library_df, use_container_width=True)

    st.subheader("Progression recommendations")
    st.dataframe(progress_df, use_container_width=True)
