import streamlit as st
import pandas as pd

from clients.hevy_client import HevyClient
from utils.hevy_processing import (
    build_exercise_library,
    build_exercise_progression,
    build_progression_recommendations,
    ensure_hevy_date_column,
    standardize_hevy_sets,
)


def get_hevy_sets_from_session() -> pd.DataFrame | None:
    """
    Pull sets from session and normalize schema/date columns
    so downstream analytics never KeyError on 'date'.
    """
    df = st.session_state.get("hevy_sets_df")
    if df is None or getattr(df, "empty", True):
        return None

    try:
        df = standardize_hevy_sets(df)
    except Exception as exc:
        # last resort: try simpler date ensure to avoid hard failure
        try:
            df = ensure_hevy_date_column(df)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            st.error(f"Hevy sets missing required date fields: {exc}")
            return None

    if df.empty or "date" not in df.columns:
        return None

    return df

def render_weightlifting_tab():
    st.header("Weightlifting")

    with st.expander("Controls", expanded=True):
        lookback_days = st.number_input(
            "Lookback days (library)", min_value=7, max_value=365, value=90, step=7
        )
        include_warmups = st.checkbox("Include warmups", value=False)

    # --- NEW: allow syncing here ---
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Sync Hevy now", use_container_width=True):
            try:
                client = HevyClient()
                workouts_df, sets_df = client.sync_workouts()
                sets_df = standardize_hevy_sets(sets_df)

                # Persist for other tabs
                st.session_state["hevy_workouts_df"] = workouts_df
                st.session_state["hevy_sets_df"] = sets_df

                st.success(f"Hevy synced: {len(workouts_df)} workouts, {len(sets_df)} sets")
            except Exception as e:
                st.error(f"Hevy sync failed: {e}")

    with c2:
        sets_df = get_hevy_sets_from_session()
        if sets_df is None:
            st.info("Sync Hevy workouts first.")
            with st.expander("Debug: session keys", expanded=False):
                st.write(sorted(list(st.session_state.keys())))
            return
        st.caption(f"Loaded Hevy sets from session: {len(sets_df)} rows")

    # --- Existing analytics ---
    lib_df = build_exercise_library(
        sets_df, lookback_days=lookback_days, include_warmups=include_warmups
    )
    st.subheader("Exercise library")
    st.dataframe(lib_df, use_container_width=True)

    prog_df = build_exercise_progression(
        sets_df, lookback_days=lookback_days, include_warmups=include_warmups
    )
    st.subheader("Progression signals")
    st.dataframe(prog_df, use_container_width=True)

    recs_df = build_progression_recommendations(prog_df)
    st.subheader("Recommendations")
    st.dataframe(recs_df, use_container_width=True)
