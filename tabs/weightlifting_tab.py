# tabs/weightlifting_tab.py
import streamlit as st
import pandas as pd
from utils.hevy_processing import (
    normalize_hevy_sets, build_exercise_library,
    build_exercise_progression, build_progression_recommendations,
    DATE_CANDIDATES, EXERCISE_CANDIDATES,
)
from utils.hevy_schema import ProgressionConfig

def render(hevy_client):
    st.header("Weightlifting")

    cfg = ProgressionConfig(
        lookback_days=st.number_input("Lookback days (library)", 7, 365, 90, 1),
        include_warmups=st.checkbox("Include warmups", value=False),
    )

    # sync
    if st.button("Sync Hevy now"):
        workouts_df, sets_df = hevy_client.sync_workouts()
        st.session_state["hevy_workouts_df"] = workouts_df
        st.session_state["hevy_sets_df"] = sets_df
        st.success(f"Hevy synced: {len(workouts_df)} workouts, {len(sets_df)} sets")

    workouts_df = st.session_state.get("hevy_workouts_df", pd.DataFrame())
    sets_df = st.session_state.get("hevy_sets_df", pd.DataFrame())

    with st.expander("Debug: Hevy raw sets"):
        st.write(f"Rows: {len(sets_df)}")
        st.write(list(sets_df.columns)[:50])
        if not sets_df.empty:
            date_nulls = {c: int(sets_df[c].isna().sum()) for c in DATE_CANDIDATES if c in sets_df.columns}
            ex_nulls = {c: int(sets_df[c].isna().sum()) for c in EXERCISE_CANDIDATES if c in sets_df.columns}
            st.write({"date_nulls": date_nulls, "exercise_nulls": ex_nulls})
        st.dataframe(sets_df.head(10))

    sets_norm = normalize_hevy_sets(sets_df, workouts_df=workouts_df, warn=st.warning)
    norm_debug = getattr(sets_norm, "attrs", {}).get("normalize_debug", {})

    with st.expander("Debug: Hevy normalized sets"):
        st.write(f"Rows: {len(sets_norm)}")
        st.write(list(sets_norm.columns))
        if norm_debug:
            st.write({"normalize_debug": norm_debug})
        if not sets_norm.empty:
            st.write({
                "min_date": str(sets_norm["date"].min()),
                "max_date": str(sets_norm["date"].max()),
                "exercise_count": int(sets_norm["exercise_name"].nunique()),
            })
        st.dataframe(sets_norm.head(10))

    if sets_norm.empty:
        st.info("No usable Hevy sets after normalization. Check column mapping in normalize_hevy_sets().")
        return

    st.subheader("Exercise library")
    lib = build_exercise_library(sets_norm, cfg, warn=st.warning)
    st.dataframe(lib, use_container_width=True)

    st.subheader("Progression signals")
    prog = build_exercise_progression(sets_norm, cfg)
    st.dataframe(prog, use_container_width=True)

    st.subheader("Recommendations")
    recs = build_progression_recommendations(prog, cfg)
    st.dataframe(recs, use_container_width=True)
