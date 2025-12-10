import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# ---- Local modules ----
from clients.hevy_client import HevyClient
from clients.garmin_client import GarminClient
from clients.googlefit_client import GoogleFitClient

# =========================================================
#  Column mapping from unified daily_df
# =========================================================
COLS = {
    # nutrition (from Google Fit)
    "calories_in": "calories_kcal",
    "protein_g": "protein_g",
    "carbs_g": "carbs_g",
    "fat_g": "fat_g",

    # activity / energy out (from Garmin)
    "calories_out": "garmin_calories_kcal",
    "steps": "garmin_steps",
    "sleep_hours": "garmin_sleep_hours",

    # resistance training (from Hevy aggregated per day)
    "hevy_sets": "hevy_sets",
    "hevy_volume": "hevy_volume_kg",
}

# =========================================================
# Helper utilities
# =========================================================
def ensure_date_column(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Guarantee a 'date' column exists if the DF is not empty.
    Tries common alternatives like 'calendarDate' or 'day'.
    """
    if df is None or df.empty:
        return df

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    df = df.copy()
    for cand in ["calendarDate", "calendar_date", "day", "Day", "DATE", "startTime"]:
        if cand in df.columns:
            df["date"] = pd.to_datetime(df[cand]).dt.date
            return df

    # last resort: if index looks like dates
    if isinstance(df.index, pd.DatetimeIndex):
        df["date"] = df.index.date

    return df


def _first_existing(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_daily_dataset(
    nutrition_df: pd.DataFrame | None,
    garmin_daily_df: pd.DataFrame | None,
    garmin_activities_df: pd.DataFrame | None,  # currently unused but kept for future
    hevy_sets_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Small, defensive unifier that creates a per-day DataFrame across
    Google Fit nutrition, Garmin daily stats, and Hevy sets.
    """

    frames = []

    # ---------- Google Fit: nutrition ----------
    if nutrition_df is not None and not nutrition_df.empty:
        n = ensure_date_column(nutrition_df)
        if n is not None and not n.empty and "date" in n.columns:
            n = n.copy()
            n["date"] = pd.to_datetime(n["date"]).dt.date

            # These are the columns produced by GoogleFitClient.aggregate_daily_macros
            agg_cols = {
                "calories_kcal": "sum",
                "protein_g": "sum",
                "carbs_g": "sum",
                "fat_g": "sum",
            }
            existing = {k: v for k, v in agg_cols.items() if k in n.columns}
            n = n.groupby("date", as_index=False).agg(existing)

            # names already aligned with COLS mapping
            frames.append(n)

    # ---------- Garmin: daily summary ----------
    if garmin_daily_df is not None and not garmin_daily_df.empty:
        g = ensure_date_column(garmin_daily_df)
        if g is not None and not g.empty and "date" in g.columns:
            g = g.copy()
            g["date"] = pd.to_datetime(g["date"]).dt.date

            # Try to infer calories / steps / sleep columns from typical Garmin exports
            cal_col = _first_existing(
                g, ["caloriesTotal", "calories_total", "activeKilocalories", "calories"]
            )
            steps_col = _first_existing(
                g, ["steps", "stepCount", "totalSteps"]
            )
            sleep_sec_col = _first_existing(
                g, ["sleepingSeconds", "sleepDurationInSeconds", "sleepSeconds"]
            )

            if sleep_sec_col:
                g["sleep_hours_tmp"] = g[sleep_sec_col] / 3600.0

            agg_spec = {}
            if cal_col:
                agg_spec[cal_col] = "sum"
            if steps_col:
                agg_spec[steps_col] = "sum"
            if "sleep_hours_tmp" in g.columns:
                agg_spec["sleep_hours_tmp"] = "sum"

            if agg_spec:
                g_day = g.groupby("date", as_index=False).agg(agg_spec)

                if cal_col:
                    g_day.rename(
                        columns={cal_col: "garmin_calories_kcal"}, inplace=True
                    )
                if steps_col:
                    g_day.rename(columns={steps_col: "garmin_steps"}, inplace=True)
                if "sleep_hours_tmp" in g_day.columns:
                    g_day.rename(
                        columns={"sleep_hours_tmp": "garmin_sleep_hours"}, inplace=True
                    )

                frames.append(g_day)

    # ---------- Hevy: sets ----------
    if hevy_sets_df is not None and not hevy_sets_df.empty:
        h = ensure_date_column(hevy_sets_df)
        if h is not None and not h.empty and "date" in h.columns:
            h = h.copy()
            h["date"] = pd.to_datetime(h["date"]).dt.date

            weight_col = _first_existing(h, ["weight_kg", "weight"])
            reps_col = _first_existing(h, ["reps", "repetitions"])

            if weight_col and reps_col:
                h["volume_tmp"] = h[weight_col].fillna(0) * h[reps_col].fillna(0)
            else:
                h["volume_tmp"] = 0.0

            h_day = (
                h.groupby("date")
                .agg(
                    hevy_sets=("id", "count") if "id" in h.columns else ("volume_tmp", "size"),
                    hevy_volume_kg=("volume_tmp", "sum"),
                )
                .reset_index()
            )

            frames.append(h_day)

    if not frames:
        return pd.DataFrame(columns=["date"])

    from functools import reduce

    daily = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"), frames
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def _ensure_date_and_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week_start"] = out["date"] - pd.to_timedelta(out["date"].dt.weekday, unit="D")
    return out


def build_weekly_energy_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_date_and_week(daily_df)

    cin_col = COLS["calories_in"]
    cout_col = COLS["calories_out"]

    if cin_col not in df.columns and cout_col not in df.columns:
        return pd.DataFrame()

    group = df.groupby("week_start")
    rows = []

    for week, g in group:
        row = {"week_start": week.date(), "days_with_data": len(g)}

        if cin_col in g.columns:
            row["calories_in_sum"] = g[cin_col].sum()
            row["calories_in_avg"] = g[cin_col].mean()
        if cout_col in g.columns:
            row["calories_out_sum"] = g[cout_col].sum()
            row["calories_out_avg"] = g[cout_col].mean()

        if "calories_in_sum" in row and "calories_out_sum" in row:
            row["energy_balance_sum"] = row["calories_in_sum"] - row["calories_out_sum"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)


def build_macro_adherence(
    daily_df: pd.DataFrame,
    target_calories: float,
    target_protein: float,
) -> pd.DataFrame:
    df = daily_df.copy()
    cin_col = COLS["calories_in"]
    prot_col = COLS["protein_g"]

    if cin_col not in df.columns and prot_col not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    if cin_col in df.columns:
        df["calories_delta"] = df[cin_col] - target_calories
        df["calories_within_5pct"] = df[cin_col].between(
            0.95 * target_calories, 1.05 * target_calories
        )

    if prot_col in df.columns:
        df["protein_delta"] = df[prot_col] - target_protein
        df["protein_met"] = df[prot_col] >= target_protein

    cols = ["date"]
    if cin_col in df.columns:
        cols += [cin_col, "calories_delta", "calories_within_5pct"]
    if prot_col in df.columns:
        cols += [prot_col, "protein_delta", "protein_met"]

    return df[cols].copy()


def build_training_load_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_date_and_week(daily_df)

    sets_col = COLS["hevy_sets"]
    volume_col = COLS["hevy_volume"]
    steps_col = COLS["steps"]

    if (sets_col not in df.columns and
        volume_col not in df.columns and
        steps_col not in df.columns):
        return pd.DataFrame()

    df["is_hevy_day"] = df[sets_col] > 0 if sets_col in df.columns else False
    if steps_col in df.columns:
        df["is_high_step_day"] = df[steps_col] >= 8000
    else:
        df["is_high_step_day"] = False

    df["is_training_day"] = df["is_hevy_day"] | df["is_high_step_day"]

    group = df.groupby("week_start")
    rows = []

    for week, g in group:
        row = {
            "week_start": week.date(),
            "days": len(g),
            "training_days": int(g["is_training_day"].sum()),
        }
        if sets_col in g.columns:
            row["hevy_sets_total"] = g[sets_col].sum()
        if volume_col in g.columns:
            row["hevy_volume_total"] = g[volume_col].sum()
        if steps_col in g.columns:
            row["steps_avg"] = g[steps_col].mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)


# =========================================================
# Instantiate API clients
# =========================================================
hevy_client = HevyClient()
garmin_client = GarminClient()
gf_client = GoogleFitClient()

# =========================================================
# UI
# =========================================================
st.title("Fitness Hub — Core Integrations Test")

# ----------------------- Hevy ----------------------------
st.header("Hevy Connection")

if st.button("Sync Hevy workouts"):
    try:
        workouts_df, sets_df = hevy_client.sync_workouts()
        st.session_state["hevy_sets_df"] = sets_df

        st.success(f"Hevy connection OK, {len(workouts_df)} workouts retrieved")
        st.subheader("Hevy sets (sample)")
        st.dataframe(sets_df.head())
    except Exception as e:
        st.error(f"Hevy error: {e}")

# ----------------------- Garmin --------------------------
st.header("Garmin Connection")

if st.button("Test Garmin"):
    try:
        daily_df_gar, activities_df = garmin_client.fetch_daily_and_activities()
        st.session_state["garmin_daily_df"] = daily_df_gar
        st.session_state["garmin_activities_df"] = activities_df

        st.write("Daily")
        st.dataframe(daily_df_gar.head())

        st.write("Activities")
        st.dataframe(activities_df.head())
    except Exception as e:
        st.error(f"Garmin error: {e}")

# ----------------------- Google Fit ----------------------
st.header("Google Fit Connection")

days_back = st.number_input("Days back", min_value=1, max_value=30, value=7, step=1)

if st.button("Test Google Fit nutrition"):
    try:
        df_gf = gf_client.aggregate_daily_macros(days_back=days_back)
        st.session_state["googlefit_nutrition_df"] = df_gf

        if df_gf.empty:
            st.info("Google Fit returned no nutrition entries for the selected period.")
        else:
            st.dataframe(df_gf)
    except Exception as e:
        st.error(f"Google Fit error: {e}")

if st.button("Debug raw Google Fit aggregate response"):
    try:
        raw = gf_client.debug_aggregate_raw(days_back=days_back)
        st.json(raw)
    except Exception as e:
        st.error(f"Google Fit debug error: {e}")

# ------------------- Unified daily view ------------------
st.subheader("Daily overview (unified dataset)")

nutrition_df = st.session_state.get("googlefit_nutrition_df")
garmin_daily_df = st.session_state.get("garmin_daily_df")
garmin_activities_df = st.session_state.get("garmin_activities_df")
hevy_sets_df = st.session_state.get("hevy_sets_df")

if (
    (nutrition_df is None or (hasattr(nutrition_df, "empty") and nutrition_df.empty))
    and (garmin_daily_df is None or (hasattr(garmin_daily_df, "empty") and garmin_daily_df.empty))
    and (garmin_activities_df is None or (hasattr(garmin_activities_df, "empty") and garmin_activities_df.empty))
    and (hevy_sets_df is None or (hasattr(hevy_sets_df, "empty") and hevy_sets_df.empty))
):
    st.info(
        "Load data from at least one source (Google Fit, Garmin, Hevy) "
        "to see the unified daily dataset."
    )
    daily_df = None
else:
    try:
        daily_df = build_daily_dataset(
            nutrition_df=nutrition_df,
            garmin_daily_df=garmin_daily_df,
            garmin_activities_df=garmin_activities_df,
            hevy_sets_df=hevy_sets_df,
        )
        st.session_state["daily_df"] = daily_df
        st.dataframe(daily_df)

        st.caption("Rows per source:")
        st.write(
            {
                "nutrition_rows": 0 if nutrition_df is None else len(nutrition_df),
                "garmin_daily_rows": 0 if garmin_daily_df is None else len(garmin_daily_df),
                "garmin_activities_rows": 0
                if garmin_activities_df is None
                else len(garmin_activities_df),
                "hevy_sets_rows": 0 if hevy_sets_df is None else len(hevy_sets_df),
                "daily_rows": len(daily_df),
            }
        )
    except Exception as e:
        daily_df = None
        st.error(f"Daily aggregation error: {e}")

# =========================================================
#              TIER-1 ANALYTICS SECTION
# =========================================================
st.header("Tier-1 analytics")

daily_df = st.session_state.get("daily_df")

if daily_df is None or daily_df.empty:
    st.info("Generate the unified daily dataset first to see analytics.")
else:
    with st.expander("Debug: available daily_df columns", expanded=False):
        st.write(list(daily_df.columns))

    # ---- Targets for adherence ----
    st.subheader("Targets")
    col1, col2 = st.columns(2)
    with col1:
        target_calories = st.number_input(
            "Target daily calories (kcal)",
            min_value=1000,
            max_value=5000,
            value=2300,
            step=50,
        )
    with col2:
        target_protein = st.number_input(
            "Target daily protein (g)",
            min_value=40,
            max_value=300,
            value=160,
            step=5,
        )

    # ---- 1. Weekly energy balance ----
    st.subheader("1. Weekly energy balance")

    weekly_energy = build_weekly_energy_summary(daily_df)
    if weekly_energy.empty:
        st.info("No calories_in / calories_out columns found for energy balance.")
    else:
        st.dataframe(weekly_energy)
        if {"calories_in_avg", "calories_out_avg"}.issubset(weekly_energy.columns):
            chart_df = weekly_energy.set_index("week_start")[
                ["calories_in_avg", "calories_out_avg"]
            ]
            st.line_chart(chart_df)

        if "energy_balance_sum" in weekly_energy.columns:
            st.bar_chart(
                weekly_energy.set_index("week_start")[["energy_balance_sum"]]
            )

    # ---- 2. Macro adherence vs targets ----
    st.subheader("2. Macro adherence vs targets")

    macro_adherence = build_macro_adherence(
        daily_df, target_calories=target_calories, target_protein=target_protein
    )
    if macro_adherence.empty:
        st.info("No nutrition columns found for macro adherence.")
    else:
        st.dataframe(macro_adherence)

        adherence_summary = {
            "days_in_range_calories_±5%": int(
                macro_adherence["calories_within_5pct"].sum()
            )
            if "calories_within_5pct" in macro_adherence.columns
            else 0,
            "total_days_with_calories": int(
                macro_adherence["calories_within_5pct"].notna().sum()
            )
            if "calories_within_5pct" in macro_adherence.columns
            else 0,
            "days_meeting_protein": int(macro_adherence["protein_met"].sum())
            if "protein_met" in macro_adherence.columns
            else 0,
            "total_days_with_protein": int(
                macro_adherence["protein_met"].notna().sum()
            )
            if "protein_met" in macro_adherence.columns
            else 0,
        }
        st.caption("Adherence summary")
        st.write(adherence_summary)

        cin_col = COLS["calories_in"]
        prot_col = COLS["protein_g"]
        plot_cols = [c for c in [cin_col, prot_col] if c in macro_adherence.columns]
        if plot_cols:
            chart_df = macro_adherence.set_index("date")[plot_cols]
            st.line_chart(chart_df)

    # ---- 3. Weekly training load & frequency ----
    st.subheader("3. Weekly training load & frequency")

    training_summary = build_training_load_summary(daily_df)
    if training_summary.empty:
        st.info("No Hevy / steps columns found for training load.")
    else:
        st.dataframe(training_summary)

        ts_chart = training_summary.set_index("week_start")
        metric_cols = [
            c
            for c in ["hevy_volume_total", "hevy_sets_total", "steps_avg"]
            if c in ts_chart.columns
        ]
        if metric_cols:
            st.bar_chart(ts_chart[metric_cols])
