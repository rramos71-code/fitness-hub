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

# =================== Unified daily view ==================
st.subheader("Daily overview (unified dataset)")

nutrition_df = st.session_state.get("googlefit_nutrition_df")
garmin_daily_df = st.session_state.get("garmin_daily_df")
garmin_activities_df = st.session_state.get("garmin_activities_df")
hevy_sets_df = st.session_state.get("hevy_sets_df")

if (
    (nutrition_df is None or nutrition_df.empty)
    and (garmin_daily_df is None or garmin_daily_df.empty)
    and (garmin_activities_df is None or garmin_activities_df.empty)
    and (hevy_sets_df is None or hevy_sets_df.empty)
):
    st.info(
        "Load data from at least one source (Google Fit, Garmin, Hevy) "
        "to see the unified daily dataset."
    )
    st.session_state["daily_df"] = None
else:
    try:
        daily_df = build_daily_dataset(
            nutrition_df=nutrition_df,
            garmin_daily_df=garmin_daily_df,
            garmin_activities_df=garmin_activities_df,
            hevy_sets_df=hevy_sets_df,
        )

        # store for Tier-1 analytics
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
        st.session_state["daily_df"] = None
        st.error(f"Daily aggregation error: {e}")

# =========================================================
#              TIER-1 ANALYTICS SECTION
# =========================================================
# ====================== Tier-1 analytics =====================
st.header("Tier-1 analytics")

daily_df = st.session_state.get("daily_df")

if daily_df is None or daily_df.empty:
    st.info("Generate the unified daily dataset first to see analytics.")
else:
    # Work on a copy and normalize date dtype
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Convenience series with NaN handled
    calories = df.get("calories_kcal", pd.Series(dtype=float)).fillna(0.0)
    g_calories = df.get("garmin_calories_kcal", pd.Series(dtype=float)).fillna(0.0)
    protein = df.get("protein_g", pd.Series(dtype=float)).fillna(0.0)
    carbs = df.get("carbs_g", pd.Series(dtype=float)).fillna(0.0)
    fat = df.get("fat_g", pd.Series(dtype=float)).fillna(0.0)
    steps = df.get("garmin_steps", pd.Series(dtype=float)).fillna(0.0)

    # 1 - KPI snapshot
    st.subheader("KPI snapshot")

    total_days = len(df)
    tracked_days = int((calories > 0).sum())
    avg_kcal = calories[calories > 0].mean() if (calories > 0).any() else 0
    avg_protein = protein[protein > 0].mean() if (protein > 0).any() else 0
    avg_steps = steps[steps > 0].mean() if (steps > 0).any() else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Tracked nutrition days", f"{tracked_days} / {total_days}")
    with c2:
        st.metric("Avg intake (kcal)", f"{avg_kcal:,.0f}")
    with c3:
        st.metric("Avg protein (g)", f"{avg_protein:,.0f}")
    with c4:
        st.metric("Avg steps per day", f"{avg_steps:,.0f}")

    # 2 - Weekly summaries
    st.subheader("Weekly summaries")

    weekly = (
        df.set_index("date")
        .resample("W-MON")  # weeks ending on Monday (so week label is Monday)
        .agg(
            {
                "calories_kcal": "mean",
                "protein_g": "mean",
                "carbs_g": "mean",
                "fat_g": "mean",
                "garmin_calories_kcal": "mean",
                "garmin_steps": "mean",
            }
        )
        .rename(
            columns={
                "calories_kcal": "avg_kcal",
                "protein_g": "avg_protein",
                "carbs_g": "avg_carbs",
                "fat_g": "avg_fat",
                "garmin_calories_kcal": "avg_garmin_kcal",
                "garmin_steps": "avg_steps",
            }
        )
        .reset_index()
    )

    # Make week label nice
    weekly["week"] = weekly["date"].dt.strftime("%Y-W%U")
    weekly_view = weekly[
        ["week", "avg_kcal", "avg_protein", "avg_carbs", "avg_fat", "avg_steps"]
    ].round(1)

    st.dataframe(weekly_view, use_container_width=True)

    # 3 - Time series charts
    st.subheader("Trends")

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("**Calories and steps over time**")
        chart_df = df[["date"]].copy()
        chart_df["intake_kcal"] = calories
        chart_df["garmin_kcal"] = g_calories
        chart_df["steps"] = steps
        chart_df = chart_df.set_index("date")
        st.line_chart(chart_df)

    with chart_cols[1]:
        st.markdown("**Macros over time**")
        macro_df = df[["date"]].copy()
        macro_df["protein_g"] = protein
        macro_df["carbs_g"] = carbs
        macro_df["fat_g"] = fat
        macro_df = macro_df.set_index("date")
        st.line_chart(macro_df)

    # 4 - Basic correlations
    st.subheader("Basic correlations")

    corr_rows = []

    if len(calories[calories > 0]) >= 3 and len(steps[steps > 0]) >= 3:
        corr_steps_kcal = calories.corr(steps)
        corr_rows.append(
            {"pair": "Calories vs steps", "correlation": corr_steps_kcal}
        )

    if len(calories[calories > 0]) >= 3 and len(g_calories[g_calories > 0]) >= 3:
        corr_intake_vs_garmin = calories.corr(g_calories)
        corr_rows.append(
            {"pair": "Intake kcal vs Garmin kcal", "correlation": corr_intake_vs_garmin}
        )

    # Optional - if you later add training volume columns from Hevy,
    # you can extend correlations here, guarded by column checks.

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_df["correlation"] = corr_df["correlation"].round(2)
        st.table(corr_df)
    else:
        st.info("Not enough overlapping data to compute correlations yet.")

    # 5 - Best and worst days
    st.subheader("Best and worst days")

    if (calories > 0).any():
        best_kcal_day = df.loc[calories.idxmin(), "date"].date()
        worst_kcal_day = df.loc[calories.idxmax(), "date"].date()
        st.write(
            f"- Lowest calorie day: **{best_kcal_day}** with {calories.min():.0f} kcal"
        )
        st.write(
            f"- Highest calorie day: **{worst_kcal_day}** with {calories.max():.0f} kcal"
        )

    if (steps > 0).any():
        best_steps_day = df.loc[steps.idxmax(), "date"].date()
        st.write(
            f"- Highest steps day: **{best_steps_day}** with {steps.max():,.0f} steps"
        )

    # 6 - Debug helper
    with st.expander("Debug: available daily_df columns"):
        st.write(list(df.columns))


# ====================== Tier-2 Insights (Intelligent Patterns) ======================
st.header("Tier-2 Insights")

daily_df = st.session_state.get("daily_df")

if daily_df is None or daily_df.empty:
    st.info("Load the unified daily dataset first to generate insights.")
else:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Extract useful series
    kcal_in = df["calories_kcal"].fillna(0)
    kcal_out = df["garmin_calories_kcal"].fillna(0)
    protein = df["protein_g"].fillna(0)
    steps = df["garmin_steps"].fillna(0)

    # Rolling windows
    last7 = df.tail(7)
    last14 = df.tail(14)

    # -----------------------------
    # HIGH-LEVEL INSIGHTS FEED
    # -----------------------------
    st.subheader("Insights Feed")

    insights = []

    # 1. Energy balance trend
    if (last7["calories_kcal"] > 0).sum() >= 3:
        avg_in = last7["calories_kcal"].mean()
        avg_out = last7["garmin_calories_kcal"].mean() if (last7["garmin_calories_kcal"] > 0).any() else None

        if avg_out:
            net = avg_in - avg_out
            if net < -200:
                insights.append(f"You're in a consistent **weekly deficit** (~{abs(net):.0f} kcal/day). Suitable for fat loss.")
            elif net > 200:
                insights.append(f"You're in a **weekly surplus** (~{net:.0f} kcal/day). Suitable for muscle gain.")
            else:
                insights.append("Your weekly energy balance is **near maintenance**.")

    # 2. Protein compliance
    if (last7["protein_g"] > 0).any():
        p_mean = last7["protein_g"].mean()
        days_high_protein = (last7["protein_g"] >= 130).sum()  # example threshold
        insights.append(
            f"Protein intake averages **{p_mean:.0f} g/day**, with {days_high_protein}/7 days above target."
        )

    # 3. Steps trend
    if (last7["garmin_steps"] > 0).any():
        steps_mean = last7["garmin_steps"].mean()
        prev7 = df.tail(14).head(7)
        if (prev7["garmin_steps"] > 0).any():
            delta = steps_mean - prev7["garmin_steps"].mean()
            if delta > 1000:
                insights.append("Your daily steps are **trending upward** compared to last week.")
            elif delta < -1000:
                insights.append("Your daily steps are **trending downward** compared to last week.")
            else:
                insights.append("Steps are stable week-over-week.")

    # 4. High-burn days and fueling mismatch
    if (kcal_out > 0).any():
        heavy_days = df[df["garmin_calories_kcal"] > 600]
        if not heavy_days.empty:
            # look for underfed days among them
            underfed = heavy_days[heavy_days["calories_kcal"] < heavy_days["garmin_calories_kcal"] - 300]
            if not underfed.empty:
                d = underfed.iloc[-1]["date"].date()
                insights.append(
                    f"On **{d}**, you had a high-burn day with **insufficient fueling**. Consider increasing carbs on such days."
                )

    # 5. Weekend pattern detection
    weekend = df[df["date"].dt.weekday >= 5]
    if not weekend.empty:
        w_kcal = weekend["calories_kcal"].mean()
        wd_kcal = df[df["date"].dt.weekday < 5]["calories_kcal"].mean()
        if w_kcal < wd_kcal - 200:
            insights.append("You tend to **undereat on weekends** compared to weekdays.")
        elif w_kcal > wd_kcal + 200:
            insights.append("You tend to **overeat on weekends** compared to weekdays.")

    # Display insights
    if not insights:
        st.info("No significant patterns detected yet.")
    else:
        for insight in insights:
            st.markdown(f"• {insight}")

    # -----------------------------
    # FLAGS (Red / Amber / Green)
    # -----------------------------
    st.subheader("Health & Performance Flags")

    flags = []

    # Red flags
    if avg_in < 1400:
        flags.append(("red", "Daily intake very low — risk of under-fueling."))
    if protein.mean() < 90:
        flags.append(("red", "Protein intake consistently low — prioritize protein-rich meals."))

    # Amber flags
    if steps_mean < 6000:
        flags.append(("amber", "Low activity trend — steps have been below 6000 on average."))

    # Green flags
    if avg_in > 1800 and protein.mean() > 120:
        flags.append(("green", "Nutrition profile supports strength and recovery."))

    # Render flags
    for level, msg in flags:
        if level == "red":
            st.error(msg)
        elif level == "amber":
            st.warning(msg)
        elif level == "green":
            st.success(msg)

    # -----------------------------
    # Special patterns
    # -----------------------------
    st.subheader("Patterns Detected")

    patterns = []

    # Low protein streak
    lowp = (df["protein_g"] < 100).rolling(3).sum()
    if (lowp == 3).any():
        patterns.append("3-day **low protein streak** detected.")

    # Calorie under-reporting days
    if (df["calories_kcal"] == 0).sum() >= 2:
        patterns.append("Several days with **no nutrition logged** — insights may be incomplete.")

    # Steps inconsistency
    if df["garmin_steps"].std() > 5000:
        patterns.append("High variability in step count — inconsistent daily movement.")

    if not patterns:
        st.info("No notable patterns this week.")
    else:
        for p in patterns:
            st.write(f"• {p}")
