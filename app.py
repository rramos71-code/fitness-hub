import json
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

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

# Where we persist per-user goals locally
GOAL_STORE_PATH = Path("user_goals.json")


# =========================================================
# Goal helpers
# =========================================================
def _safe_parse_date(raw) -> date | None:
    if raw is None or raw == "":
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except Exception:
        return None


def _default_goal_dates() -> tuple[date, date | None]:
    start = date.today() - timedelta(days=28)
    return start, None


def load_goal_store() -> dict:
    if not GOAL_STORE_PATH.exists():
        return {}
    try:
        with GOAL_STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_goal_store(store: dict) -> None:
    try:
        with GOAL_STORE_PATH.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as exc:
        st.warning(f"Could not persist goals: {exc}")


def load_user_goal(user_id: str) -> dict:
    store = load_goal_store()
    raw = store.get(user_id) or {}

    start_default, end_default = _default_goal_dates()

    return {
        "target_calories": raw.get("target_calories", 2300.0),
        "target_protein": raw.get("target_protein", 140.0),
        "target_steps": raw.get("target_steps", 8000),
        "start_date": _safe_parse_date(raw.get("start_date")) or start_default,
        "end_date": _safe_parse_date(raw.get("end_date")) or end_default,
    }


def save_user_goal(user_id: str, goal: dict) -> None:
    store = load_goal_store()
    payload = goal.copy()
    payload["start_date"] = payload["start_date"].isoformat() if payload.get("start_date") else None
    payload["end_date"] = payload["end_date"].isoformat() if payload.get("end_date") else None
    store[user_id] = payload
    save_goal_store(store)


def apply_goal_columns(df: pd.DataFrame, goal: dict) -> pd.DataFrame:
    if df is None or df.empty or goal is None:
        return df

    target_cal = goal.get("target_calories") or 0
    target_protein = goal.get("target_protein") or 0
    target_steps = goal.get("target_steps") or 0
    start = goal.get("start_date")
    end = goal.get("end_date")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    mask = out["date"].dt.date >= start if start else True
    if isinstance(mask, bool):
        mask = pd.Series(mask, index=out.index)
    if end:
        mask = mask & (out["date"].dt.date <= end)
    out["goal_active"] = mask

    if target_cal > 0 and "calories_kcal" in out.columns:
        out["goal_calorie_delta"] = out["calories_kcal"] - target_cal
        out["goal_calorie_met"] = out["calories_kcal"].between(
            0.95 * target_cal, 1.05 * target_cal
        )
    else:
        out["goal_calorie_delta"] = pd.NA
        out["goal_calorie_met"] = False

    if target_protein > 0 and "protein_g" in out.columns:
        out["goal_protein_delta"] = out["protein_g"] - target_protein
        out["goal_protein_met"] = out["protein_g"] >= target_protein
    else:
        out["goal_protein_delta"] = pd.NA
        out["goal_protein_met"] = False

    if target_steps > 0 and "garmin_steps" in out.columns:
        out["goal_steps_delta"] = out["garmin_steps"] - target_steps
        out["goal_steps_met"] = out["garmin_steps"] >= target_steps
    else:
        out["goal_steps_delta"] = pd.NA
        out["goal_steps_met"] = False

    out.loc[~out["goal_active"], ["goal_calorie_met", "goal_protein_met", "goal_steps_met"]] = False
    return out


def compute_weekly_with_goals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["week_start"] = tmp["date"] - pd.to_timedelta(tmp["date"].dt.weekday, unit="D")

    agg = tmp.groupby("week_start").agg(
        avg_kcal=("calories_kcal", "mean"),
        avg_protein=("protein_g", "mean"),
        avg_steps=("garmin_steps", "mean"),
        goal_kcal_delta=("goal_calorie_delta", "mean"),
        goal_calorie_compliance=("goal_calorie_met", "mean"),
        goal_protein_delta=("goal_protein_delta", "mean"),
        goal_protein_compliance=("goal_protein_met", "mean"),
        goal_steps_compliance=("goal_steps_met", "mean"),
        days_with_data=("date", "count"),
    )

    agg = agg.reset_index().sort_values("week_start")
    agg["week"] = agg["week_start"].dt.strftime("%Y-%m-%d")
    return agg


def streak_lengths(series: pd.Series) -> tuple[int, int]:
    longest = 0
    current = 0
    for val in series.fillna(False).tolist():
        if bool(val):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return current, longest


def rolling_delta(series: pd.Series, window: int = 7) -> tuple[float, float] | None:
    """
    Compare the latest window average vs the previous window average.
    Returns (delta, pct_change) if we have enough data, otherwise None.
    """
    clean = series.dropna()
    if len(clean) < window * 2:
        return None
    recent = clean.tail(window).mean()
    prev = clean.tail(window * 2).head(window).mean()
    if pd.isna(prev) or prev == 0:
        pct = 0.0
    else:
        pct = (recent - prev) / prev
    return recent - prev, pct


def build_text_pdf(title: str, body_lines: list[str]) -> bytes:
    """Minimal single-page PDF generator for tabular summaries without extra deps."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    lines = [title, ""] + body_lines
    y = 770
    chunks = []
    for line in lines:
        safe = escape(line)
        chunks.append(f"BT /F1 12 Tf 40 {y} Td ({safe}) Tj ET")
        y -= 16
        if y <= 40:
            break  # keep it single-page

    stream = "\n".join(chunks)
    stream_bytes = stream.encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append(b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj")
    objects.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj"
    )
    objects.append(
        f"4 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("latin-1")
        + stream_bytes
        + b"\nendstream endobj"
    )
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj + b"\n"

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += b"trailer\n<< /Size " + str(len(objects) + 1).encode("latin-1") + b" /Root 1 0 R >>\n"
    pdf += b"startxref\n" + str(xref_pos).encode("latin-1") + b"\n%%EOF"
    return pdf


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
    garmin_activities_df: pd.DataFrame | None,
    hevy_sets_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Unify per-day DataFrame across Google Fit, Garmin, and Hevy."""
    frames = []

    if nutrition_df is not None and not nutrition_df.empty:
        n = ensure_date_column(nutrition_df)
        if n is not None and not n.empty and "date" in n.columns:
            n = n.copy()
            n["date"] = pd.to_datetime(n["date"]).dt.date
            agg_cols = {
                "calories_kcal": "sum",
                "protein_g": "sum",
                "carbs_g": "sum",
                "fat_g": "sum",
            }
            existing = {k: v for k, v in agg_cols.items() if k in n.columns}
            n = n.groupby("date", as_index=False).agg(existing)
            frames.append(n)

    if garmin_daily_df is not None and not garmin_daily_df.empty:
        g = ensure_date_column(garmin_daily_df)
        if g is not None and not g.empty and "date" in g.columns:
            g = g.copy()
            g["date"] = pd.to_datetime(g["date"]).dt.date

            cal_col = _first_existing(
                g, ["caloriesTotal", "calories_total", "activeKilocalories", "calories"]
            )
            steps_col = _first_existing(g, ["steps", "stepCount", "totalSteps"])
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
                    g_day.rename(columns={cal_col: "garmin_calories_kcal"}, inplace=True)
                if steps_col:
                    g_day.rename(columns={steps_col: "garmin_steps"}, inplace=True)
                if "sleep_hours_tmp" in g_day.columns:
                    g_day.rename(columns={"sleep_hours_tmp": "garmin_sleep_hours"}, inplace=True)

                frames.append(g_day)

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

    daily = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), frames)
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

    if (sets_col not in df.columns and volume_col not in df.columns and steps_col not in df.columns):
        return pd.DataFrame()

    df["is_hevy_day"] = df[sets_col] > 0 if sets_col in df.columns else False
    df["is_high_step_day"] = df[steps_col] >= 8000 if steps_col in df.columns else False
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
st.title("Fitness Hub - Core Integrations Test")

# ----------------------- Goals ----------------------------
st.header("User and Goals")

user_id_input = st.text_input("User id (used to save goals)", value=st.session_state.get("user_id", "default"))
user_id = user_id_input.strip() or "default"
st.session_state["user_id"] = user_id

loaded_goal = load_user_goal(user_id)
if "goal_config" not in st.session_state:
    st.session_state["goal_config"] = loaded_goal
current_goal = st.session_state.get("goal_config", loaded_goal)

with st.form("goal_form"):
    c_goal1, c_goal2, c_goal3 = st.columns(3)
    with c_goal1:
        target_cal = st.number_input(
            "Daily calorie target (kcal)",
            min_value=1000.0,
            max_value=6000.0,
            value=float(current_goal.get("target_calories", loaded_goal["target_calories"])),
            step=50.0,
        )
    with c_goal2:
        target_protein = st.number_input(
            "Daily protein target (g)",
            min_value=20.0,
            max_value=400.0,
            value=float(current_goal.get("target_protein", loaded_goal["target_protein"])),
            step=5.0,
        )
    with c_goal3:
        target_steps = st.number_input(
            "Daily steps target",
            min_value=1000,
            max_value=30000,
            value=int(current_goal.get("target_steps", loaded_goal["target_steps"])),
            step=500,
        )

    start_date_input = st.date_input(
        "Goal start date",
        value=current_goal.get("start_date") or loaded_goal["start_date"],
    )
    use_end_date = st.checkbox("Set an end date", value=current_goal.get("end_date") is not None)
    end_date_input = None
    if use_end_date:
        default_end = current_goal.get("end_date") or (start_date_input + timedelta(days=28))
        end_date_input = st.date_input("Goal end date", value=default_end, min_value=start_date_input)

    submit_goal = st.form_submit_button("Save goals")
    if submit_goal:
        goal_payload = {
            "target_calories": target_cal,
            "target_protein": target_protein,
            "target_steps": target_steps,
            "start_date": start_date_input,
            "end_date": end_date_input if use_end_date else None,
        }
        save_user_goal(user_id, goal_payload)
        st.session_state["goal_config"] = goal_payload
        current_goal = goal_payload
        st.success(f"Goals saved for {user_id}")

st.caption(
    f"Active goal window: {current_goal.get('start_date')} to "
    f"{current_goal.get('end_date') or 'open'} | "
    f"{current_goal.get('target_calories')} kcal, "
    f"{current_goal.get('target_protein')} g protein, "
    f"{current_goal.get('target_steps')} steps per day."
)

# ----------------------- Personalization ----------------------------
st.subheader("Personalization (used for training load guidance)")

c_body, c_rpe = st.columns(2)

with c_body:
    body_weight = st.number_input(
        "Body weight (kg)",
        min_value=30.0,
        max_value=250.0,
        value=float(st.session_state.get("body_weight", 75.0)),
        step=0.5,
    )
    st.session_state["body_weight"] = body_weight

with c_rpe:
    rpe_raw_default = st.session_state.get("rpe_log_raw", "7,8,7")
    rpe_raw = st.text_input(
        "Recent session RPEs (comma separated)",
        value=rpe_raw_default,
        help="Add your last few lifting sessions (e.g. 7,7.5,8,8.5)",
    )
    st.session_state["rpe_log_raw"] = rpe_raw

def _parse_rpe(raw: str) -> list[float]:
    vals = []
    for chunk in str(raw).split(","):
        try:
            vals.append(float(chunk.strip()))
        except Exception:
            continue
    return [v for v in vals if 0 <= v <= 10]

rpe_log = _parse_rpe(rpe_raw)
st.session_state["rpe_log"] = rpe_log

if rpe_log:
    st.caption(f"RPE entries saved: {len(rpe_log)} | Avg RPE: {sum(rpe_log)/len(rpe_log):.1f}")
else:
    st.caption("No RPE entries yet. Add a few to tailor recovery guidance.")

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

        if current_goal:
            daily_df = apply_goal_columns(daily_df, current_goal)

        st.session_state["daily_df"] = daily_df

        st.dataframe(daily_df)

        st.caption("Rows per source:")
        st.write(
            {
                "nutrition_rows": 0 if nutrition_df is None else len(nutrition_df),
                "garmin_daily_rows": 0 if garmin_daily_df is None else len(garmin_daily_df),
                "garmin_activities_rows": 0 if garmin_activities_df is None else len(garmin_activities_df),
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
st.header("Tier-1 analytics")

daily_df = st.session_state.get("daily_df")

if daily_df is None or daily_df.empty:
    st.info("Generate the unified daily dataset first to see analytics.")
else:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    goal_cfg = st.session_state.get("goal_config")
    if goal_cfg:
        df = apply_goal_columns(df, goal_cfg)

    calories = df.get("calories_kcal", pd.Series(dtype=float)).fillna(0.0)
    g_calories = df.get("garmin_calories_kcal", pd.Series(dtype=float)).fillna(0.0)
    protein = df.get("protein_g", pd.Series(dtype=float)).fillna(0.0)
    carbs = df.get("carbs_g", pd.Series(dtype=float)).fillna(0.0)
    fat = df.get("fat_g", pd.Series(dtype=float)).fillna(0.0)
    steps = df.get("garmin_steps", pd.Series(dtype=float)).fillna(0.0)

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

    st.subheader("Weekly summaries")

    weekly = compute_weekly_with_goals(df)
    if weekly.empty:
        st.info("Weekly summaries will appear once you have logged data.")
    else:
        weekly_view = weekly[
            [
                "week",
                "avg_kcal",
                "avg_protein",
                "avg_steps",
                "goal_kcal_delta",
                "goal_calorie_compliance",
                "goal_protein_compliance",
                "goal_steps_compliance",
            ]
        ].copy()
        weekly_view = weekly_view.round(
            {
                "avg_kcal": 1,
                "avg_protein": 1,
                "avg_steps": 1,
                "goal_kcal_delta": 1,
                "goal_calorie_compliance": 2,
                "goal_protein_compliance": 2,
                "goal_steps_compliance": 2,
            }
        )
        weekly_view.rename(
            columns={
                "goal_kcal_delta": "kcal delta vs goal",
                "goal_calorie_compliance": "calorie goal hit rate",
                "goal_protein_compliance": "protein goal hit rate",
                "goal_steps_compliance": "steps goal hit rate",
            },
            inplace=True,
        )
        st.dataframe(weekly_view, use_container_width=True)

        csv_bytes = weekly_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download weekly summary (CSV)",
            data=csv_bytes,
            file_name="weekly_summary.csv",
            mime="text/csv",
        )

        text_table = weekly_view.to_string(index=False).splitlines()
        pdf_bytes = build_text_pdf("Weekly summary", text_table)
        st.download_button(
            "Download weekly summary (PDF)",
            data=pdf_bytes,
            file_name="weekly_summary.pdf",
            mime="application/pdf",
        )

    st.subheader("Trends")

    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("**Calories over time**")
        chart_df = df[["date"]].copy()
        chart_df["intake_kcal"] = calories
        chart_df["garmin_kcal"] = g_calories
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

    st.subheader("Best and worst days")

    if (calories > 0).any():
        best_kcal_day = df.loc[calories.idxmin(), "date"].date()
        worst_kcal_day = df.loc[calories.idxmax(), "date"].date()
        st.write(f"- Lowest calorie day: {best_kcal_day} with {calories.min():.0f} kcal")
        st.write(f"- Highest calorie day: {worst_kcal_day} with {calories.max():.0f} kcal")

    if (steps > 0).any():
        best_steps_day = df.loc[steps.idxmax(), "date"].date()
        st.write(f"- Highest steps day: {best_steps_day} with {steps.max():,.0f} steps")

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

    goal_cfg = st.session_state.get("goal_config")
    if goal_cfg:
        df = apply_goal_columns(df, goal_cfg)

    kcal_in = df["calories_kcal"].fillna(0)
    kcal_out = df["garmin_calories_kcal"].fillna(0)
    protein = df["protein_g"].fillna(0)
    steps = df["garmin_steps"].fillna(0)
    volume = df["hevy_volume_kg"].fillna(0) if "hevy_volume_kg" in df.columns else pd.Series(dtype=float)

    last7 = df.tail(7)
    last14 = df.tail(14)

    avg_in_last7 = last7["calories_kcal"].mean() if not last7.empty else 0
    avg_steps_last7 = last7["garmin_steps"].mean() if not last7.empty else 0
    avg_volume_last7 = last7["hevy_volume_kg"].mean() if "hevy_volume_kg" in last7.columns else 0

    body_weight = st.session_state.get("body_weight")
    rpe_log = st.session_state.get("rpe_log", [])
    avg_rpe = sum(rpe_log) / len(rpe_log) if rpe_log else None

    st.subheader("Insights Feed")

    insight_cards: list[dict] = []

    def add_card(category: str, title: str, value: str, *, delta: str | None = None, description: str | None = None):
        insight_cards.append(
            {
                "category": category,
                "title": title,
                "value": value,
                "delta": delta,
                "description": description,
            }
        )

    if "goal_active" in df.columns and df["goal_active"].any():
        recent_goal_days = last7[last7["goal_active"]]
        if not recent_goal_days.empty:
            cal_comp = recent_goal_days["goal_calorie_met"].mean()
            cal_delta = recent_goal_days["goal_calorie_delta"].mean()
            _, long_streak = streak_lengths(recent_goal_days["goal_calorie_met"])
            add_card(
                "Goals",
                "Calorie goal hit rate",
                f"{cal_comp * 100:.0f}%",
                delta=f"{cal_delta:+.0f} kcal vs target",
                description=f"Longest streak: {long_streak} days",
            )

            deficit_streak, _ = streak_lengths(recent_goal_days["goal_calorie_delta"] < -150)
            surplus_streak, _ = streak_lengths(recent_goal_days["goal_calorie_delta"] > 150)
            if deficit_streak >= 3:
                add_card("Goals", "Deficit streak", f"{deficit_streak} days", description="Running calorie deficit vs goal.")
            if surplus_streak >= 3:
                add_card("Goals", "Surplus streak", f"{surplus_streak} days", description="Running calorie surplus vs goal.")

        if "goal_protein_met" in df.columns and not last7.empty:
            protein_comp = last7[last7["goal_active"]]["goal_protein_met"].mean()
            add_card(
                "Goals",
                "Protein goal hit rate",
                f"{protein_comp * 100:.0f}%",
                description="Share of goal-active days hitting protein in last week",
            )

    if len(last7) >= 5:
        kcal_delta = rolling_delta(df["calories_kcal"], window=3)
        if kcal_delta:
            delta, pct = kcal_delta
            direction = "▲" if delta > 0 else "▼"
            add_card(
                "Energy",
                "3-day calorie trend",
                f"{direction} {abs(delta):.0f} kcal",
                delta=f"{pct:+.0%} vs prior window",
            )

    if (last7["calories_kcal"] > 0).sum() >= 3:
        avg_out = last7["garmin_calories_kcal"].mean() if (last7["garmin_calories_kcal"] > 0).any() else None
        if avg_out:
            net = avg_in_last7 - avg_out
            balance_label = "deficit" if net < -200 else "surplus" if net > 200 else "near even"
            add_card(
                "Energy",
                "Weekly energy balance",
                f"{net:+.0f} kcal/day",
                description=f"Last 7d average ({balance_label})",
            )

    if (steps > 0).any():
        steps_delta = rolling_delta(df["garmin_steps"].fillna(0), window=3)
        if steps_delta:
            delta, pct = steps_delta
            direction = "▲" if delta > 0 else "▼"
            add_card(
                "Activity",
                "3-day steps trend",
                f"{direction} {abs(delta):,.0f}",
                delta=f"{pct:+.0%} vs prior window",
            )

        prev7 = df.tail(14).head(7)
        if (prev7["garmin_steps"] > 0).any():
            delta = avg_steps_last7 - prev7["garmin_steps"].mean()
            trend = "▲" if delta > 0 else "▼" if delta < 0 else "→"
            add_card(
                "Activity",
                "Steps week-over-week",
                f"{avg_steps_last7:,.0f} steps",
                delta=f"{trend} {delta:+,.0f} vs prior week",
            )

    if (protein > 0).any():
        low_protein_days = (protein < 90).sum()
        if low_protein_days >= 3:
            add_card("Nutrition", "Low protein days", str(int(low_protein_days)), description="Logged under 90g protein")

    if len(last7) >= 3 and (volume > 0).any():
        vol_delta = rolling_delta(df["hevy_volume_kg"].fillna(0), window=3)
        if vol_delta:
            delta, pct = vol_delta
            direction = "▲" if delta > 0 else "▼"
            add_card(
                "Training",
                "Lifting volume trend",
                f"{direction} {abs(delta):.0f} kg",
                delta=f"{pct:+.0%} vs prior window",
            )

        add_card(
            "Training",
            "Avg lifting volume (7d)",
            f"{avg_volume_last7:,.0f} kg/day",
            description="Based on last 7 logged days",
        )

    if avg_rpe:
        if avg_rpe >= 8:
            add_card("Recovery", "Average RPE", f"{avg_rpe:.1f}", description="High effort—prioritize recovery")
        elif avg_rpe <= 6.5:
            add_card("Recovery", "Average RPE", f"{avg_rpe:.1f}", description="Effort is comfortable; progression possible")

    if (kcal_out > 0).any():
        heavy_days = df[df["garmin_calories_kcal"] > 600]
        if not heavy_days.empty:
            underfed = heavy_days[heavy_days["calories_kcal"] < heavy_days["garmin_calories_kcal"] - 300]
            if not underfed.empty:
                d = underfed.iloc[-1]["date"].date()
                add_card("Energy", "High burn day", f"{d}", description="Burn exceeded intake by >300 kcal")

    weekend = df[df["date"].dt.weekday >= 5]
    if not weekend.empty:
        w_kcal = weekend["calories_kcal"].mean()
        wd_kcal = df[df["date"].dt.weekday < 5]["calories_kcal"].mean()
        delta = w_kcal - wd_kcal
        add_card(
            "Habits",
            "Weekend intake",
            f"{w_kcal:,.0f} kcal",
            delta=f"{delta:+.0f} vs weekdays",
        )

    if not insight_cards:
        st.info("No significant patterns detected yet.")
    else:
        categories = sorted({c["category"] for c in insight_cards})
        for cat in categories:
            st.markdown(f"**{cat}**")
            cat_cards = [c for c in insight_cards if c["category"] == cat]
            for i in range(0, len(cat_cards), 3):
                cols = st.columns(min(3, len(cat_cards) - i))
                for col, card in zip(cols, cat_cards[i : i + 3]):
                    with col:
                        st.metric(card["title"], card["value"], delta=card.get("delta"))
                        if card.get("description"):
                            st.caption(card["description"])

    st.subheader("Health & Performance Flags")

    flags = []

    if "goal_active" in df.columns and df["goal_active"].any():
        goal_days = last7[last7["goal_active"]]
        cal_comp = goal_days["goal_calorie_met"].mean() if not goal_days.empty else 0
        prot_comp = goal_days["goal_protein_met"].mean() if not goal_days.empty else 0
        steps_comp = goal_days["goal_steps_met"].mean() if not goal_days.empty else 0

        if cal_comp < 0.35:
            flags.append(("red", "Calorie goal rarely met over the last week."))
        elif cal_comp < 0.6:
            flags.append(("amber", "Calorie goal compliance is moderate."))

        if prot_comp < 0.4:
            flags.append(("amber", "Protein goal often missed."))

        if steps_comp < 0.5:
            flags.append(("amber", "Steps goal often missed."))

        if cal_comp >= 0.7 and prot_comp >= 0.7 and steps_comp >= 0.7:
            flags.append(("green", "Strong compliance across calorie, protein, and steps goals."))
    else:
        if avg_in_last7 < 1400:
            flags.append(("red", "Daily intake very low; risk of under-fueling."))
        if protein.mean() < 90:
            flags.append(("red", "Protein intake consistently low; prioritize protein-rich meals."))
        if avg_steps_last7 < 6000:
            flags.append(("amber", "Low activity trend; steps below 6000 on average."))
        if avg_in_last7 > 1800 and protein.mean() > 120:
            flags.append(("green", "Nutrition profile supports strength and recovery."))

    if avg_rpe and avg_rpe >= 8.5:
        flags.append(("amber", f"Average RPE is high ({avg_rpe:.1f}); watch fatigue and sleep."))
    if body_weight and body_weight > 0 and avg_volume_last7 > 0:
        rel_weekly = (last7["hevy_volume_kg"].sum() if "hevy_volume_kg" in last7.columns else 0) / body_weight
        if rel_weekly > 180:
            flags.append(("amber", f"Heavy relative lifting load this week (~{rel_weekly:.0f} kg/kg bodyweight)."))

    for level, msg in flags:
        if level == "red":
            st.error(msg)
        elif level == "amber":
            st.warning(msg)
        elif level == "green":
            st.success(msg)

    st.subheader("Patterns Detected")

    patterns = []

    if "goal_calorie_delta" in df.columns:
        surplus_run = streak_lengths(df["goal_calorie_delta"] > 200)[0]
        deficit_run = streak_lengths(df["goal_calorie_delta"] < -200)[0]
        if surplus_run >= 3:
            patterns.append("Calorie surplus streak detected (3+ days above goal).")
        if deficit_run >= 3:
            patterns.append("Calorie deficit streak detected (3+ days below goal).")

    lowp = (df["protein_g"] < 100).rolling(3).sum()
    if (lowp == 3).any():
        patterns.append("3-day low protein streak detected.")

    if (df["calories_kcal"] == 0).sum() >= 2:
        patterns.append("Several days with no nutrition logged; insights may be incomplete.")

    if df["garmin_steps"].std() > 5000:
        patterns.append("High variability in step count; daily movement inconsistent.")

    if not patterns:
        st.info("No notable patterns this week.")
    else:
        for p in patterns:
            st.write(f"- {p}")
