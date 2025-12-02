import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

from hevy_api.client import HevyClient
from garminconnect import Garmin, GarminConnectAuthenticationError


# ------------- Feature flags (F1 - F10) -------------

FEATURE_FLAGS = {
    "F1": True,   # Hevy set level flattening
    "F2": True,   # Workout template mapping
    "F3": True,   # Basic progression rules per exercise
    "F4": True,   # Next workout recommendation tables
    "F5": True,   # Weekly volume and intensity summary
    "F6": True,   # Plateau detection
    "F7": True,   # Deload and variation suggestions
    "F8": True,   # Readiness indicator (basic)
    "F9": True,   # Goal specific presets
    "F10": True,  # Exercise PR tracking
}


# ------------- Session keys -------------

HEVY_WORKOUTS_RAW_KEY = "hevy_workouts_raw"
HEVY_WORKOUTS_DF_KEY = "hevy_workouts_df"
HEVY_SETS_DF_KEY = "hevy_sets_df"
GARMIN_DF_KEY = "garmin_df"
SETTINGS_KEY = "settings"


# ------------- Goal presets (F9) -------------

GOAL_PRESETS = {
    "recomp": {
        "name": "Recomposition",
        "rep_range_low": 6,
        "rep_range_high": 10,
        "kg_step": 2.5,
        "plateau_weeks": 4,
    },
    "hypertrophy": {
        "name": "Hypertrophy",
        "rep_range_low": 8,
        "rep_range_high": 12,
        "kg_step": 2.5,
        "plateau_weeks": 3,
    },
    "strength": {
        "name": "Strength",
        "rep_range_low": 3,
        "rep_range_high": 6,
        "kg_step": 5.0,
        "plateau_weeks": 5,
    },
    "maintenance": {
        "name": "Maintenance",
        "rep_range_low": 5,
        "rep_range_high": 10,
        "kg_step": 0.0,
        "plateau_weeks": 6,
    },
}


# ------------- Init session state -------------

def init_session_state():
    if HEVY_WORKOUTS_RAW_KEY not in st.session_state:
        st.session_state[HEVY_WORKOUTS_RAW_KEY] = []
    if HEVY_WORKOUTS_DF_KEY not in st.session_state:
        st.session_state[HEVY_WORKOUTS_DF_KEY] = pd.DataFrame()
    if HEVY_SETS_DF_KEY not in st.session_state:
        st.session_state[HEVY_SETS_DF_KEY] = pd.DataFrame()
    if GARMIN_DF_KEY not in st.session_state:
        st.session_state[GARMIN_DF_KEY] = pd.DataFrame()
    if SETTINGS_KEY not in st.session_state:
        st.session_state[SETTINGS_KEY] = {
            "goal_type": "recomp",
        }


# ------------- Settings sidebar (F9) -------------

def sidebar_settings():
    settings = st.session_state[SETTINGS_KEY]

    st.sidebar.header("Training settings")

    goal_type = st.sidebar.selectbox(
        "Primary goal",
        options=list(GOAL_PRESETS.keys()),
        format_func=lambda k: GOAL_PRESETS[k]["name"],
        index=list(GOAL_PRESETS.keys()).index(settings.get("goal_type", "recomp")),
    )

    st.session_state[SETTINGS_KEY]["goal_type"] = goal_type

    preset = GOAL_PRESETS[goal_type]
    st.sidebar.markdown(
        f"**Preset**: {preset['name']}  \n"
        f"Target reps: {preset['rep_range_low']} - {preset['rep_range_high']}  \n"
        f"Weight step: {preset['kg_step']} kg  \n"
        f"Plateau detection: {preset['plateau_weeks']} weeks"
    )


def get_current_goal_config():
    goal_type = st.session_state[SETTINGS_KEY].get("goal_type", "recomp")
    return GOAL_PRESETS.get(goal_type, GOAL_PRESETS["recomp"])


# ------------- Hevy helpers (connection) -------------

def get_hevy_client() -> HevyClient:
    api_key = st.secrets.get("HEVY_API_KEY", None)
    if not api_key:
        api_key = os.getenv("HEVY_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Hevy API key not found. Define HEVY_API_KEY in Streamlit secrets "
            "or as an environment variable."
        )

    os.environ["HEVY_API_KEY"] = api_key
    return HevyClient()


def hevy_workouts_to_df(workouts) -> pd.DataFrame:
    """Flatten top level workouts (not sets) into a simple DataFrame."""
    records = []
    for w in workouts:
        if hasattr(w, "model_dump"):
            data = w.model_dump()
        elif hasattr(w, "dict"):
            data = w.dict()
        else:
            data = getattr(w, "__dict__", None)

        if not data:
            continue

        records.append(data)

    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    return df


# ------------- Hevy set level flattening (F1) -------------

def _first_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def hevy_workouts_to_sets_df(workouts) -> pd.DataFrame:
    """
    F1: Flatten Hevy workouts into a per set DataFrame.

    Columns (best effort):
    date, workout_id, workout_name, template_name, exercise_name, set_index,
    set_type, weight_kg, reps, rir, tempo, notes
    """
    rows = []

    for w in workouts:
        if hasattr(w, "model_dump"):
            wd = w.model_dump()
        elif hasattr(w, "dict"):
            wd = w.dict()
        else:
            wd = getattr(w, "__dict__", None)

        if not wd:
            continue

        workout_id = wd.get("id") or wd.get("workout_id") or wd.get("uuid")
        workout_name = wd.get("name") or wd.get("title") or wd.get("workout_name", "Workout")
        # F2: template is equal to workout name for now
        template_name = workout_name

        date_key = _first_key(wd, ["start_time", "performed_at", "date", "started_at"])
        if date_key:
            try:
                date_val = pd.to_datetime(wd[date_key])
            except Exception:
                date_val = None
        else:
            date_val = None

        exercises = wd.get("exercises") or wd.get("workout_exercises") or []

        for ex in exercises:
            ex_name = ex.get("name") or ex.get("exercise_name", "Exercise")
            ex_type = ex.get("type") or ex.get("exercise_type", "")

            sets = ex.get("sets") or ex.get("exercise_sets") or []
            for idx, s in enumerate(sets, start=1):
                w_key = _first_key(
                    s,
                    ["weight_kg", "weight", "kg", "weight_value"],
                )
                weight_kg = s.get(w_key) if w_key else None

                r_key = _first_key(
                    s,
                    ["reps", "rep_count", "repetitions"],
                )
                reps = s.get(r_key) if r_key else None

                rir_key = _first_key(s, ["rir", "rpe"])
                rir = s.get(rir_key) if rir_key else None

                tempo = s.get("tempo")

                set_type = s.get("set_type") or s.get("type") or "work"

                notes = s.get("notes") or s.get("comment")

                rows.append(
                    {
                        "date": date_val,
                        "workout_id": workout_id,
                        "workout_name": workout_name,
                        "template_name": template_name,
                        "exercise_name": ex_name,
                        "exercise_type": ex_type,
                        "set_index": idx,
                        "set_type": set_type,
                        "weight_kg": weight_kg,
                        "reps": reps,
                        "rir": rir,
                        "tempo": tempo,
                        "notes": notes,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def sync_hevy():
    try:
        client = get_hevy_client()
    except RuntimeError as e:
        st.error(str(e))
        return

    response = client.get_workouts()
    workouts = response.workouts or []

    st.session_state[HEVY_WORKOUTS_RAW_KEY] = workouts

    workouts_df = hevy_workouts_to_df(workouts)
    st.session_state[HEVY_WORKOUTS_DF_KEY] = workouts_df

    if FEATURE_FLAGS["F1"]:
        sets_df = hevy_workouts_to_sets_df(workouts)
        st.session_state[HEVY_SETS_DF_KEY] = sets_df

    st.success(f"Hevy: retrieved {len(workouts)} workouts.")


# ------------- Garmin helpers -------------

def get_garmin_client() -> Garmin:
    email = st.secrets.get("GARMIN_EMAIL")
    password = st.secrets.get("GARMIN_PASSWORD")

    if not email or not password:
        raise RuntimeError(
            "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD in Streamlit secrets."
        )

    return Garmin(email, password)


def sync_garmin():
    try:
        client = get_garmin_client()
    except RuntimeError as e:
        st.error(str(e))
        return

    try:
        client.login()
        activities = client.get_activities(0, 30)
        df = pd.DataFrame(activities)
        st.session_state[GARMIN_DF_KEY] = df
        st.success(f"Garmin: retrieved {len(activities)} activities.")
    except GarminConnectAuthenticationError:
        st.error("Garmin authentication failed. Check email or password in secrets.")
    except Exception as e:
        st.error(f"Error communicating with Garmin: {e}")


# ------------- F3: Basic progression rules per exercise -------------

def compute_progression(sets_df: pd.DataFrame, goal_cfg: dict) -> pd.DataFrame:
    """
    F3:
    For each exercise, decide increase / keep / decrease weight
    based on latest session average reps and goal rep range.
    """
    if sets_df.empty:
        return pd.DataFrame()

    df = sets_df.copy()
    df = df[df["set_type"].str.lower().isin(["work", "working", "top", "backoff"])]

    if "date" in df.columns:
        df = df.sort_values("date")

    low = goal_cfg["rep_range_low"]
    high = goal_cfg["rep_range_high"]
    kg_step = goal_cfg["kg_step"]

    progression_rows = []

    for ex_name, g in df.groupby("exercise_name"):
        if "date" not in g.columns:
            continue

        sessions = g.groupby(g["date"].dt.date)

        session_stats = []
        for d, s in sessions:
            if s["reps"].notna().sum() == 0 or s["weight_kg"].notna().sum() == 0:
                continue
            avg_reps = s["reps"].astype(float).mean()
            avg_weight = s["weight_kg"].astype(float).mean()
            session_stats.append((d, avg_reps, avg_weight))

        if len(session_stats) == 0:
            continue

        session_stats = sorted(session_stats, key=lambda x: x[0])
        latest = session_stats[-1]
        latest_date, latest_reps, latest_weight = latest

        if latest_reps >= high and kg_step > 0:
            action = "increase"
            rec_weight = latest_weight + kg_step
        elif latest_reps <= (low - 2):
            action = "decrease"
            rec_weight = max(0.0, latest_weight - kg_step)
        else:
            action = "keep"
            rec_weight = latest_weight

        progression_rows.append(
            {
                "exercise_name": ex_name,
                "latest_date": latest_date,
                "latest_avg_reps": round(latest_reps, 1),
                "latest_avg_weight_kg": round(latest_weight, 2),
                "action": action,
                "recommended_weight_kg": round(rec_weight, 2),
                "target_reps_low": low,
                "target_reps_high": high,
            }
        )

    if not progression_rows:
        return pd.DataFrame()

    return pd.DataFrame(progression_rows)


# ------------- F4: Next workout recommendation per template -------------

def build_next_workout_tables(sets_df: pd.DataFrame, progression_df: pd.DataFrame):
    """
    F4:
    For each template_name, construct "next workout" table
    using last session for that template plus progression rules.
    """
    if sets_df.empty or progression_df.empty:
        return {}

    df = sets_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date")

    result = {}

    for template_name, g_t in df.groupby("template_name"):
        if "date" not in g_t.columns:
            continue
        last_date = g_t["date"].dropna().max()
        if pd.isna(last_date):
            continue

        last_workout = g_t[g_t["date"] == last_date]

        merged = last_workout.merge(
            progression_df,
            on="exercise_name",
            how="left",
            suffixes=("", "_prog"),
        )

        rec_rows = []
        for ex_name, ex_g in merged.groupby("exercise_name"):
            row = ex_g.iloc[0]
            rec_rows.append(
                {
                    "exercise_name": ex_name,
                    "set_count": ex_g["set_index"].max(),
                    "last_weight_kg": row.get("weight_kg"),
                    "last_avg_reps": row.get("latest_avg_reps"),
                    "action": row.get("action"),
                    "recommended_weight_kg": row.get("recommended_weight_kg"),
                    "target_reps_low": row.get("target_reps_low"),
                    "target_reps_high": row.get("target_reps_high"),
                }
            )

        if rec_rows:
            result[template_name] = pd.DataFrame(rec_rows).sort_values("exercise_name")

    return result


# ------------- F5: Weekly volume and intensity summary -------------

def compute_weekly_summary(sets_df: pd.DataFrame) -> pd.DataFrame:
    if sets_df.empty:
        return pd.DataFrame()

    df = sets_df.copy()
    if "date" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["date"])
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    df["volume"] = df["weight_kg"].fillna(0) * df["reps"].fillna(0)

    agg = (
        df.groupby(["week", "exercise_name"])
        .agg(
            total_sets=("set_index", "count"),
            total_reps=("reps", "sum"),
            total_volume=("volume", "sum"),
            avg_weight_kg=("weight_kg", "mean"),
        )
        .reset_index()
    )

    return agg.sort_values(["week", "exercise_name"])


# ------------- F6: Plateau detection -------------

def detect_plateaus(weekly_df: pd.DataFrame, goal_cfg: dict) -> pd.DataFrame:
    """
    F6:
    Compare last N weeks vs previous N weeks in volume.
    If improvement < 5 percent, mark as plateau.
    """
    if weekly_df.empty:
        return pd.DataFrame()

    plateau_weeks = goal_cfg["plateau_weeks"]
    threshold = 0.05

    out_rows = []

    for ex_name, g in weekly_df.groupby("exercise_name"):
        g = g.sort_values("week")
        if len(g) < plateau_weeks * 2:
            continue

        recent = g.tail(plateau_weeks)
        prev = g.iloc[-plateau_weeks * 2: -plateau_weeks]

        recent_vol = recent["total_volume"].sum()
        prev_vol = prev["total_volume"].sum()

        if prev_vol <= 0:
            continue

        change = (recent_vol - prev_vol) / prev_vol

        status = "plateau" if change < threshold else "progressing"

        out_rows.append(
            {
                "exercise_name": ex_name,
                "prev_volume": round(prev_vol, 1),
                "recent_volume": round(recent_vol, 1),
                "volume_change_pct": round(change * 100, 1),
                "status": status,
            }
        )

    if not out_rows:
        return pd.DataFrame()

    return pd.DataFrame(out_rows).sort_values("volume_change_pct")


# ------------- F7: Deload and variation suggestions -------------

def simple_muscle_group(ex_name: str) -> str:
    name = ex_name.lower()
    if any(k in name for k in ["bench", "press", "chest"]):
        return "chest"
    if any(k in name for k in ["row", "lat", "pulldown", "pull up", "pullup"]):
        return "back"
    if any(k in name for k in ["squat", "leg press", "lunge"]):
        return "legs"
    if any(k in name for k in ["deadlift", "rdl", "hinge"]):
        return "posterior chain"
    if any(k in name for k in ["curl", "bicep"]):
        return "biceps"
    if any(k in name for k in ["tricep", "extension"]):
        return "triceps"
    if any(k in name for k in ["shoulder", "overhead", "lateral raise", "deltoid"]):
        return "shoulders"
    return "other"


def build_deload_suggestions(plateaus_df: pd.DataFrame) -> pd.DataFrame:
    if plateaus_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in plateaus_df.iterrows():
        if row["status"] != "plateau":
            continue

        ex_name = row["exercise_name"]
        mg = simple_muscle_group(ex_name)
        suggestion_type = "deload" if row["recent_volume"] > row["prev_volume"] else "variation"

        if suggestion_type == "deload":
            suggestion = (
                "Deload: reduce working weight by about 10 percent for 1 week, "
                "keep same rep range, then reassess."
            )
        else:
            suggestion = (
                "Variation: swap this exercise for a close variation for 4 to 6 weeks, "
                "then retest the original."
            )

        rows.append(
            {
                "exercise_name": ex_name,
                "muscle_group": mg,
                "volume_change_pct": row["volume_change_pct"],
                "suggestion_type": suggestion_type,
                "suggestion": suggestion,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("exercise_name")


# ------------- F8: Simple readiness indicator -------------

def compute_readiness(sets_df: pd.DataFrame) -> str:
    """
    F8:
    Simple readiness based on last 3 days training volume.
    """
    if sets_df.empty or "date" not in sets_df.columns:
        return "unknown"

    df = sets_df.copy()
    df = df.dropna(subset=["date"])
    df["date_only"] = df["date"].dt.date
    df["volume"] = df["weight_kg"].fillna(0) * df["reps"].fillna(0)

    today = datetime.utcnow().date()
    three_days_ago = today - timedelta(days=3)

    recent = df[df["date_only"] >= three_days_ago]
    if recent.empty:
        return "green"

    daily_vol = (
        recent.groupby("date_only")["volume"]
        .sum()
        .reindex([three_days_ago + timedelta(days=i) for i in range(4)], fill_value=0)
    )

    last_three = daily_vol.iloc[-3:]
    days_trained = (last_three > 0).sum()
    avg_vol = last_three.mean()

    if days_trained == 3 and avg_vol > 20000:
        return "red"
    if days_trained >= 2 and avg_vol > 10000:
        return "orange"
    return "green"


# ------------- F10: PR tracking -------------

def compute_prs(sets_df: pd.DataFrame) -> pd.DataFrame:
    """
    F10:
    Best weight per rep count plus 1RM estimate per exercise.
    """
    if sets_df.empty:
        return pd.DataFrame()

    df = sets_df.copy()
    df = df.dropna(subset=["reps", "weight_kg"])
    if df.empty:
        return pd.DataFrame()

    best = (
        df.groupby(["exercise_name", "reps"])
        .agg(max_weight_kg=("weight_kg", "max"))
        .reset_index()
    )

    best["est_1rm_kg"] = best.apply(
        lambda r: r["max_weight_kg"] * (1 + r["reps"] / 30.0), axis=1
    )

    idx = best.groupby("exercise_name")["est_1rm_kg"].idxmax()
    pr_main = best.loc[idx].reset_index(drop=True)
    pr_main["est_1rm_kg"] = pr_main["est_1rm_kg"].round(2)
    pr_main["max_weight_kg"] = pr_main["max_weight_kg"].round(2)

    return pr_main.sort_values("exercise_name")


# ------------- Main app -------------

def main():
    st.set_page_config(
        page_title="Fitness Hub - Training brain",
        page_icon="💪",
        layout="wide",
    )

    init_session_state()
    sidebar_settings()

    goal_cfg = get_current_goal_config()

    st.title("Fitness Hub - Training brain")
    st.caption("Hevy plus Garmin integration with progression logic and recommendations (F1 - F10).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Hevy workouts"):
            with st.spinner("Syncing from Hevy"):
                sync_hevy()
    with col2:
        if st.button("Sync Garmin activities"):
            with st.spinner("Syncing from Garmin"):
                sync_garmin()

    sets_df = st.session_state[HEVY_SETS_DF_KEY]
    workouts_df = st.session_state[HEVY_WORKOUTS_DF_KEY]

    st.markdown("---")

    tabs = st.tabs(
        [
            "Sets (F1 & F2)",
            "Next workouts (F3 & F4)",
            "Weekly summary (F5)",
            "Plateaus & deload (F6 & F7)",
            "Readiness (F8)",
            "PRs (F10)",
            "Debug",
        ]
    )

    # F1 & F2
    with tabs[0]:
        st.subheader("Hevy set level data (F1) with template mapping (F2)")
        st.write(f"Workouts shape: {workouts_df.shape}")
        st.write(f"Sets shape: {sets_df.shape}")
        if not sets_df.empty:
            st.dataframe(sets_df.head(50))

    # F3 & F4
    with tabs[1]:
        st.subheader("Next workout recommendations (F3 & F4)")

        if sets_df.empty:
            st.info("Sync Hevy first to see recommendations.")
        else:
            progression_df = compute_progression(sets_df, goal_cfg)
            if progression_df.empty:
                st.info("Not enough data yet to compute progression.")
            else:
                st.markdown("**Per exercise progression table (F3)**")
                st.dataframe(progression_df)

                st.markdown("**Per template next workout tables (F4)**")
                templates = build_next_workout_tables(sets_df, progression_df)
                if not templates:
                    st.info("No template based next workout tables available yet.")
                else:
                    for t_name, t_df in templates.items():
                        st.markdown(f"**Template: {t_name}**")
                        st.dataframe(t_df)

    # F5
    with tabs[2]:
        st.subheader("Weekly volume and intensity summary (F5)")
        if sets_df.empty:
            st.info("Sync Hevy first.")
        else:
            weekly_df = compute_weekly_summary(sets_df)
            if weekly_df.empty:
                st.info("Not enough data yet to build weekly summary.")
            else:
                st.dataframe(weekly_df)

    # F6 & F7
    with tabs[3]:
        st.subheader("Plateaus and deload or variation suggestions (F6 & F7)")
        if sets_df.empty:
            st.info("Sync Hevy first.")
        else:
            weekly_df = compute_weekly_summary(sets_df)
            plateaus_df = detect_plateaus(weekly_df, goal_cfg)
            if plateaus_df.empty:
                st.info("No plateaus detected yet.")
            else:
                st.markdown("**Plateau status per exercise (F6)**")
                st.dataframe(plateaus_df)

                suggestions_df = build_deload_suggestions(plateaus_df)
                if suggestions_df.empty:
                    st.info("No suggestions generated yet.")
                else:
                    st.markdown("**Deload and variation suggestions (F7)**")
                    st.dataframe(suggestions_df)

    # F8
    with tabs[4]:
        st.subheader("Readiness indicator (F8)")
        readiness = compute_readiness(sets_df)
        if readiness == "green":
            st.success("Readiness: GREEN - you can push today based on recent training load.")
        elif readiness == "orange":
            st.warning("Readiness: ORANGE - consider being conservative with weight or sets today.")
        elif readiness == "red":
            st.error("Readiness: RED - recent training load is high on consecutive days, consider backing off.")
        else:
            st.info("Readiness: unknown - not enough recent data to estimate.")

        st.caption(
            "This is a first simple version based on training volume only. "
            "We can later enrich it with Garmin sleep and recovery metrics."
        )

    # F10
    with tabs[5]:
        st.subheader("Exercise PRs (F10)")
        if sets_df.empty:
            st.info("Sync Hevy first.")
        else:
            prs_df = compute_prs(sets_df)
            if prs_df.empty:
                st.info("No PRs detected yet. You need sets with weight and reps.")
            else:
                st.dataframe(prs_df)

    # Debug
    with tabs[6]:
        st.subheader("Debug")
        st.markdown("**Hevy workouts raw (type)**")
        st.write(type(st.session_state[HEVY_WORKOUTS_RAW_KEY]))
        st.markdown("**Hevy workouts DataFrame**")
        st.write(workouts_df.shape)
        if not workouts_df.empty:
            st.dataframe(workouts_df.head())

        st.markdown("**Garmin activities DataFrame**")
        g_df = st.session_state[GARMIN_DF_KEY]
        st.write(g_df.shape)
        if not g_df.empty:
            st.dataframe(g_df.head())

        st.markdown("**Settings**")
        st.json(st.session_state[SETTINGS_KEY])

        st.markdown("**Feature flags**")
        st.json(FEATURE_FLAGS)


if __name__ == "__main__":
    main()
