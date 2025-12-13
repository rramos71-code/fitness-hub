import pandas as pd
from datetime import datetime, timedelta


def _epley_1rm(weight: float, reps: float) -> float:
    return float(weight) * (1.0 + float(reps) / 30.0)


def _format_top_set(weight: float, reps: float) -> str:
    if pd.isna(weight) or pd.isna(reps):
        return ""
    return f"{weight:.0f} x {reps:.0f}"


def progression_rules_v1(exercise_df: pd.DataFrame, rep_low: int, rep_high: int, kg_step: float) -> dict:
    """
    Given working sets for a single exercise, derive a simple load recommendation.
    """
    if exercise_df is None or exercise_df.empty:
        return {
            "latest_date": None,
            "latest_avg_weight_kg": None,
            "latest_avg_reps": None,
            "action": "keep",
            "recommended_weight_kg": None,
            "reason": "No working sets available.",
        }

    # Ensure sorted to get latest
    exercise_df = exercise_df.copy().sort_values("date_day")
    latest_date = exercise_df["date_day"].max()
    latest_session = exercise_df[exercise_df["date_day"] == latest_date]

    latest_avg_weight = latest_session["weight_kg"].mean()
    latest_avg_reps = latest_session["reps"].mean()

    if pd.isna(latest_avg_reps) or pd.isna(latest_avg_weight):
        return {
            "latest_date": latest_date,
            "latest_avg_weight_kg": latest_avg_weight,
            "latest_avg_reps": latest_avg_reps,
            "action": "keep",
            "recommended_weight_kg": latest_avg_weight,
            "reason": "Insufficient data to determine recommendation.",
        }

    if latest_avg_reps >= rep_high:
        action = "increase"
        recommended = latest_avg_weight + kg_step
        reason = f"Avg reps {latest_avg_reps:.1f} >= {rep_high}; increase by {kg_step:g} kg."
    elif latest_avg_reps <= rep_low - 2:
        action = "decrease"
        recommended = max(latest_avg_weight - kg_step, 0)
        reason = f"Avg reps {latest_avg_reps:.1f} <= {rep_low-2}; decrease by {kg_step:g} kg."
    else:
        action = "keep"
        recommended = latest_avg_weight
        reason = f"Avg reps {latest_avg_reps:.1f} within target; keep load."

    return {
        "latest_date": latest_date,
        "latest_avg_weight_kg": latest_avg_weight,
        "latest_avg_reps": latest_avg_reps,
        "action": action,
        "recommended_weight_kg": recommended,
        "reason": reason,
    }


def build_progression_recommendations(sets_df: pd.DataFrame, goal_type: str = "recomp") -> pd.DataFrame:
    """
    Build per-exercise load recommendations based on recent session performance.
    """
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(
            columns=[
                "exercise_name",
                "latest_date",
                "latest_avg_weight_kg",
                "latest_avg_reps",
                "action",
                "recommended_weight_kg",
                "target_rep_range",
                "reason",
                "confidence",
            ]
        )

    goal_map = {
        "recomp": (6, 10, 2.5),
        "hypertrophy": (8, 12, 2.5),
        "strength": (3, 6, 5.0),
        "maintenance": (5, 10, 0.0),
    }
    rep_low, rep_high, kg_step = goal_map.get(goal_type, goal_map["recomp"])
    target_rep_range = f"{rep_low}-{rep_high}"

    rows = []
    for ex_name, g in sets_df.groupby("exercise_name"):
        g = g[g["is_working_set"]]
        if g.empty:
            continue

        rec = progression_rules_v1(g, rep_low, rep_high, kg_step)

        # Confidence based on working set count in latest session
        latest_session = g[g["date_day"] == rec["latest_date"]] if rec["latest_date"] is not None else pd.DataFrame()
        ws_count = len(latest_session)
        if ws_count >= 3:
            conf = 1.0
        elif ws_count == 2:
            conf = 0.7
        elif ws_count == 1:
            conf = 0.5
        else:
            conf = 0.3

        rows.append(
            {
                "exercise_name": ex_name,
                "latest_date": rec["latest_date"],
                "latest_avg_weight_kg": rec["latest_avg_weight_kg"],
                "latest_avg_reps": rec["latest_avg_reps"],
                "action": rec["action"],
                "recommended_weight_kg": rec["recommended_weight_kg"],
                "target_rep_range": target_rep_range,
                "reason": rec["reason"],
                "confidence": conf,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("latest_date", ascending=False).reset_index(drop=True)
    return out

def build_exercise_library(sets_df: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """
    Build per-exercise training summary from canonical Hevy sets.

    Args:
        sets_df: Canonicalized sets DataFrame (see utils.hevy_processing)
        lookback_days: window for considering sets (default 90 days)

    Returns:
        DataFrame with per-exercise aggregates:
        exercise_name, last_trained_date, sessions_28d, volume_7d_kg,
        volume_28d_kg, last_avg_weight_kg, last_avg_reps, last_top_set,
        est_1rm_kg, trend_28d
    """
    if sets_df is None or sets_df.empty:
        return pd.DataFrame(
            columns=[
                "exercise_name",
                "last_trained_date",
                "sessions_28d",
                "volume_7d_kg",
                "volume_28d_kg",
                "last_avg_weight_kg",
                "last_avg_reps",
                "last_top_set",
                "est_1rm_kg",
                "trend_28d",
            ]
        )

    df = sets_df.copy()
    today = datetime.utcnow().date()
    lookback_start = today - timedelta(days=lookback_days)
    df = df[df["date_day"] >= lookback_start]

    # Only working sets
    df = df[df["is_working_set"]]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "exercise_name",
                "last_trained_date",
                "sessions_28d",
                "volume_7d_kg",
                "volume_28d_kg",
                "last_avg_weight_kg",
                "last_avg_reps",
                "last_top_set",
                "est_1rm_kg",
                "trend_28d",
            ]
        )

    df["volume"] = df["weight_kg"] * df["reps"]

    seven_start = today - timedelta(days=7)
    four_start = today - timedelta(days=14)
    four_prev_start = today - timedelta(days=28)
    twenty8_start = today - timedelta(days=28)

    rows = []
    for ex_name, g in df.groupby("exercise_name"):
        if g.empty:
            continue

        last_trained = g["date_day"].max()
        recent_28 = g[g["date_day"] >= twenty8_start]
        recent_7 = g[g["date_day"] >= seven_start]

        sessions_28 = recent_28["date_day"].nunique()
        vol_7 = recent_7["volume"].sum()
        vol_28 = recent_28["volume"].sum()

        # Last session stats
        last_session = g[g["date_day"] == last_trained]
        last_avg_weight = last_session["weight_kg"].mean()
        last_avg_reps = last_session["reps"].mean()
        top_set_row = (
            last_session.sort_values(["weight_kg", "reps"], ascending=[False, False])
            .head(1)
        )
        if not top_set_row.empty:
            top_weight = top_set_row.iloc[0]["weight_kg"]
            top_reps = top_set_row.iloc[0]["reps"]
        else:
            top_weight = float("nan")
            top_reps = float("nan")
        last_top_set = _format_top_set(top_weight, top_reps)

        # Epley 1RM best in last 28d
        best_set = recent_28.sort_values(["weight_kg", "reps"], ascending=[False, False]).head(1)
        if not best_set.empty:
            best_weight = best_set.iloc[0]["weight_kg"]
            best_reps = best_set.iloc[0]["reps"]
            est_1rm = _epley_1rm(best_weight, best_reps)
        else:
            est_1rm = float("nan")

        # Trend: volume last 14d vs prior 14d
        last14 = g[(g["date_day"] >= four_start)]
        prev14 = g[(g["date_day"] < four_start) & (g["date_day"] >= four_prev_start)]
        v_last14 = last14["volume"].sum()
        v_prev14 = prev14["volume"].sum()
        if v_prev14 == 0 and v_last14 == 0:
            trend = "flat"
        elif v_prev14 == 0 and v_last14 > 0:
            trend = "up"
        else:
            delta = (v_last14 - v_prev14) / v_prev14
            if delta > 0.05:
                trend = "up"
            elif delta < -0.05:
                trend = "down"
            else:
                trend = "flat"

        rows.append(
            {
                "exercise_name": ex_name,
                "last_trained_date": last_trained,
                "sessions_28d": sessions_28,
                "volume_7d_kg": vol_7,
                "volume_28d_kg": vol_28,
                "last_avg_weight_kg": last_avg_weight,
                "last_avg_reps": last_avg_reps,
                "last_top_set": last_top_set,
                "est_1rm_kg": est_1rm,
                "trend_28d": trend,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("last_trained_date", ascending=False).reset_index(drop=True)
    return out
