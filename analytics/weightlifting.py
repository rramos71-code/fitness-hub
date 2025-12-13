import pandas as pd
from datetime import datetime, timedelta


def _epley_1rm(weight: float, reps: float) -> float:
    return float(weight) * (1.0 + float(reps) / 30.0)


def _format_top_set(weight: float, reps: float) -> str:
    if pd.isna(weight) or pd.isna(reps):
        return ""
    return f"{weight:.0f} x {reps:.0f}"


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
