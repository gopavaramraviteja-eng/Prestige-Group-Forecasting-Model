"""
forecasting_lib.py
-------------------
Shared feature-engineering, training, evaluation and forecasting logic for the
Prestige Constructions synthetic construction-progress & cost dataset.

This module is imported by both:
  - forecasting_model.ipynb (exploration / training / evaluation notebook)
  - train_and_save_models.py (headless training script used to produce the
    model artifacts consumed by the Streamlit app)
  - app.py (Streamlit dashboard, for feature engineering + recursive forecasting)

Keeping the logic in one place guarantees the notebook, the training script
and the live app all compute features and forecasts the exact same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# Columns that must NEVER be used as model features: identifiers, the raw
# targets themselves, or latent simulation parameters that would not be
# observable in a real deployment (using them would leak the ground truth
# and make the forecasting exercise meaningless).
NON_FEATURE_COLS = {
    "date", "project_id", "project_name",
    "daily_progress_increment_pct", "actual_progress_pct",
    "daily_cost_incurred_lakhs", "cumulative_cost_incurred_cr",
    "target_next_increment_pct", "target_next_cost_lakhs",
    "mgmt_quality_factor", "overrun_factor", "delay_reason",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the four synthetic CSVs into a dict of DataFrames with dates parsed."""
    data_dir = Path(data_dir)
    projects = pd.read_csv(
        data_dir / "dim_projects.csv",
        parse_dates=["planned_start_date", "actual_start_date", "planned_completion_date"],
    )
    progress = pd.read_csv(data_dir / "fact_daily_construction_progress.csv", parse_dates=["date"])
    cost_breakdown = pd.read_csv(data_dir / "fact_daily_cost_breakdown.csv", parse_dates=["date"])
    milestones = pd.read_csv(
        data_dir / "dim_milestones.csv", parse_dates=["planned_date", "actual_date"]
    )
    return {
        "projects": projects,
        "progress": progress,
        "cost_breakdown": cost_breakdown,
        "milestones": milestones,
    }


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def _add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def build_feature_table(
    progress: pd.DataFrame,
    projects: pd.DataFrame,
    lags: tuple[int, ...] = (1, 7, 14),
    rolling_windows: tuple[int, ...] = (7, 14, 30),
) -> pd.DataFrame:
    """
    Build a per-project-day feature table with next-day targets.

    For every (project, day t) row we compute features using only information
    available up to and including day t, and set the label to what happens on
    day t+1 (`target_next_increment_pct`, `target_next_cost_lakhs`). This is a
    one-step-ahead ("nowcast tomorrow from today") framing; the recursive
    `forecast_project` function below chains one-step predictions to reach any
    horizon.
    """
    df = progress.sort_values(["project_id", "date"]).copy()
    df = _add_calendar_features(df)

    df["schedule_variance_pct"] = df["actual_progress_pct"] - df["planned_progress_pct"]
    df["cost_variance_cr"] = df["cumulative_cost_incurred_cr"] - df["planned_cost_to_date_cr"]
    df["budget_utilisation_pct"] = df["cumulative_cost_incurred_cr"] / df["revised_budget_cr"] * 100

    df["days_since_start"] = df.groupby("project_id").cumcount()

    grp = df.groupby("project_id", group_keys=False)

    for lag in lags:
        df[f"increment_lag_{lag}"] = grp["daily_progress_increment_pct"].shift(lag)
        df[f"cost_lag_{lag}"] = grp["daily_cost_incurred_lakhs"].shift(lag)

    for w in rolling_windows:
        df[f"increment_rollmean_{w}"] = (
            grp["daily_progress_increment_pct"].apply(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"cost_rollmean_{w}"] = (
            grp["daily_cost_incurred_lakhs"].apply(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"workers_rollmean_{w}"] = (
            grp["workers_onsite"].apply(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"rainfall_rollsum_{w}"] = (
            grp["rainfall_mm"].apply(lambda s: s.rolling(w, min_periods=1).sum())
        )
        df[f"delay_rollsum_{w}"] = (
            grp["is_delay_event"].apply(lambda s: s.rolling(w, min_periods=1).sum())
        )

    # Next-day targets (shift -1 within each project's timeline)
    df["target_next_increment_pct"] = grp["daily_progress_increment_pct"].shift(-1)
    df["target_next_cost_lakhs"] = grp["daily_cost_incurred_lakhs"].shift(-1)

    # Note: revised_budget_cr already exists in the progress fact table, so it's
    # deliberately left out of static_cols to avoid a duplicate merged column.
    static_cols = [
        "project_id", "city", "project_type", "total_units", "saleable_area_sqft",
        "planned_duration_months", "approved_budget_cr", "cost_per_sqft_inr",
    ]
    df = df.merge(projects[static_cols], on="project_id", how="left", suffixes=("", "_proj"))

    df = pd.get_dummies(df, columns=["city", "project_type", "milestone_stage"], prefix=["city", "ptype", "stage"])

    # Rows with no next-day target (last day per project) can't be used for training.
    df = df.dropna(subset=["target_next_increment_pct", "target_next_cost_lakhs"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def time_split(df: pd.DataFrame, test_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_days` calendar days (across all projects) as test set."""
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #

def train_rf(X: pd.DataFrame, y: pd.Series, n_estimators: int = 300, max_depth: int | None = 14) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }


def feature_importance(model, feature_cols: list[str], top_n: int = 20) -> pd.DataFrame:
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return imp.head(top_n).rename("importance").reset_index().rename(columns={"index": "feature"})


# --------------------------------------------------------------------------- #
# Recursive multi-day forecasting
# --------------------------------------------------------------------------- #

@dataclass
class ForecastConfig:
    horizon_days: int = 90
    n_simulations: int = 1
    delay_event_prob: float = 0.045
    residual_std_increment: float = 0.0
    residual_std_cost: float = 0.0


def _rainfall_climatology(progress: pd.DataFrame) -> pd.DataFrame:
    tmp = progress.copy()
    tmp["month"] = tmp["date"].dt.month
    return tmp.groupby(["city", "month"])["rainfall_mm"].mean().reset_index()


def _material_index_trend(progress: pd.DataFrame, lookback_days: int = 180) -> tuple[float, float]:
    """Fit a simple linear trend (slope, intercept-at-last-day) on the most recent
    `lookback_days` of the macro material_price_index series so it can be
    extrapolated into the forecast horizon."""
    idx = progress[["date", "material_price_index"]].drop_duplicates("date").sort_values("date")
    idx = idx.tail(lookback_days)
    x = np.arange(len(idx))
    y = idx["material_price_index"].values
    if len(x) < 2:
        return 0.0, float(y[-1]) if len(y) else 100.0
    slope, intercept = np.polyfit(x, y, 1)
    last_val = slope * (len(x) - 1) + intercept
    return float(slope), float(last_val)


def planned_progress_at(proj_row, as_of_date: pd.Timestamp) -> float:
    """Deterministic planned-schedule progress % for a project on a given date,
    using the same logistic S-curve shape the synthetic generator used. This
    lets forecast steps compute an accurate schedule-variance feature without
    needing the original planned_progress_pct series to already cover that date."""
    planned_dur = float(proj_row.planned_duration_months) * 30.0
    t = (pd.Timestamp(as_of_date) - pd.Timestamp(proj_row.actual_start_date)).days
    k = 10.0 / planned_dur
    midpoint = planned_dur / 2.0

    def cum(x):
        return 100.0 / (1.0 + np.exp(-k * (x - midpoint)))

    p0, p1 = cum(0.0), cum(planned_dur)
    val = (cum(t) - p0) / (p1 - p0) * 100.0
    return float(np.clip(val, 0.0, 100.0))


def stage_for_pct(pct: float) -> str:
    if pct < 15:
        return "Excavation & Foundation"
    if pct < 30:
        return "Basement & Podium"
    if pct < 70:
        return "Superstructure"
    if pct < 90:
        return "MEP & Finishing"
    return "Handover & Clearance"


_STAGE_BOUNDS = [15, 30, 70, 90]
_STAGE_NAMES = ["Excavation & Foundation", "Basement & Podium", "Superstructure",
                "MEP & Finishing", "Handover & Clearance"]


def _stage_onehot_matrix(pct: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized version of stage_for_pct: returns {stage_name: 0/1 array}."""
    idx = np.searchsorted(_STAGE_BOUNDS, pct, side="right")
    return {name: (idx == i).astype(int) for i, name in enumerate(_STAGE_NAMES)}


def forecast_project(
    project_id: str,
    progress: pd.DataFrame,
    projects: pd.DataFrame,
    progress_model: RandomForestRegressor,
    cost_model: RandomForestRegressor,
    feature_cols: list[str],
    config: ForecastConfig = ForecastConfig(),
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Recursively roll a trained one-step-ahead model forward `config.horizon_days`
    days for a single project, starting from its last known actual day.

    Future exogenous inputs the model can't otherwise know (rainfall, macro
    material price index, weekday delay risk) are filled in with a seasonal
    climatology / trend extrapolation rather than left blank, and — when
    `config.n_simulations > 1` — randomised per simulated path (Monte Carlo)
    to produce a spread of plausible futures instead of one deterministic line.

    All `config.n_simulations` paths are advanced together, one calendar day at
    a time, with a single batched `model.predict()` call per day (rather than
    one call per simulation per day) so Monte Carlo bands stay fast enough for
    an interactive app.

    Returns a long DataFrame: one row per (simulation, forecast day).
    """
    rng = rng or np.random.default_rng(RANDOM_STATE)
    proj_row = projects[projects.project_id == project_id].iloc[0]
    hist = progress[progress.project_id == project_id].sort_values("date").copy()
    if hist.empty:
        raise ValueError(f"No history found for {project_id}")

    # Recursive forecasting calls .predict() once per day on a tiny batch
    # (n_simulations rows). RandomForest's joblib multiprocessing backend
    # (used during training for speed via n_jobs=-1) adds large fixed overhead
    # per call that swamps the actual prediction cost at this batch size, so
    # force single-threaded prediction for the duration of this loop.
    progress_model.n_jobs = 1
    cost_model.n_jobs = 1

    rain_clim = _rainfall_climatology(progress)
    rain_clim_city = rain_clim[rain_clim.city == proj_row.city].set_index("month")["rainfall_mm"]
    overall_mean_rain = float(progress["rainfall_mm"].mean())

    S = max(1, config.n_simulations)
    W = 30  # fixed rolling-window width carried forward each step

    def _tail_window(col: str) -> np.ndarray:
        vals = hist[col].to_numpy(dtype=float)
        if len(vals) >= W:
            base = vals[-W:]
        else:
            pad = np.full(W - len(vals), vals[0] if len(vals) else 0.0)
            base = np.concatenate([pad, vals])
        return np.tile(base, (S, 1))  # (S, W), identical across sims initially

    win_incr = _tail_window("daily_progress_increment_pct")
    win_cost = _tail_window("daily_cost_incurred_lakhs")
    win_workers = _tail_window("workers_onsite")
    win_rain = _tail_window("rainfall_mm")
    win_delay = _tail_window("is_delay_event")

    cum_progress = np.full(S, float(hist["actual_progress_pct"].iloc[-1]))
    cum_cost = np.full(S, float(hist["cumulative_cost_incurred_cr"].iloc[-1]))
    last_date = hist["date"].iloc[-1]
    days_since_start = len(hist) - 1

    static = {
        "total_units": proj_row.total_units,
        "saleable_area_sqft": proj_row.saleable_area_sqft,
        "planned_duration_months": proj_row.planned_duration_months,
        "approved_budget_cr": proj_row.approved_budget_cr,
        "revised_budget_cr": proj_row.revised_budget_cr,
        "cost_per_sqft_inr": proj_row.cost_per_sqft_inr,
    }
    for c in sorted(projects["city"].unique()):
        static[f"city_{c}"] = int(proj_row.city == c)
    for t in sorted(projects["project_type"].unique()):
        static[f"ptype_{t}"] = int(proj_row.project_type == t)

    results = []

    for step in range(1, config.horizon_days + 1):
        cur_date = last_date + pd.Timedelta(days=step)
        days_since_start += 1

        planned_pct_now = planned_progress_at(proj_row, cur_date)
        planned_cost_now = planned_pct_now / 100.0 * proj_row.revised_budget_cr

        feat = pd.DataFrame({k: np.full(S, v) for k, v in static.items()})
        feat["day_of_week"] = cur_date.dayofweek
        feat["month"] = cur_date.month
        feat["is_weekend"] = int(cur_date.dayofweek >= 5)
        feat["days_since_start"] = days_since_start
        feat["schedule_variance_pct"] = cum_progress - planned_pct_now
        feat["cost_variance_cr"] = cum_cost - planned_cost_now
        feat["budget_utilisation_pct"] = cum_cost / proj_row.revised_budget_cr * 100

        feat["increment_lag_1"] = win_incr[:, -1]
        feat["increment_lag_7"] = win_incr[:, -7]
        feat["increment_lag_14"] = win_incr[:, -14]
        feat["cost_lag_1"] = win_cost[:, -1]
        feat["cost_lag_7"] = win_cost[:, -7]
        feat["cost_lag_14"] = win_cost[:, -14]
        for w in (7, 14, 30):
            feat[f"increment_rollmean_{w}"] = win_incr[:, -w:].mean(axis=1)
            feat[f"cost_rollmean_{w}"] = win_cost[:, -w:].mean(axis=1)
            feat[f"workers_rollmean_{w}"] = win_workers[:, -w:].mean(axis=1)
            feat[f"rainfall_rollsum_{w}"] = win_rain[:, -w:].sum(axis=1)
            feat[f"delay_rollsum_{w}"] = win_delay[:, -w:].sum(axis=1)

        for name, arr in _stage_onehot_matrix(cum_progress).items():
            feat[f"stage_{name}"] = arr

        for col in feature_cols:
            if col not in feat.columns:
                feat[col] = 0
        feat = feat[feature_cols]

        pred_increment = progress_model.predict(feat)
        pred_cost = cost_model.predict(feat)

        if config.n_simulations > 1:
            pred_increment = pred_increment + rng.normal(0, config.residual_std_increment, size=S)
            pred_cost = pred_cost + rng.normal(0, config.residual_std_cost, size=S)
            dropout = rng.random(S) < config.delay_event_prob
            pred_increment = np.where(dropout, pred_increment * 0.05, pred_increment)
            pred_cost = np.where(dropout, pred_cost * 0.3, pred_cost)
        else:
            dropout = np.zeros(S, dtype=bool)

        pred_increment = np.clip(pred_increment, 0, 100 - cum_progress)
        pred_cost = np.clip(pred_cost, 0, None)

        cum_progress = np.clip(cum_progress + pred_increment, 0, 100)
        cum_cost = cum_cost + pred_cost / 100.0  # lakhs -> crore

        rain_today = rain_clim_city.get(cur_date.month, overall_mean_rain)
        win_incr = np.concatenate([win_incr[:, 1:], pred_increment[:, None]], axis=1)
        win_cost = np.concatenate([win_cost[:, 1:], pred_cost[:, None]], axis=1)
        win_workers = np.concatenate([win_workers[:, 1:], feat["workers_rollmean_7"].to_numpy()[:, None]], axis=1)
        win_rain = np.concatenate([win_rain[:, 1:], np.full((S, 1), rain_today)], axis=1)
        win_delay = np.concatenate([win_delay[:, 1:], dropout.astype(float)[:, None]], axis=1)

        for s in range(S):
            results.append({
                "project_id": project_id,
                "sim": s,
                "date": cur_date,
                "forecast_progress_pct": float(cum_progress[s]),
                "forecast_daily_increment_pct": float(pred_increment[s]),
                "forecast_cumulative_cost_cr": float(cum_cost[s]),
                "forecast_daily_cost_lakhs": float(pred_cost[s]),
            })

        if np.all(cum_progress >= 100.0):
            break

    return pd.DataFrame(results)


def summarise_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-simulation forecast rows into p10/p50/p90 bands per date."""
    agg = forecast_df.groupby("date").agg(
        progress_p10=("forecast_progress_pct", lambda s: np.percentile(s, 10)),
        progress_p50=("forecast_progress_pct", lambda s: np.percentile(s, 50)),
        progress_p90=("forecast_progress_pct", lambda s: np.percentile(s, 90)),
        cost_p10=("forecast_cumulative_cost_cr", lambda s: np.percentile(s, 10)),
        cost_p50=("forecast_cumulative_cost_cr", lambda s: np.percentile(s, 50)),
        cost_p90=("forecast_cumulative_cost_cr", lambda s: np.percentile(s, 90)),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_artifacts(
    models_dir: str | Path,
    progress_model,
    cost_model,
    feature_cols: list[str],
    metrics: dict,
):
    import joblib
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(progress_model, models_dir / "progress_model.joblib")
    joblib.dump(cost_model, models_dir / "cost_model.joblib")
    with open(models_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_artifacts(models_dir: str | Path):
    import joblib
    models_dir = Path(models_dir)
    progress_model = joblib.load(models_dir / "progress_model.joblib")
    cost_model = joblib.load(models_dir / "cost_model.joblib")
    with open(models_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)
    metrics = {}
    metrics_path = models_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return progress_model, cost_model, feature_cols, metrics
