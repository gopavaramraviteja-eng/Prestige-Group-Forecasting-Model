"""
Streamlit app: Prestige Constructions — Construction Progress & Cost Forecasting

Run with:
    streamlit run app.py

On first run it trains the two RandomForest models (progress + cost) from the
CSVs in ./data and caches them to ./models (~30-60s, one time only). Every
run after that loads the cached models instantly. Delete the ./models folder
to force a retrain (e.g. after replacing the data with real project data).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecasting_lib as fl

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

st.set_page_config(page_title="Prestige Construction Forecasting", layout="wide", page_icon="🏗️")

PALETTE = {
    "actual": "#4C78A8",
    "forecast": "#E45756",
    "band": "rgba(228, 87, 86, 0.18)",
    "budget": "#333333",
    "planned": "#9CA3AF",
}


# --------------------------------------------------------------------------- #
# Cached data / model loading (and first-run training)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading data and models (first run trains the models — about a minute)...")
def get_pipeline():
    data = fl.load_data(DATA_DIR)

    required = ["progress_model.joblib", "cost_model.joblib", "feature_cols.json", "metrics.json"]
    if not all((MODELS_DIR / f).exists() for f in required):
        feat = fl.build_feature_table(data["progress"], data["projects"])
        feature_cols = fl.get_feature_columns(feat)
        train, test = fl.time_split(feat, test_days=90)
        X_train, X_test = train[feature_cols], test[feature_cols]

        progress_model = fl.train_rf(X_train, train["target_next_increment_pct"], n_estimators=300)
        cost_model = fl.train_rf(X_train, train["target_next_cost_lakhs"], n_estimators=300)

        progress_metrics = fl.evaluate(progress_model, X_test, test["target_next_increment_pct"])
        cost_metrics = fl.evaluate(cost_model, X_test, test["target_next_cost_lakhs"])
        metrics = {
            "progress_model": progress_metrics,
            "cost_model": cost_metrics,
            "residual_std_increment": float(np.std(
                progress_model.predict(X_test) - test["target_next_increment_pct"])),
            "residual_std_cost": float(np.std(
                cost_model.predict(X_test) - test["target_next_cost_lakhs"])),
        }
        fl.save_artifacts(MODELS_DIR, progress_model, cost_model, feature_cols, metrics)

    progress_model, cost_model, feature_cols, metrics = fl.load_artifacts(MODELS_DIR)
    return data, progress_model, cost_model, feature_cols, metrics


@st.cache_data(show_spinner="Running forecast simulation...")
def run_forecast(project_id: str, horizon_days: int, n_simulations: int,
                  resid_incr: float, resid_cost: float) -> pd.DataFrame:
    data, progress_model, cost_model, feature_cols, _ = get_pipeline()
    cfg = fl.ForecastConfig(
        horizon_days=horizon_days,
        n_simulations=n_simulations,
        residual_std_increment=resid_incr,
        residual_std_cost=resid_cost,
    )
    return fl.forecast_project(
        project_id, data["progress"], data["projects"], progress_model, cost_model, feature_cols, cfg
    )


data, progress_model, cost_model, feature_cols, metrics = get_pipeline()
projects, progress, cost_breakdown, milestones = (
    data["projects"], data["progress"], data["cost_breakdown"], data["milestones"]
)
AS_OF = progress["date"].max()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("🏗️ Prestige Forecasting")
st.sidebar.caption("Synthetic data — for forecasting-model demonstration only.")
page = st.sidebar.radio("View", ["Portfolio overview", "Project deep-dive", "Model quality"])

st.sidebar.divider()
st.sidebar.markdown(f"**Data as of:** {AS_OF.date()}")
st.sidebar.markdown(f"**Projects:** {len(projects)}")
if metrics:
    st.sidebar.markdown(
        f"**Model R²:** progress {metrics['progress_model']['r2']:.2f} · "
        f"cost {metrics['cost_model']['r2']:.2f}"
    )


# --------------------------------------------------------------------------- #
# Page: Portfolio overview
# --------------------------------------------------------------------------- #

def portfolio_page():
    st.title("Portfolio overview")

    last_rows = (
        progress.sort_values("date").groupby("project_id").tail(1)
        .merge(projects, on="project_id", suffixes=("", "_proj"))
    )

    total_budget = projects["revised_budget_cr"].sum()
    total_cost = last_rows["cumulative_cost_incurred_cr"].sum()
    avg_progress = (last_rows["actual_progress_pct"] * projects.set_index("project_id").loc[
        last_rows["project_id"], "revised_budget_cr"].values).sum() / total_budget
    n_delayed = (last_rows["actual_progress_pct"] < last_rows["planned_progress_pct"] - 5).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio budget (revised)", f"₹{total_budget:,.0f} cr")
    c2.metric("Cost incurred to date", f"₹{total_cost:,.0f} cr", f"{total_cost/total_budget*100:.0f}% of budget")
    c3.metric("Budget-weighted progress", f"{avg_progress:.1f}%")
    c4.metric("Projects behind schedule (>5pts)", f"{n_delayed} / {len(projects)}")

    st.subheader("All projects")
    tbl = last_rows[[
        "project_id", "project_name", "city", "project_type", "status_as_of_today",
        "planned_progress_pct", "actual_progress_pct", "cumulative_cost_incurred_cr", "revised_budget_cr",
    ]].copy()
    tbl["schedule_variance_pts"] = tbl["actual_progress_pct"] - tbl["planned_progress_pct"]
    tbl["budget_utilisation_pct"] = tbl["cumulative_cost_incurred_cr"] / tbl["revised_budget_cr"] * 100
    tbl = tbl.sort_values("schedule_variance_pts")
    st.dataframe(
        tbl.rename(columns={
            "project_id": "ID", "project_name": "Project", "city": "City", "project_type": "Type",
            "status_as_of_today": "Status", "planned_progress_pct": "Planned %",
            "actual_progress_pct": "Actual %", "cumulative_cost_incurred_cr": "Cost to date (cr)",
            "revised_budget_cr": "Budget (cr)", "schedule_variance_pts": "Schedule variance (pts)",
            "budget_utilisation_pct": "Budget used %",
        }).style.format({
            "Planned %": "{:.1f}", "Actual %": "{:.1f}", "Cost to date (cr)": "{:.1f}",
            "Budget (cr)": "{:.1f}", "Schedule variance (pts)": "{:+.1f}", "Budget used %": "{:.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Progress by city")
        city_prog = last_rows.groupby("city")["actual_progress_pct"].mean().sort_values()
        fig = go.Figure(go.Bar(x=city_prog.values, y=city_prog.index, orientation="h",
                                marker_color=PALETTE["actual"]))
        fig.update_layout(xaxis_title="Avg. actual progress %", margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Delay events by reason (portfolio-wide)")
        reasons = progress[progress.is_delay_event == 1]["delay_reason"].value_counts()
        fig = go.Figure(go.Bar(x=reasons.values, y=reasons.index, orientation="h",
                                marker_color=PALETTE["forecast"]))
        fig.update_layout(xaxis_title="Number of delay-days", margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most schedule-delayed projects (milestone variance)")
    ms = milestones.dropna(subset=["variance_days"]).groupby("project_id")["variance_days"].mean().sort_values(ascending=False)
    ms_tbl = ms.head(8).reset_index().merge(projects[["project_id", "project_name", "city"]], on="project_id")
    ms_tbl = ms_tbl.rename(columns={"variance_days": "Avg. milestone delay (days)", "project_name": "Project", "city": "City"})
    st.dataframe(ms_tbl[["project_id", "Project", "City", "Avg. milestone delay (days)"]]
                 .rename(columns={"project_id": "ID"}).style.format({"Avg. milestone delay (days)": "{:.0f}"}),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Project deep-dive
# --------------------------------------------------------------------------- #

def deep_dive_page():
    st.title("Project deep-dive")

    proj_labels = {
        row.project_id: f"{row.project_id} — {row.project_name} ({row.city})"
        for row in projects.itertuples()
    }
    pid = st.sidebar.selectbox("Project", options=list(proj_labels.keys()),
                                format_func=lambda x: proj_labels[x])

    st.sidebar.subheader("Forecast settings")
    horizon = st.sidebar.slider("Forecast horizon (days)", 30, 365, 180, step=30)
    n_sims = st.sidebar.slider("Monte Carlo simulations", 1, 50, 20, step=1,
                                help="More simulations = smoother uncertainty band, slower to compute.")

    proj_row = projects[projects.project_id == pid].iloc[0]
    hist = progress[progress.project_id == pid].sort_values("date")
    last = hist.iloc[-1]

    st.subheader(f"{proj_row.project_name} — {proj_row.city}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", proj_row.status_as_of_today)
    c2.metric("Actual progress", f"{last.actual_progress_pct:.1f}%",
              f"{last.actual_progress_pct - last.planned_progress_pct:+.1f} pts vs. plan")
    c3.metric("Cost to date", f"₹{last.cumulative_cost_incurred_cr:.1f} cr",
              f"{last.cumulative_cost_incurred_cr / proj_row.revised_budget_cr * 100:.0f}% of budget")
    c4.metric("Revised budget", f"₹{proj_row.revised_budget_cr:.1f} cr")
    c5.metric("Planned duration", f"{proj_row.planned_duration_months} months")

    if proj_row.status_as_of_today == "Completed":
        st.success("This project is complete — showing full historical curve (no forward forecast needed).")
        forecast_df, band = pd.DataFrame(), pd.DataFrame()
    else:
        resid_incr = metrics.get("residual_std_increment", 0.02)
        resid_cost = metrics.get("residual_std_cost", 5.0)
        forecast_df = run_forecast(pid, horizon, n_sims, resid_incr, resid_cost)
        band = fl.summarise_forecast(forecast_df) if len(forecast_df) else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["Progress", "Cost", "Milestones & delays"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["planned_progress_pct"], name="Planned schedule",
                                  line=dict(color=PALETTE["planned"], dash="dot")))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual_progress_pct"], name="Actual (history)",
                                  line=dict(color=PALETTE["actual"], width=2.5)))
        if len(band):
            fig.add_trace(go.Scatter(x=pd.concat([band["date"], band["date"][::-1]]),
                                      y=pd.concat([band["progress_p90"], band["progress_p10"][::-1]]),
                                      fill="toself", fillcolor=PALETTE["band"],
                                      line=dict(color="rgba(0,0,0,0)"), name="p10-p90", showlegend=True))
            fig.add_trace(go.Scatter(x=band["date"], y=band["progress_p50"], name="Forecast (median)",
                                      line=dict(color=PALETTE["forecast"], width=2, dash="dash")))
        fig.update_layout(yaxis_title="Cumulative progress %", height=430,
                           margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        if len(band):
            done = band[band["progress_p50"] >= 99.9]
            if len(done):
                eta = done.iloc[0]["date"]
                st.info(f"Median forecast completion: **{eta.date()}** "
                        f"({(eta - AS_OF).days} days from now).")
            else:
                st.warning(f"Not forecast to reach 100% within the {horizon}-day horizon at the current pace.")

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["planned_cost_to_date_cr"], name="Planned budget curve",
                                  line=dict(color=PALETTE["planned"], dash="dot")))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["cumulative_cost_incurred_cr"], name="Actual (history)",
                                  line=dict(color=PALETTE["actual"], width=2.5)))
        if len(band):
            fig.add_trace(go.Scatter(x=pd.concat([band["date"], band["date"][::-1]]),
                                      y=pd.concat([band["cost_p90"], band["cost_p10"][::-1]]),
                                      fill="toself", fillcolor=PALETTE["band"],
                                      line=dict(color="rgba(0,0,0,0)"), name="p10-p90", showlegend=True))
            fig.add_trace(go.Scatter(x=band["date"], y=band["cost_p50"], name="Forecast (median)",
                                      line=dict(color=PALETTE["forecast"], width=2, dash="dash")))
        fig.add_hline(y=proj_row.revised_budget_cr, line=dict(color=PALETTE["budget"], dash="dash"),
                      annotation_text="Revised budget")
        fig.update_layout(yaxis_title="Cumulative cost (INR crore)", height=430,
                           margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        if len(forecast_df):
            at_completion = forecast_df.sort_values("date").groupby("sim").apply(
                lambda g: g.loc[g["forecast_progress_pct"].idxmax()]
            )
            over_budget_frac = (at_completion["forecast_cumulative_cost_cr"] > proj_row.revised_budget_cr).mean()
            median_cost_at_completion = at_completion["forecast_cumulative_cost_cr"].median()
            st.info(
                f"Across {forecast_df['sim'].nunique()} simulated futures, median forecast cost at the "
                f"simulated end-of-horizon is **₹{median_cost_at_completion:.1f} cr** "
                f"vs. a ₹{proj_row.revised_budget_cr:.1f} cr revised budget — "
                f"**{over_budget_frac*100:.0f}%** of simulations exceed budget."
            )

        cb = cost_breakdown[cost_breakdown.project_id == pid]
        cat_totals = cb.groupby("category")["cost_incurred_lakhs"].sum().sort_values(ascending=False) / 100
        fig2 = go.Figure(go.Bar(x=cat_totals.index, y=cat_totals.values, marker_color=PALETTE["actual"]))
        fig2.update_layout(title="Cost incurred to date by category", yaxis_title="INR crore",
                            height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        ms = milestones[milestones.project_id == pid].copy()
        ms["variance_days"] = ms["variance_days"]
        st.dataframe(
            ms[["milestone_name", "planned_date", "actual_date", "variance_days"]]
            .rename(columns={"milestone_name": "Milestone", "planned_date": "Planned",
                              "actual_date": "Actual", "variance_days": "Variance (days)"}),
            use_container_width=True, hide_index=True,
        )
        delays = hist[hist.is_delay_event == 1]
        st.write(f"**{len(delays)}** delay-days recorded in this project's history.")
        if len(delays):
            st.dataframe(delays[["date", "delay_reason", "rainfall_mm"]]
                         .rename(columns={"date": "Date", "delay_reason": "Reason", "rainfall_mm": "Rainfall (mm)"}),
                         use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Model quality
# --------------------------------------------------------------------------- #

def model_quality_page():
    st.title("Model quality")
    st.caption("Diagnostics for the two underlying RandomForest models (trained on data up to the "
               "90-day-earlier cutoff, evaluated on the most recent 90 days).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Progress model (next-day % increment)")
        m = metrics.get("progress_model", {})
        st.metric("R²", f"{m.get('r2', float('nan')):.3f}")
        st.metric("MAE", f"{m.get('mae', float('nan')):.4f} pts/day")
        st.metric("RMSE", f"{m.get('rmse', float('nan')):.4f} pts/day")
    with col2:
        st.subheader("Cost model (next-day cost, INR lakh)")
        m = metrics.get("cost_model", {})
        st.metric("R²", f"{m.get('r2', float('nan')):.3f}")
        st.metric("MAE", f"₹{m.get('mae', float('nan')):.2f} lakh/day")
        st.metric("RMSE", f"₹{m.get('rmse', float('nan')):.2f} lakh/day")

    st.subheader("Feature importance")
    tab1, tab2 = st.tabs(["Progress model", "Cost model"])
    with tab1:
        imp = fl.feature_importance(progress_model, feature_cols, top_n=15)
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                                marker_color=PALETTE["actual"]))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        imp = fl.feature_importance(cost_model, feature_cols, top_n=15)
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                                marker_color=PALETTE["forecast"]))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Both models deliberately exclude the simulator's latent generation parameters "
        "(`mgmt_quality_factor`, `overrun_factor`) — a real deployment would never observe those, "
        "so leaving them in would leak the answer and overstate accuracy. See "
        "`forecasting_lib.NON_FEATURE_COLS` and the notebook's section 3 for details."
    )


# --------------------------------------------------------------------------- #

if page == "Portfolio overview":
    portfolio_page()
elif page == "Project deep-dive":
    deep_dive_page()
else:
    model_quality_page()
