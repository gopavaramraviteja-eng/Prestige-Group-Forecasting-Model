# Prestige Construction Forecasting — ML Notebook + Streamlit App

A forecasting pipeline built on top of the synthetic construction progress &
cost dataset (18 residential projects, 6 cities, 2 years of daily data). It
predicts day-to-day construction progress and cost, then chains those
one-step predictions into multi-month forecasts with Monte Carlo uncertainty
bands — surfaced through both a Jupyter notebook (for analysis) and a
Streamlit app (for an interactive dashboard).

**Everything here is synthetic** — see `data/README_data_dictionary.md` for
how it was generated. Swap in real project exports with the same column
names and the whole pipeline keeps working unchanged.

## What's in this folder

| File | Purpose |
|---|---|
| `data/` | The 4 synthetic CSVs + their data dictionary |
| `forecasting_lib.py` | Shared library: feature engineering, training, evaluation, recursive Monte Carlo forecasting. Imported by everything else below. |
| `forecasting_model.ipynb` | The main notebook — EDA, model training/evaluation, a backtest, and a live forecast demo. Already executed, with outputs saved, so you can read it without re-running anything. |
| `train_and_save_models.py` | Headless script that does the same training as the notebook and saves the models to `models/`. Handy for CI or if you don't want to open Jupyter. |
| `app.py` | The Streamlit dashboard: portfolio overview, per-project deep-dive with a forecast chart, model-quality diagnostics. |
| `build_notebook.py` | Generates `forecasting_model.ipynb` from Python source (so the notebook's content is easy to diff/edit as plain code). Only needed if you want to change the notebook's structure — edit this and re-run it, rather than hand-editing the `.ipynb` JSON. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
pip install -r requirements.txt

# Option A: open the notebook (already executed — just read it, or Cell > Run All to rerun)
jupyter notebook forecasting_model.ipynb

# Option B: launch the dashboard directly
streamlit run app.py
```

The first time you run either the notebook or the app, it trains two
RandomForest models (~30–60 seconds) and caches them to `models/`. Every run
after that loads the cached models instantly. **Delete the `models/` folder**
to force a retrain — e.g. after you swap in real data, or after editing
`forecasting_lib.py`'s feature engineering.

## How the forecasting works

**Framing:** rather than one big multi-week model, we train two models to
answer a narrow question — *given everything known up to today, what happens
tomorrow?* — for (1) daily construction progress % and (2) daily cost
incurred. A `RandomForestRegressor` is trained on ~2 years of history, held
out on the most recent 90 days to check it isn't overfitting.

**Reaching further than one day:** `forecast_project()` in `forecasting_lib.py`
takes that one-step model and calls it repeatedly, day by day, feeding each
prediction back in as tomorrow's lagged input — a standard recursive
forecasting approach. Exogenous inputs the model needs but can't otherwise
know about the future (rainfall, macro material prices) are filled from
seasonal climatology / trend extrapolation rather than left blank.

**Uncertainty bands:** setting `n_simulations > 1` runs many such recursive
paths in parallel, each with randomised day-to-day noise (drawn from the
model's own validation residuals) and randomised delay-event risk (drawn from
the historical ~4.5% delay-day rate). Collapsing those paths into p10/p50/p90
per day gives a genuine forecast band instead of one brittle line — this is
what both the notebook's section 7 and the Streamlit app's project
deep-dive page plot.

**What's deliberately excluded:** the synthetic data generator used two
latent parameters (`mgmt_quality_factor`, `overrun_factor`) to control how
efficiently/over-budget each project runs. Those are simulation internals a
real deployment would never observe, so they're explicitly excluded from the
model's features (see `NON_FEATURE_COLS` in `forecasting_lib.py`) — including
them would leak the answer and make the model look artificially better than
it would be on real data.

## Using this on real data

Replace the 4 CSVs in `data/` with real exports using the exact same column
names (see `data/README_data_dictionary.md` for the schema), delete
`models/`, and rerun the notebook or `streamlit run app.py`. Nothing else
needs to change. If your real data has different columns (e.g. no cost
category breakdown), you'll only need to adjust `forecasting_lib.build_feature_table`
and the parts of `app.py` that reference `cost_breakdown`.

## Known limitations / good next steps

- RandomForest was chosen for simplicity and interpretable feature
  importances; gradient boosting (XGBoost/LightGBM) is worth trying for
  potentially better accuracy on a larger real dataset.
- The Monte Carlo band uses a fixed residual std and a fixed historical
  delay-event rate — a proper quantile regression model would give sharper,
  better-calibrated intervals.
- Forecast horizon and simulation count are both configurable in the
  Streamlit sidebar; larger values are slower (each simulation-day still
  needs one batched model prediction call).
- The app trains on first launch if `models/` is missing — for a shared/
  production deployment, run `train_and_save_models.py` once ahead of time
  and ship the `models/` folder alongside the app instead.
