"""
train_and_save_models.py
-------------------------
Headless training script: builds features from the synthetic construction
dataset, trains the two RandomForest models (next-day progress increment,
next-day cost), evaluates them on a held-out time window, and saves the
trained models + feature list + metrics to ./models/ so the Streamlit app
(app.py) can load them without needing to retrain or run the notebook.

Usage:
    python train_and_save_models.py [--data-dir data] [--models-dir models] [--test-days 90]
"""

import argparse
import datetime
import time

import numpy as np

import forecasting_lib as fl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading data from '{args.data_dir}' ...")
    data = fl.load_data(args.data_dir)
    for name, df in data.items():
        print(f"  {name}: {df.shape}")

    print("Building feature table ...")
    feat = fl.build_feature_table(data["progress"], data["projects"])
    feature_cols = fl.get_feature_columns(feat)
    print(f"  {feat.shape[0]} rows, {len(feature_cols)} features")

    train, test = fl.time_split(feat, test_days=args.test_days)
    print(f"  train: {train.shape[0]} rows (up to {train.date.max().date()})")
    print(f"  test:  {test.shape[0]} rows ({test.date.min().date()} .. {test.date.max().date()})")

    X_train, X_test = train[feature_cols], test[feature_cols]

    print(f"Training progress-increment model (n_estimators={args.n_estimators}) ...")
    progress_model = fl.train_rf(X_train, train["target_next_increment_pct"], n_estimators=args.n_estimators)
    progress_metrics = fl.evaluate(progress_model, X_test, test["target_next_increment_pct"])
    print("  progress model test metrics:", progress_metrics)

    print(f"Training daily-cost model (n_estimators={args.n_estimators}) ...")
    cost_model = fl.train_rf(X_train, train["target_next_cost_lakhs"], n_estimators=args.n_estimators)
    cost_metrics = fl.evaluate(cost_model, X_test, test["target_next_cost_lakhs"])
    print("  cost model test metrics:", cost_metrics)

    # Residual std on the held-out test set — used by the Streamlit app to size
    # Monte Carlo noise for forecast confidence bands.
    resid_std_increment = float(np.std(progress_model.predict(X_test) - test["target_next_increment_pct"]))
    resid_std_cost = float(np.std(cost_model.predict(X_test) - test["target_next_cost_lakhs"]))

    metrics = {
        "progress_model": progress_metrics,
        "cost_model": cost_metrics,
        "residual_std_increment": resid_std_increment,
        "residual_std_cost": resid_std_cost,
        "test_days": args.test_days,
        "n_estimators": args.n_estimators,
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "n_features": len(feature_cols),
        "trained_at": datetime.datetime.utcnow().isoformat(),
    }

    print(f"Saving artifacts to '{args.models_dir}' ...")
    fl.save_artifacts(args.models_dir, progress_model, cost_model, feature_cols, metrics)

    top_feat = fl.feature_importance(progress_model, feature_cols, top_n=10)
    print("Top progress-model features:")
    print(top_feat.to_string(index=False))

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
