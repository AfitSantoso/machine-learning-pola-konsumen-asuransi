"""Temporal training, untouched final evaluation and reusable model artifacts."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, mean_absolute_error,
                             mean_poisson_deviance, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (ACQ_FEATURES, ACQ_NUMERIC, CATEGORICAL, FEATURES,
                     NUMERIC, TARGET, split_months)


def preprocess(numeric=NUMERIC, categorical=CATEGORICAL):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False))]), categorical),
    ])


def candidate_models(seed=26):
    models = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=600, C=1.0, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=130, max_leaf_nodes=15, min_samples_leaf=55,
            l2_regularization=8, learning_rate=.06,
            early_stopping=False, random_state=seed),
    }
    return {name: Pipeline([("preprocess", preprocess()), ("model", model)])
            for name, model in models.items()}


def capacity_selection(frame, fraction=.10):
    """Rank separately each month: respects operational monthly capacity."""
    chosen = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("snapshot_date"):
        k = max(1, int(np.ceil(len(group) * fraction)))
        idx = group.sort_values(["score", "customer_id"], ascending=[False, True]).index[:k]
        chosen.loc[idx] = True
    return chosen


def classifier_metrics(frame):
    y, p = frame[TARGET], frame.score
    selected = capacity_selection(frame)
    precision = float(y.loc[selected].mean())
    return {"rows": len(frame), "positives": int(y.sum()),
            "prevalence": float(y.mean()), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "precision_at_10pct": precision,
            "recall_at_10pct": float(y.loc[selected].sum() / y.sum()),
            "lift_at_10pct": precision / float(y.mean())}


def rolling_backtest(df, seed):
    rows = []
    folds = [("2024-06-01", "2024-08-01", "2024-10-01"),
             ("2024-09-01", "2024-11-01", "2025-01-01"),
             ("2024-12-01", "2025-02-01", "2025-04-01")]
    for fold, (end, start, stop) in enumerate(folds, 1):
        train = df.loc[df.snapshot_date <= end]
        valid = df.loc[(df.snapshot_date >= start) & (df.snapshot_date <= stop)]
        assert train.label_end_date.max() < valid.snapshot_date.min()
        for name, model in candidate_models(seed).items():
            model.fit(train[FEATURES], train[TARGET])
            scored = valid[["customer_id", "snapshot_date", TARGET]].copy()
            scored["score"] = model.predict_proba(valid[FEATURES])[:, 1]
            rows.append({"fold": fold, "model": name, "train_end": end,
                         "validation_start": start, "validation_end": stop,
                         **classifier_metrics(scored)})
    return pd.DataFrame(rows)


def cluster_intervals(scored, seed=26, repeats=120):
    """Customer bootstrap preserves repeat rows, conditional on fitted model.

    Temporal shocks shared across people are not captured by this interval.
    """
    rng = np.random.default_rng(seed)
    codes, ids = pd.factorize(scored.customer_id)
    chosen = capacity_selection(scored).to_numpy()
    y, p = scored[TARGET].to_numpy(), scored.score.to_numpy()
    values = []
    for _ in range(repeats):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        weights = counts[codes]
        prevalence = np.average(y, weights=weights)
        values.append([average_precision_score(y, p, sample_weight=weights),
                       np.average(y[chosen], weights=weights[chosen]) / prevalence])
    limits = np.quantile(values, [.025, .975], axis=0)
    return {name: {"lower_95": float(limits[0, i]), "upper_95": float(limits[1, i])}
            for i, name in enumerate(["average_precision", "lift_at_10pct"])}


def train_churn(df, out, seed=26):
    tables, models_dir = out / "artifacts/tables", out / "artifacts/models"
    splits = split_months(df)
    train, valid, test = [splits[s] for s in ["train", "validation", "test"]]
    assert train.label_end_date.max() < valid.snapshot_date.min()
    assert valid.label_end_date.max() < test.snapshot_date.min()
    backtest = rolling_backtest(df, seed)
    backtest.to_csv(tables / "rolling_backtest.csv", index=False)
    models, metric_rows, predictions = candidate_models(seed), [], {}
    # Candidate parameters are fixed before examining validation or final test.
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        for split, frame in [("validation", valid), ("test", test)]:
            pred = frame[["customer_id", "snapshot_date", "region", "income_band", TARGET]].copy()
            pred["score"] = model.predict_proba(frame[FEATURES])[:, 1]
            predictions[(name, split)] = pred
            metric_rows.append({"model": name, "split": split, **classifier_metrics(pred)})
    metrics = pd.DataFrame(metric_rows)
    selection = metrics.loc[(metrics.split == "validation") &
                            (metrics.model != "prevalence_baseline")]
    winner = selection.sort_values(["average_precision", "brier"],
                                   ascending=[False, True]).iloc[0].model
    model = models[winner]
    validation_scores = predictions[(winner, "validation")]
    scored = predictions[(winner, "test")].copy()
    threshold = float(validation_scores.score.quantile(.90))
    scored["above_validation_threshold"] = scored.score >= threshold
    scored["selected_top10_month"] = capacity_selection(scored)
    # Expected-value scenario: save_fraction is conditional rescue probability,
    # not a measured treatment effect. No achieved profit is claimed.
    policy = test[["premium_idr", "months_to_renewal", "late_payments_3m", "complaints_3m"]]
    scored = scored.join(policy)
    scored["assumed_retained_margin_idr"] = scored.premium_idr * 12 * .20
    scored["scenario_net_value_idr"] = scored.score * .20 * scored.assumed_retained_margin_idr - 15_000
    scored["suggested_review"] = np.select(
        [scored.complaints_3m > 0, scored.late_payments_3m > 0,
         scored.months_to_renewal <= 1],
        ["Tinjau keluhan layanan", "Tawarkan bantuan pembayaran", "Pengingat renewal"],
        default="Tinjau kebutuhan proteksi")
    metrics["selected_on_validation"] = metrics.model == winner
    metrics.to_csv(tables / "churn_metrics.csv", index=False)
    scored.to_csv(tables / "churn_test_predictions.csv", index=False)
    per_month = []
    for month, group in scored.groupby("snapshot_date"):
        if group[TARGET].nunique() > 1:
            per_month.append({"snapshot_date": month, **classifier_metrics(group)})
    pd.DataFrame(per_month).to_csv(tables / "churn_metrics_by_month.csv", index=False)
    subgroup = []
    for col in ["region", "income_band"]:
        for name, group in scored.groupby(col, dropna=False):
            if group[TARGET].nunique() > 1:
                subgroup.append({"dimension": col, "group": name, **classifier_metrics(group)})
    pd.DataFrame(subgroup).to_csv(tables / "subgroup_diagnostics.csv", index=False)
    # Explanation uses validation, never optimizes against test importance.
    sample = valid.sample(min(4500, len(valid)), random_state=seed)
    imp = permutation_importance(model, sample[FEATURES], sample[TARGET],
                                 scoring="average_precision", n_repeats=4,
                                 random_state=seed, n_jobs=1)
    importance = pd.DataFrame({"feature": FEATURES, "ap_drop_mean": imp.importances_mean,
                               "ap_drop_std": imp.importances_std}).sort_values("ap_drop_mean", ascending=False)
    importance.to_csv(tables / "permutation_importance_validation.csv", index=False)
    confusion = confusion_matrix(scored[TARGET], scored.above_validation_threshold, labels=[0, 1])
    pd.DataFrame(confusion, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        tables / "confusion_matrix.csv")
    capacity_rows = []
    for fraction in [.05, .10, .20]:
        picked = capacity_selection(scored, fraction)
        for rescue in [.05, .10, .20, .30]:
            net = scored.score * rescue * scored.assumed_retained_margin_idr - 15_000
            capacity_rows.append({"capacity": fraction, "assumed_rescue_fraction": rescue,
                                  "contacts": int(picked.sum()),
                                  "precision": float(scored.loc[picked, TARGET].mean()),
                                  "hypothetical_net_value_idr": float(net.loc[picked].sum())})
    pd.DataFrame(capacity_rows).to_csv(tables / "retention_sensitivity.csv", index=False)
    bundle = {"task": "churn", "pipeline": model, "features": FEATURES, "model_name": winner,
              "threshold": threshold, "seed": seed, "train_end": "2025-06-01",
              "target": TARGET, "synthetic": True}
    joblib.dump(bundle, models_dir / "churn_model.joblib")
    example = test.loc[test.snapshot_date == test.snapshot_date.max(),
                       ["customer_id", "snapshot_date"] + FEATURES].copy()
    example.head(50).to_csv(out / "data/processed/scoring_example.csv", index=False)
    return {"selected_model": winner, "metrics": metrics, "scored": scored,
            "importance": importance, "threshold": threshold, "confusion": confusion,
            "intervals": cluster_intervals(scored, seed), "splits": splits}


def train_acquisition(df, out):
    train = df.loc[df.week < "2025-07-01"]
    valid = df.loc[(df.week >= "2025-08-01") & (df.week < "2026-01-01")]
    test = df.loc[df.week >= "2026-02-01"]
    # One-step-ahead weekly backtest. Each lag1 is observed before that week.
    poisson = Pipeline([("preprocess", preprocess(ACQ_NUMERIC, ["region"])),
                        ("model", PoissonRegressor(alpha=.2, max_iter=500))])
    poisson.fit(train[ACQ_FEATURES], train.new_policies)
    rows, results = [], {}
    for split, part in [("validation", valid), ("test", test)]:
        predictions = {"lag1_naive": part.new_policies_lag1.to_numpy(),
                       "poisson_regression": poisson.predict(part[ACQ_FEATURES])}
        for name, scores in predictions.items():
            scored = part[["week", "region", "new_policies"]].copy()
            scored["prediction"] = scores
            weekly = scored.groupby("week")[["new_policies", "prediction"]].sum()
            rows.append({"model": name, "split": split, "region_week_mae": mean_absolute_error(
                scored.new_policies, scores), "weekly_total_mae": mean_absolute_error(
                weekly.new_policies, weekly.prediction), "weekly_total_wape": float(
                (weekly.new_policies - weekly.prediction).abs().sum() / weekly.new_policies.sum()),
                "poisson_deviance": mean_poisson_deviance(scored.new_policies, np.maximum(scores, .001))})
            results[(name, split)] = scored
    metrics = pd.DataFrame(rows)
    winner = metrics.loc[metrics.split == "validation"].sort_values("weekly_total_mae").iloc[0].model
    scored = results[(winner, "test")].copy()
    # Operational residual thresholds fitted ONLY using validation residuals.
    residual = results[(winner, "validation")].copy()
    residual["residual"] = residual.new_policies - residual.prediction
    bounds = residual.residual.quantile([.025, .975]).tolist()
    scored["residual"] = scored.new_policies - scored.prediction
    scored["unusual_residual"] = (scored.residual < bounds[0]) | (scored.residual > bounds[1])
    metrics["selected_on_validation"] = metrics.model == winner
    metrics.to_csv(out / "artifacts/tables/acquisition_metrics.csv", index=False)
    scored.to_csv(out / "artifacts/tables/acquisition_test_predictions.csv", index=False)
    joblib.dump({"task": "acquisition", "pipeline": poisson if winner == "poisson_regression" else None,
                 "features": ACQ_FEATURES,
                 "selected_model": winner, "residual_bounds": bounds, "synthetic": True},
                out / "artifacts/models/acquisition_model.joblib")
    return {"metrics": metrics, "scored": scored, "selected_model": winner,
            "residual_bounds": bounds}
