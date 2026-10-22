"""Batch inference from a trusted local model bundle; no training required."""
import argparse
from pathlib import Path
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    missing = sorted(set(bundle["features"]) - set(frame.columns))
    if missing:
        parser.error(f"Missing feature columns: {missing}")
    result = frame[[c for c in ["customer_id", "snapshot_date", "week", "region"] if c in frame]].copy()
    if bundle.get("task", "churn") == "churn":
        result["churn_probability"] = bundle["pipeline"].predict_proba(frame[bundle["features"]])[:, 1]
        result["above_validation_threshold"] = result.churn_probability >= bundle["threshold"]
    else:
        result["expected_new_policies"] = (
            frame.new_policies_lag1 if bundle["selected_model"] == "lag1_naive"
            else bundle["pipeline"].predict(frame[bundle["features"]]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
