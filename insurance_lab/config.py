"""Shared feature contracts; IDs, targets and latent truth are excluded."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "churn_next_month"
NUMERIC = ["age", "tenure_months", "premium_idr", "premium_change_pct",
           "economic_stress_index", "months_to_renewal", "late_payments_3m",
           "complaints_3m", "claims_3m", "days_since_contact", "app_sessions_1m",
           "premium_to_income", "renewal_payment_interaction"]
CATEGORICAL = ["region", "product", "income_band"]
FEATURES = NUMERIC + CATEGORICAL
ACQ_NUMERIC = ["new_policies_lag1", "new_policies_lag4", "leads_lag1",
               "search_index_lag1", "referrals_lag1", "campaign_lag2",
               "campaign_planned", "season_sin", "season_cos"]
ACQ_FEATURES = ACQ_NUMERIC + ["region"]


def split_months(frame):
    """A full month embargo allows historical outcome labels to mature."""
    date = frame["snapshot_date"]
    return {
        "train": frame.loc[date <= "2025-06-01"].copy(),
        "validation": frame.loc[(date >= "2025-08-01") & (date <= "2025-12-01")].copy(),
        "test": frame.loc[date >= "2026-02-01"].copy(),
    }
