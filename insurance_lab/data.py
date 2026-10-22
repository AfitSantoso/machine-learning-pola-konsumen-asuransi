"""Point-in-time synthetic data. Internal shocks are never predictive columns.

All covariates are generated before outcomes. Economic indices are invented;
this laboratory is not an empirical study of Indonesian consumers.
"""
import numpy as np
import pandas as pd

from .config import FEATURES, TARGET

REGIONS = ["Jabodetabek", "Jawa_lain", "Sumatera", "Kalimantan", "Indonesia_timur"]


def simulate_policies(n_customers=6000, seed=26):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    n = n_customers
    entry = np.where(rng.random(n) < .62, 0, rng.integers(1, len(months) - 6, n))
    initial_tenure = np.where(entry == 0, rng.integers(1, 49, n), 0)
    income_band = rng.choice(["Low", "Mid", "High"], n, p=[.36, .45, .19])
    income = np.select([income_band == "Low", income_band == "Mid"],
                       [3_800_000, 8_500_000], default=19_000_000)
    base_premium = rng.integers(180, 850, n) * 1000
    age = rng.integers(22, 66, n)
    region = rng.choice(REGIONS, n, p=[.34, .29, .18, .11, .08])
    product = rng.choice(["Kesehatan", "Jiwa", "Kendaraan"], n, p=[.5, .32, .18])
    unobserved_risk, sensitivity = rng.normal(0, .45, n), rng.beta(2, 3, n)
    active = np.ones(n, dtype=bool)
    late_history, complaint_history, claim_history = [], [], []
    contact_recency = rng.integers(1, 120, n)
    records = []
    for i, date in enumerate(months):
        eligible = active & (entry <= i)
        hike = int(date >= pd.Timestamp("2025-01-01"))
        delayed_shock = int(date.year >= 2025 and date.month in [4, 5, 6])
        stress = .24 + .07 * np.sin(i / 4) + .48 * delayed_shock
        tenure = np.maximum(initial_tenure + i - entry, 0)
        to_renewal = (12 - tenure % 12) % 12
        renewal = to_renewal <= 1
        # Activity occurs during the month BEFORE the snapshot.
        late = rng.binomial(1, np.clip(.035 + .16 * (income_band == "Low")
                                     + .22 * stress + .10 * sensitivity, 0, .9))
        complaint = rng.binomial(1, .035 + .13 * delayed_shock, n)
        claim = rng.binomial(1, .06 + .07 * (product == "Kesehatan"))
        # Remove pre-inception activity before rolling sums or target generation.
        pre_inception = entry >= i
        late[pre_inception], complaint[pre_inception], claim[pre_inception] = 0, 0, 0
        late_history.append(late)
        complaint_history.append(complaint)
        claim_history.append(claim)
        lp, cp, cl = [np.sum(h[-3:], axis=0) for h in
                      [late_history, complaint_history, claim_history]]
        contact_recency = np.where(rng.random(n) < .22, rng.integers(1, 28, n),
                                   np.minimum(contact_recency + 30, 365))
        sessions = rng.poisson(np.clip(4.5 - .65 * lp - .7 * cp, .2, 8))
        sessions[(entry == i) & (entry > 0)] = 0
        premium = base_premium * (1 + .10 * hike)
        logit = (-5.0 + .66 * lp + .65 * cp + .75 * renewal
                 + .48 * (income_band == "Low") + .003 * contact_recency
                 - .10 * sessions + .85 * stress + .08 * hike
                 + .65 * renewal * lp + .75 * delayed_shock * sensitivity
                 + unobserved_risk)
        outcome = rng.binomial(1, 1 / (1 + np.exp(-logit))) * eligible
        block = pd.DataFrame({
            "customer_id": np.arange(1, n + 1), "snapshot_date": date,
            "feature_as_of": date - pd.Timedelta(days=1),
            "label_end_date": date + pd.offsets.MonthBegin(1),
            "entry_date": [months[j] for j in entry],
            "is_inception_cohort": entry > 0,
            "age": age, "region": region, "product": product,
            "income_band": income_band, "tenure_months": tenure,
            "premium_idr": premium.astype(int), "income_monthly_idr": income,
            "premium_change_pct": .10 * hike,
            "economic_stress_index": round(stress, 4),
            "months_to_renewal": to_renewal, "late_payments_3m": lp,
            "complaints_3m": cp, "claims_3m": cl,
            "days_since_contact": contact_recency, "app_sessions_1m": sessions,
            TARGET: outcome.astype(int),
        })
        records.append(block.loc[eligible].copy())
        active &= outcome == 0
    raw = pd.concat(records, ignore_index=True)
    raw.loc[rng.random(len(raw)) < .012, "app_sessions_1m"] = np.nan
    raw.loc[rng.random(len(raw)) < .006, "region"] = np.nan
    return raw


def engineer_policy_features(raw):
    df = raw.copy()
    df["premium_to_income"] = df.premium_idr / df.income_monthly_idr
    df["renewal_payment_interaction"] = (df.months_to_renewal <= 1) * df.late_payments_3m
    return df


def simulate_acquisition(seed=26):
    rng = np.random.default_rng(seed + 100)
    rows = []
    weeks = pd.date_range("2023-01-02", "2026-09-21", freq="W-MON")
    for r, region in enumerate(REGIONS):
        campaigns = []
        for i, week in enumerate(weeks):
            campaign = int(i % 13 in [4, 5])
            campaigns.append(campaign)
            lag_campaign = campaigns[-3] if len(campaigns) >= 3 else 0
            demand = int((week.year == 2025 and week.month in [8, 9]) or
                         (week.year == 2026 and week.month in [7, 8]))
            search = max(5, 35 + 32 * demand + 8 * np.sin(i * 2 * np.pi / 52) + rng.normal(0, 4))
            referrals = rng.poisson(12 + 24 * demand)
            leads = max(10, int(90 + 1.5 * search + referrals + 18 * campaign + rng.normal(0, 15)))
            previous = rows[-1] if i > 0 else None
            mu = (8 + .10 * (previous["inbound_leads"] if previous else leads)
                  + .22 * (previous["referrals"] if previous else referrals)
                  + 8 * lag_campaign) * (1.1 - .10 * r)
            rows.append({"week": week, "region": region,
                         "campaign_planned": campaign, "inbound_leads": leads,
                         "search_index": round(search, 3), "referrals": referrals,
                         "new_policies": rng.poisson(mu)})
    return pd.DataFrame(rows)


def engineer_acquisition_features(raw):
    df = raw.sort_values(["region", "week"]).copy()
    grouped = df.groupby("region", sort=False)
    for source, lag, name in [
        ("new_policies", 1, "new_policies_lag1"),
        ("new_policies", 4, "new_policies_lag4"),
        ("inbound_leads", 1, "leads_lag1"),
        ("search_index", 1, "search_index_lag1"),
        ("referrals", 1, "referrals_lag1"),
        ("campaign_planned", 2, "campaign_lag2"),
    ]:
        df[name] = grouped[source].shift(lag)
    angle = df.week.dt.isocalendar().week.astype(float) * 2 * np.pi / 52.18
    df["season_sin"], df["season_cos"] = np.sin(angle), np.cos(angle)
    return df.dropna().reset_index(drop=True)


def audit_policy_data(df):
    """Reject invalid keys, timestamps, labels and post-churn snapshots."""
    assert not df.duplicated(["customer_id", "snapshot_date"]).any(), "Duplicate snapshot"
    assert (df.feature_as_of < df.snapshot_date).all(), "Future feature timestamp"
    assert (df.snapshot_date < df.label_end_date).all(), "Invalid label window"
    assert df[TARGET].isin([0, 1]).all(), "Nonbinary target"
    assert (df.premium_idr > 0).all(), "Nonpositive premium"
    assert not set(FEATURES) & {TARGET, "customer_id", "label_end_date"}
    ordered = df.sort_values(["customer_id", "snapshot_date"])
    prior = ordered.groupby("customer_id")[TARGET].cumsum() - ordered[TARGET]
    assert (prior == 0).all(), "Snapshot after terminal churn"
    return {"rows": len(df), "customers": int(df.customer_id.nunique()),
            "months": int(df.snapshot_date.nunique()), "duplicate_keys": 0,
            "churn_prevalence": float(df[TARGET].mean()),
            "missing_by_column": df.isna().sum().to_dict()}
