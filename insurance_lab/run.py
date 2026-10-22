"""Orchestrate complete data, EDA, models, reporting and teaching artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits
from .config import ROOT
from .data import (audit_policy_data, engineer_acquisition_features,
                   engineer_policy_features, simulate_acquisition, simulate_policies)
from .modeling import train_acquisition, train_churn
from .notebooks import create_notebooks
from .presentation import create_presentation
from .reporting import write_reports
from .visuals import create_figures

LOG = logging.getLogger("insurance_lab")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--skip-slides", action="store_true")
    args = parser.parse_args()
    if args.customers < 1000:
        parser.error("Use at least 1000 customers for stable evaluation groups.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    start = perf_counter()
    out = args.output_dir.resolve()
    for folder in ["data/raw", "data/processed", "artifacts/models", "artifacts/tables",
                   "reports/figures", "slides", "notebooks"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    if out != ROOT:
        # Keep experiment outputs self-contained: report links and notebook imports
        # must still work when the output directory is outside the source tree.
        for folder in ["insurance_lab", "docs", "scripts", "tests"]:
            shutil.copytree(ROOT / folder, out / folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename in ["build_case.py", "requirements.txt", "requirements-lock.txt", "README.md"]:
            shutil.copy2(ROOT / filename, out / filename)
    LOG.info("1/6 Simulating point-in-time policy and acquisition data")
    raw = simulate_policies(args.customers, args.seed)
    raw.to_csv(out / "data/raw/policy_monthly_synthetic.csv", index=False)
    df = engineer_policy_features(raw)
    df.to_csv(out / "data/processed/policy_features.csv", index=False)
    acq_raw = simulate_acquisition(args.seed)
    acq_raw.to_csv(out / "data/raw/acquisition_weekly_synthetic.csv", index=False)
    acq_features = engineer_acquisition_features(acq_raw)
    acq_features.to_csv(out / "data/processed/acquisition_features.csv", index=False)
    audit = audit_policy_data(df)
    (out / "artifacts/data_audit.json").write_text(json.dumps(audit, indent=2, default=int), encoding="utf-8")
    LOG.info("2/6 Training churn baselines, rolling validation and final test")
    # Modest thread count avoids CPU oversubscription on participant laptops.
    with threadpool_limits(limits=2):
        churn = train_churn(df, out, args.seed)
        LOG.info("3/6 Fitting weekly acquisition model and baseline")
        acquisition = train_acquisition(acq_features, out)
    LOG.info("4/6 Creating EDA, diagnostics, diagrams and visual atlas")
    monthly, figures = create_figures(df, acq_raw, acq_features, churn, acquisition, out)
    LOG.info("5/6 Writing insight report, offline dashboard and notebooks")
    summary = write_reports(churn, acquisition, monthly, audit, figures, out, args.seed)
    create_notebooks(out)
    slide_count = 0
    if not args.skip_slides:
        LOG.info("6/6 Creating editable PowerPoint and companion PDF")
        slide_count = create_presentation(summary, out)
    packages = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "joblib",
                "python-pptx", "nbformat", "nbclient", "ipykernel"]
    checksums = {}
    for folder in ["data", "reports", "slides", "artifacts/models", "artifacts/tables", "notebooks"]:
        for path in sorted((out / folder).rglob("*")):
            if path.is_file():
                checksums[str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"seed": args.seed, "requested_customers": args.customers,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {package: importlib.metadata.version(package) for package in packages},
                "churn_model": churn["selected_model"], "acquisition_model": acquisition["selected_model"],
                "slides": slide_count, "figures": len(figures), "sha256": checksums,
                "runtime_seconds": round(perf_counter() - start, 2)}
    (out / "artifacts/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Complete: %s slides, %s figures, churn model=%s, output=%s (%.1fs)",
             slide_count, len(figures), churn["selected_model"], out, perf_counter() - start)


if __name__ == "__main__":
    main()
