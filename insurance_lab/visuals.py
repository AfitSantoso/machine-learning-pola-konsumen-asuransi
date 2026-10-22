"""Chart and process-diagram generation, entirely reproducible from model outputs."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import TARGET, NUMERIC

NAVY, BLUE, TEAL, ORANGE, RED = "#112640", "#2563eb", "#089f94", "#e89526", "#d94b64"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": NAVY,
                         "text.color": NAVY, "axes.prop_cycle": plt.cycler(color=[BLUE, TEAL, ORANGE, RED]),
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def diagram(path, title, items, subtitle, rows=2):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    fig.suptitle(title, fontsize=20, x=.06, ha="left", y=.96)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    cols = int(np.ceil(len(items) / rows))
    for i, (heading, body) in enumerate(items):
        row, col = divmod(i, cols)
        x, y, w, h = .02 + col * .245, .65 - row * .39, .215, .27
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012",
                                   facecolor=NAVY, edgecolor="none"))
        ax.text(x + .018, y + h - .055, f"{i + 1:02d}  {heading}", color="#57e1cf", weight="bold", fontsize=10)
        ax.text(x + .018, y + h - .115, textwrap.fill(body, 27), va="top", color="white", fontsize=10)
        if col < cols - 1 and i < len(items) - 1:
            ax.add_patch(FancyArrowPatch((x+w+.004, y+h/2), (x+.245-.012, y+h/2),
                                        arrowstyle="-|>", mutation_scale=15, color=TEAL))
    ax.text(.02, .04, subtitle, fontsize=10, color=NAVY)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)


def create_figures(df, acquisition_raw, acquisition, churn, acq, out):
    style()
    destination = out / "reports/figures"
    paths = []

    def save(fig, name):
        fig.text(.01, .005, "SYNTHETIC DATA | Insurance Behavior Lab | Bukan hasil bisnis nyata", fontsize=8, color="#64748b")
        fig.tight_layout(rect=(0, .025, 1, 1))
        path = destination / f"{name}.png"
        fig.savefig(path, dpi=165); paths.append(path); plt.close(fig)

    diagram(destination / "learning_path.png", "Learning path | dari pertanyaan ke keputusan", [
        ("Business", "Problem brief, target, unit analisis dan hipotesis"),
        ("Audit data", "Timestamp, missingness, duplikasi, censoring"),
        ("EDA", "Tren, cohort, segmen dan jeda perilaku"),
        ("Fitur", "Snapshot historis dan temporal split"),
        ("Model", "Baseline, logistik, gradient boosting, Poisson"),
        ("Validasi", "Backtest, AP, lift, MAE dan calibration"),
        ("Interpretasi", "Permutation importance dan batas kausal"),
        ("Keputusan", "Prioritas kontak, eksperimen dan monitoring"),
    ], "4 sesi x 4 jam | setiap tahap menghasilkan artefak yang dapat ditinjau")
    paths.append(destination / "learning_path.png")
    diagram(destination / "ml_process.png", "Proses machine learning | batch scoring", [
        ("Sumber", "Policy, pembayaran, keluhan dan CRM historis"),
        ("Validasi", "Kunci unik dan feature_as_of sebelum snapshot"),
        ("Feature store", "Agregat 3 bulan, renewal, rasio premi"),
        ("Training", "Pipeline imputer, encoder dan estimator"),
        ("Evaluasi", "Validation memilih model, test terakhir"),
        ("Artifact", "Pipeline joblib, versi dan feature contract"),
        ("Scoring", "Probabilitas dan kapasitas per bulan"),
        ("Feedback", "Holdout intervensi, outcome dan drift"),
    ], "Pipeline tutorial dijalankan lokal; CRM dan eksperimen adalah rancangan pengembangan.")
    paths.append(destination / "ml_process.png")
    diagram(destination / "hypothesis_map.png", "Peta hipotesis | panah belum membuktikan penyebab", [
        ("Harga", "Premi naik 10% pada Januari 2025"),
        ("Keterjangkauan", "Anggaran tertekan; dampak dapat tertunda"),
        ("Pembayaran", "Tunggakan dan pengalaman layanan"),
        ("Renewal", "Keputusan bertahan atau berhenti"),
        ("Konteks", "Kebutuhan proteksi, kabar, kompetitor"),
        ("Pencarian", "Minat organik dan referral"),
        ("Campaign", "Exposure, jeda respons dan channel"),
        ("Akuisisi", "Lead, conversion dan polis baru"),
    ], "Confounder dan faktor tak tercatat dapat memengaruhi banyak simpul sekaligus.")
    paths.append(destination / "hypothesis_map.png")
    monthly = df.groupby("snapshot_date").agg(active=(TARGET, "size"), churned=(TARGET, "sum"),
        churn_rate=(TARGET, "mean"), premium=("premium_idr", "mean"),
        late=("late_payments_3m", "mean"), complaints=("complaints_3m", "mean"),
        stress=("economic_stress_index", "mean"))
    monthly.to_csv(out / "artifacts/tables/monthly_portfolio.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(monthly.index, monthly.churn_rate * 100, marker=".", lw=2)
    axes[0].axvline(pd.Timestamp("2025-01-01"), color=ORANGE, ls="--", label="Premi +10%")
    for year in [2025, 2026]:
        axes[0].axvspan(pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-06-30"), color=RED, alpha=.10)
    axes[0].set(title="Premi naik sekarang; lonjakan churn muncul pada periode lain", ylabel="Churn bulanan (%)")
    axes[0].legend()
    axes[1].bar(monthly.index, monthly.active, width=22, alpha=.8, label="Polis aktif di awal bulan")
    ax2 = axes[1].twinx(); ax2.plot(monthly.index, monthly.churned, color=RED, label="Jumlah churn")
    axes[1].set_ylabel("Denominator aktif"); ax2.set_ylabel("Jumlah churn", color=RED)
    save(fig, "churn_trend")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    missing = df.isna().mean().sort_values(ascending=False).head(6) * 100
    axes[0].barh(missing.index[::-1], missing.values[::-1]); axes[0].set(title="Data quality: missingness", xlabel="% baris")
    axes[1].hist(df.tenure_months, bins=25, color=TEAL); axes[1].set(title="Distribusi tenure", xlabel="Bulan", ylabel="Snapshot")
    save(fig, "data_quality")
    segmented = df.assign(renewal=np.where(df.months_to_renewal <= 1, "Dekat renewal", "Di luar renewal"),
                          payment=np.where(df.late_payments_3m > 0, "Ada tunggakan", "Tanpa tunggakan"))
    table = segmented.groupby(["renewal", "payment"])[TARGET].agg(["mean", "size", "sum"]).reset_index()
    table.to_csv(out / "artifacts/tables/segment_renewal_payment.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(table.renewal + " | " + table.payment, table["mean"] * 100, color=[BLUE, TEAL, BLUE, TEAL])
    for i, row in table.iterrows():
        ax.text(row["mean"] * 100 + .1, i, f"n={int(row['size']):,}", va="center", fontsize=9)
    ax.set(xlabel="Churn (%)", title="Interaksi renewal dan riwayat pembayaran | deskriptif"); ax.margins(x=.25)
    save(fig, "renewal_payment")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for col in ["late", "complaints", "stress"]:
        standardized = (monthly[col] - monthly[col].mean()) / monthly[col].std()
        ax.plot(monthly.index, standardized, label=col)
    ax.set(title="Sinyal yang sudah tersedia sebelum outcome", ylabel="Z-score tiap seri (skala berbeda)"); ax.legend()
    save(fig, "leading_signals")
    # Inception cohorts only, full follow-up required at each displayed age.
    new = df.loc[df.is_inception_cohort].copy()
    new["cohort"] = new.entry_date.dt.to_period("Q").astype(str)
    counts = new.groupby(["cohort", "tenure_months"])[TARGET].agg(["sum", "size"])
    survival = (1 - counts["sum"] / counts["size"]).groupby(level=0).cumprod().unstack()
    survival = survival.reindex(columns=range(13))
    for cohort, group in new.groupby("cohort"):
        max_age = (df.snapshot_date.max().to_period("M") - group.entry_date.max().to_period("M")).n
        survival.loc[cohort, survival.columns > max_age] = np.nan
    survival.to_csv(out / "artifacts/tables/cohort_retention.csv")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(survival, vmin=.4, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(len(survival)), survival.index)
    ax.set_xticks(range(13), range(1, 14))
    ax.set(title="Retention cohort baru | sel kosong = follow-up belum lengkap",
           xlabel="Akhir bulan ke- sejak inception", ylabel="Kuartal inception")
    fig.colorbar(im, ax=ax, label="Probabilitas bertahan (product-limit)")
    save(fig, "cohort_retention")
    weekly = acquisition_raw.groupby("week").agg(new_policies=("new_policies", "sum"),
        leads=("inbound_leads", "sum"), campaign=("campaign_planned", "max"),
        referrals=("referrals", "sum"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    axes[0].plot(weekly.index, weekly.new_policies, label="Polis baru")
    mask = weekly.campaign == 1
    axes[0].scatter(weekly.index[mask], weekly.new_policies[mask], facecolors="none", edgecolors=ORANGE, label="Minggu campaign")
    axes[0].set(title="Akuisisi mingguan | respons campaign dapat memiliki jeda", ylabel="Polis baru"); axes[0].legend()
    axes[1].plot(weekly.index, weekly.leads, label="Lead"); axes[1].plot(weekly.index, weekly.referrals, label="Referral")
    axes[1].set_ylabel("Volume"); axes[1].legend()
    save(fig, "acquisition_trend")
    lag_corr = pd.DataFrame({"lag_weeks": range(7), "correlation": [
        weekly.campaign.shift(lag).corr(weekly.new_policies) for lag in range(7)]})
    lag_corr.to_csv(out / "artifacts/tables/campaign_lag_correlations.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.7)); ax.bar(lag_corr.lag_weeks, lag_corr.correlation)
    ax.set(title="Campaign vs akuisisi: korelasi pada beberapa lag", xlabel="Lag minggu", ylabel="Korelasi Pearson (bukan efek kausal)")
    save(fig, "campaign_lag")
    metrics = churn["metrics"]; test_metrics = metrics.loc[metrics.split == "test"]
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.barh(test_metrics.model, test_metrics.average_precision)
    ax.axvline(test_metrics.prevalence.iloc[0], ls="--", color=ORANGE, label="Prevalensi test")
    ax.set(title="Perbandingan model | test terakhir", xlabel="Average precision (AP)"); ax.legend()
    save(fig, "model_comparison")
    scored = churn["scored"]
    y, p = scored[TARGET], scored.score
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0].plot(recall, precision); axes[0].axhline(y.mean(), color=ORANGE, ls="--")
    axes[0].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
    fpr, tpr, _ = roc_curve(y, p)
    axes[1].plot(fpr, tpr); axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    save(fig, "pr_roc")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    observed, predicted = calibration_curve(y, p, n_bins=8, strategy="quantile")
    ax.plot(predicted, observed, marker="o", label=churn["selected_model"])
    extent = max(max(observed), max(predicted)) * 1.1
    ax.plot([0, extent], [0, extent], "--", color="gray", label="Kalibrasi ideal")
    ax.set(title="Reliability diagram | test terakhir", xlabel="Rata-rata probabilitas", ylabel="Proporsi churn aktual"); ax.legend()
    save(fig, "calibration")
    deciles = []
    for _, group in scored.groupby("snapshot_date"):
        group = group.sort_values(["score", "customer_id"], ascending=[False, True]).copy()
        group["decile"] = np.minimum(10, (np.arange(len(group)) * 10 / len(group)).astype(int) + 1)
        deciles.append(group)
    lift = pd.concat(deciles).groupby("decile")[TARGET].mean() / y.mean()
    fig, ax = plt.subplots(figsize=(9, 4.7)); ax.bar(lift.index, lift.values); ax.axhline(1, ls="--", c=ORANGE)
    ax.set(title="Lift per desil | ranking dilakukan terpisah setiap bulan", xlabel="Desil risiko (1 tertinggi)", ylabel="Lift vs prevalensi test")
    save(fig, "decile_lift")
    top = churn["importance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5)); ax.barh(top.feature, top.ap_drop_mean, xerr=top.ap_drop_std, color=TEAL)
    ax.set(title="Permutation importance | validation, 4 pengulangan", xlabel="Penurunan AP ketika fitur diacak (bukan efek kausal)")
    save(fig, "feature_importance")
    fig, ax = plt.subplots(figsize=(6.7, 4.7)); matrix = churn["confusion"]
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=19,
                color="white" if value > matrix.max() / 2 else NAVY)
    ax.set_xticks([0, 1], ["Bertahan", "Churn"]); ax.set_yticks([0, 1], ["Bertahan", "Churn"])
    ax.set(title="Confusion matrix | threshold dari validation", xlabel="Prediksi", ylabel="Aktual")
    save(fig, "confusion_matrix")
    forecast = acq["scored"].groupby("week")[["new_policies", "prediction"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(forecast.index, forecast.new_policies, label="Aktual")
    ax.plot(forecast.index, forecast.prediction, label="Prediksi 1 minggu", ls="--")
    ax.set(title="Backtest akuisisi | one-step-ahead, test terakhir", ylabel="Total polis baru / minggu"); ax.legend()
    save(fig, "acquisition_forecast")
    sensitivity = pd.read_csv(out / "artifacts/tables/retention_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for capacity, group in sensitivity.groupby("capacity"):
        ax.plot(group.assumed_rescue_fraction * 100, group.hypothetical_net_value_idr / 1e6,
                marker="o", label=f"Kontak top {capacity:.0%}")
    ax.axhline(0, ls="--", c="gray")
    ax.set(title="Skenario biaya-manfaat | efek intervensi BELUM diukur",
           xlabel="Asumsi proporsi calon churn yang berhasil diselamatkan (%)", ylabel="Nilai harapan bersih (Rp juta)"); ax.legend()
    save(fig, "business_sensitivity")
    train = churn["splits"]["train"]; test = churn["splits"]["test"]
    shifts = ((test[NUMERIC].mean() - train[NUMERIC].mean()) / train[NUMERIC].std()).dropna().sort_values()
    shifts.to_csv(out / "artifacts/tables/standardized_mean_shift.csv", header=["standardized_mean_difference"])
    fig, ax = plt.subplots(figsize=(11, 5)); ax.barh(shifts.index, shifts.values)
    ax.axvline(0, c="gray", lw=1); ax.set(title="Perubahan distribusi train ke test", xlabel="Selisih mean / standar deviasi train (diagnostik deskriptif)")
    save(fig, "distribution_shift")
    # A compact management page combines the four most useful operational views.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0, 0].plot(monthly.index, monthly.churn_rate * 100); axes[0, 0].set(title="Churn bulanan", ylabel="%")
    axes[0, 1].plot(weekly.index, weekly.new_policies, color=TEAL); axes[0, 1].set(title="Polis baru mingguan")
    axes[1, 0].bar(lift.index, lift.values); axes[1, 0].axhline(1, ls="--", c=ORANGE); axes[1, 0].set(title="Lift per desil", xlabel="Desil")
    axes[1, 1].barh(top.feature.tail(5), top.ap_drop_mean.tail(5), color=TEAL); axes[1, 1].set(title="Sinyal prediktif (validation)")
    fig.suptitle("Insurance Behavior Lab | Executive dashboard", fontsize=18, weight="bold")
    save(fig, "executive_dashboard")
    with PdfPages(out / "reports/visual_atlas.pdf") as pdf:
        for path in paths:
            fig, ax = plt.subplots(figsize=(13, 8)); ax.imshow(plt.imread(path)); ax.axis("off")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return monthly, paths
