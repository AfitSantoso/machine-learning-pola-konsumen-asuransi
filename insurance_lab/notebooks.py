"""Generate guided notebooks with real pipeline calls and discussion prompts."""
import nbformat as nbf


def create_notebooks(out):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md("""# Insurance Behavior Lab — EDA sampai model

Data **sintetis**, bukan fakta bisnis. Jalankan sel berurutan. Notebook memanggil modul yang sama
dengan pipeline CLI sehingga preprocessing dan evaluasi tetap konsisten.
Sasaran: memahami tiga misteri bisnis, membangun prediksi, dan merancang eksperimen.
Output notebook disimpan di `artifacts/notebook_run/` agar hasil utama tetap mudah dibandingkan.
"""), code("""from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
if not (ROOT / 'insurance_lab').exists():
    ROOT = ROOT.parent
assert (ROOT / 'insurance_lab').exists(), 'Buka notebook dari folder proyek atau notebooks/'
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, Image
from insurance_lab.config import TARGET, FEATURES, split_months
from insurance_lab.data import (simulate_policies, engineer_policy_features,
    simulate_acquisition, engineer_acquisition_features, audit_policy_data)
from insurance_lab.modeling import train_churn, train_acquisition
OUT = ROOT / 'artifacts/notebook_run'
for folder in ['artifacts/tables', 'artifacts/models', 'data/processed', 'reports/figures']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)
SEED = 26
"""), md("""## 1. Rumuskan masalah sebelum melihat model

Tulis dua hipotesis untuk setiap fenomena: premi naik tanpa churn langsung, churn melonjak
beberapa waktu kemudian, dan akuisisi meningkat di luar campaign.
Definisikan unit, horizon, outcome dan data tambahan yang diperlukan.
Jangan membuka formula simulator sebelum latihan hipotesis selesai.
"""), code("""raw = simulate_policies(n_customers=6000, seed=SEED)
df = engineer_policy_features(raw)
audit = audit_policy_data(df)
display(pd.Series({k: v for k, v in audit.items() if k != 'missing_by_column'}))
display(df.head())
"""), md("""## 2. Audit data dan waktu

Apa arti `feature_as_of`, `snapshot_date`, dan `label_end_date`?
Mengapa satu nasabah muncul berulang? Bolehkah ia muncul sesudah terminal churn?
Missingness diisi **di dalam pipeline yang fit pada train**, bukan di seluruh DataFrame.
"""), code("""display(df.isna().mean().sort_values(ascending=False).head(8))
display(df[FEATURES].describe(include='all').T)
assert (df.feature_as_of < df.snapshot_date).all()
assert (df.label_end_date > df.snapshot_date).all()
"""), md("""## 3. EDA: rate, denominator, segmen dan cohort

Bandingkan churn rate dan jumlah churn. Apakah perubahan jumlah pelanggan aktif mengubah
interpretasi? Hipotesis mana yang sesuai dengan pola renewal dan riwayat pembayaran?
"""), code("""monthly = df.groupby('snapshot_date').agg(
    active=(TARGET, 'size'), churned=(TARGET, 'sum'), churn_rate=(TARGET, 'mean'))
display(monthly.tail(12))
monthly.churn_rate.mul(100).plot(figsize=(12, 4), title='Churn bulanan (%) | sintetis');
"""), code("""segment = df.assign(near_renewal=df.months_to_renewal.le(1),
                    payment_issue=df.late_payments_3m.gt(0))
display(segment.groupby(['near_renewal', 'payment_issue'])[TARGET].agg(['size', 'mean']))
"""), md("""## 4. Fitur dan pemisahan waktu

Horizon adalah satu bulan kalender. Jul 2025 dan Jan 2026 adalah embargo.
Nasabah yang sama dapat berada di beberapa split pada waktu berbeda: ini meniru scoring
portofolio. Generalisasi ke nasabah yang benar-benar baru memerlukan evaluasi tambahan.
"""), code("""splits = split_months(df)
display(pd.DataFrame([{'split': name, 'rows': len(part),
    'start': part.snapshot_date.min(), 'end': part.snapshot_date.max(),
    'prevalence': part[TARGET].mean()} for name, part in splits.items()]))
assert splits['train'].label_end_date.max() < splits['validation'].snapshot_date.min()
assert splits['validation'].label_end_date.max() < splits['test'].snapshot_date.min()
display(df[['premium_idr', 'income_monthly_idr', 'premium_to_income',
            'months_to_renewal', 'late_payments_3m', 'renewal_payment_interaction']].head())
"""), md("""## 5. Baseline dan model churn

Sel ini menjalankan tiga rolling backtest, fit train, pemilihan AP validation, final test,
permutation importance validation dan bootstrap customer. Model tidak dipilih dari test.
Perhatikan bahwa `average_precision` bukan trapezoidal PR-AUC.
"""), code("""churn = train_churn(df, OUT, SEED)
print('Model terpilih:', churn['selected_model'])
display(churn['metrics'].round(4))
display(pd.read_csv(OUT / 'artifacts/tables/rolling_backtest.csv').round(4))
"""), md("""## 6. Error analysis dan kalibrasi

Top 10% dipilih per bulan. Confusion matrix memakai threshold tetap validation.
Mengapa hasil kedua aturan berbeda? Reliability diagram bersifat diagnostik;
recalibration belum diterapkan. Brier yang kecil saja belum menjamin kalibrasi baik.
"""), code("""from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay
scored = churn['scored']
CalibrationDisplay.from_predictions(scored[TARGET], scored.score, n_bins=8, strategy='quantile');
ConfusionMatrixDisplay(churn['confusion']).plot();
display(churn['importance'].head(10))
display(scored.loc[scored.above_validation_threshold & scored[TARGET].eq(0)].head(3))
display(scored.loc[~scored.above_validation_threshold & scored[TARGET].eq(1)].head(3))
"""), md("""## 7. Akuisisi mingguan: jalur yang berbeda

Dataset akuisisi ini simulasi analitis terpisah dari daftar nasabah.
Lag hanya memakai minggu yang sudah terjadi. Semua wilayah dalam minggu sama masuk
split yang sama. Ini prediksi satu minggu ke depan pada setiap origin, bukan multi-step.
"""), code("""acquisition_raw = simulate_acquisition(SEED)
acquisition_features = engineer_acquisition_features(acquisition_raw)
acquisition = train_acquisition(acquisition_features, OUT)
display(acquisition['metrics'].round(3))
display(acquisition['scored'].loc[acquisition['scored'].unusual_residual].head())
"""), md("""## 8. Buat seluruh visual dan baca dashboard

Periksa cohort retention: sel kosong adalah follow-up belum lengkap. Periksa grafik
sensitivitas: keuntungan bergantung pada asumsi rescue fraction yang belum diukur.
"""), code("""from insurance_lab.visuals import create_figures
monthly, figures = create_figures(df, acquisition_raw, acquisition_features,
                                 churn, acquisition, OUT)
display(Image(filename=str(OUT / 'reports/figures/executive_dashboard.png')))
display(Image(filename=str(OUT / 'reports/figures/cohort_retention.png')))
"""), md("""## 9. Kesimpulan dan keputusan

Isi memo empat paragraf: fakta deskriptif; kualitas prediksi; hal yang belum diketahui;
eksperimen berikutnya. Jangan menyebut importance sebagai sebab atau angka simulasi
sebagai fakta Indonesia. Bandingkan kapasitas kontak 5%, 10%, 20% dengan skenario biaya.
"""), code("""display(pd.read_csv(OUT / 'artifacts/tables/retention_sensitivity.csv'))
display(pd.DataFrame(churn['intervals']))
"""), md("""## 10. Uji inference dari artifact

Contoh ini memakai snapshot historis sintetis, bukan nasabah live. Gunakan joblib
hanya dari sumber tepercaya.
"""), code("""import joblib
bundle = joblib.load(OUT / 'artifacts/models/churn_model.joblib')
example = pd.read_csv(OUT / 'data/processed/scoring_example.csv')
example['churn_probability'] = bundle['pipeline'].predict_proba(example[bundle['features']])[:, 1]
display(example[['customer_id', 'snapshot_date', 'churn_probability']].head(10))
""")]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3 (insurance lab)", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    nbf.write(notebook, out / "notebooks/01_guided_end_to_end.ipynb")
    challenge = nbf.v4.new_notebook(cells=[
        md("""# Student challenge — cari pola dan pertahankan argumen

Gunakan environment dan dataset dari notebook utama. Jangan memakai final test untuk tuning.
Setiap jawaban memerlukan kode, visual dan batas interpretasi.
"""),
        md("""## A. Pola dan bukti (25 poin)
1. Buat lima hipotesis dan cara membantahnya.
2. Bandingkan churn menurut income_band dan bulan; hindari agregasi yang menutupi perubahan mix.
3. Jelaskan sel kosong cohort heatmap dan bias bila semua nasabah awal dianggap cohort baru.
"""), code("# Tulis analisis Anda di sini\n"),
        md("""## B. Modeling dan temporal validation (30 poin)
1. Uji ablation tanpa premium_to_income dan income_band menggunakan validation saja.
2. Bandingkan stabilitas AP dan lift pada tiga rolling fold.
3. Usulkan desain periode kalibrasi tambahan tanpa menyentuh test.
"""), code("# Tulis eksperimen Anda di sini\n"),
        md("""## C. Akuisisi dan keputusan (25 poin)
1. Mengapa korelasi lag campaign bukan ROI kausal?
2. Bandingkan MAE per wilayah dengan MAE total mingguan.
3. Buat rancangan retention experiment dengan unit, outcome, guardrail dan kebutuhan power.
"""), code("# Tulis evaluasi Anda di sini\n"),
        md("""## D. Presentasi (20 poin)
Sampaikan pitch lima menit: problem 1 menit, EDA 1 menit, model 1 menit,
keputusan/eksperimen 1 menit, keterbatasan 1 menit. Sertakan satu temuan yang menggugurkan
hipotesis awal. Jangan mengklaim hasil simulator sebagai temuan bisnis nyata.
""")])
    challenge.metadata = notebook.metadata
    nbf.write(challenge, out / "notebooks/02_student_challenge.ipynb")
