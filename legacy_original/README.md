# Insurance Consumer Behavior — ML Professional Workshop

## Isi paket

- `Insurance_Consumer_Behavior_ML_Masterclass.pptx` — materi 30 slide, Bahasa Indonesia
- `build_case.py` — script lengkap membuat data sintetis, EDA, model logistik dan random forest, dan grafik
- `insurance_policy_monthly_synthetic.csv` — data longitudinal **sintetis**
- `insurance_acquisition_monthly_synthetic.csv` — data agregat akuisisi **sintetis**
- `model_metrics.csv`, `model_feature_importance.csv` — contoh hasil
- `*.png` — lima grafik praktikum

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn
python build_case.py
```

Additional advanced libraries (optional): `pip install seaborn shap statsmodels lifelines ruptures xgboost imbalanced-learn mlflow evidently`.

## Catatan penting metodologi

1. Semua data dan anomali disimulasikan untuk belajar; angka hasil model BUKAN hasil bisnis nyata.
2. `churn_next_30d` adalah indikator kejadian churn pada interval setelah snapshot awal bulan tersebut (month-ahead label); kolom fitur dimaksudkan tersedia di awal periode. Dalam data nyata, pastikan timestamp setiap fitur benar-benar sebelum scoring.
3. Pembagian waktu: training Jan 2025–Jan 2026; test Feb–Okt 2026. Untuk skenario 30 hari, gunakan embargo/gap serta finalisasi label sesuai ketentuan data produksi. Baris pelanggan dapat berulang pada bulan berlainan.
4. Simulator memasukkan latent unobserved risk dan kejadian demand yang **tidak tersedia** sebagai fitur. Peserta tidak boleh mengklaim mengetahui penyebab lonjakan dari dataset observasional ini.
5. Accuracy dan ROC-AUC saja tidak cukup untuk churn langka; periksa PR-AUC, lift@K, calibration, business cost dan distribusi menurut cohort.
6. Untuk mengukur dampak promosi/kampanye, gunakan control/holdout yang memadai. Feature importance, correlation, dan model predictions bukan bukti kausal.
7. Grafik importance adalah impurity-based RF importance: berpotensi bias pada feature ber-cardinality tinggi atau kontinu. Untuk analisis lebih mendalam, gunakan permutation importance pada data holdout atau SHAP.
8. Data akuisisi hanya 22 titik bulanan; tidak cukup untuk analisis kausal ataupun demand forecasting yang kokoh. Kumpulkan frekuensi mingguan/harian dengan attribution channel jika ingin implementasi produksi.

## Challenge praktikum

**Business:** tulis tiga hipotesis untuk no-immediate-churn, delayed-churn, dan acquisition-spike. Tunjukkan informasi tambahan yang diperlukan untuk menguji setiap hipotesis.

**EDA:** buat chart churn by `renewal_window`, `late_payment_last_month`, `income_band`, plus segmented churn time trends.

**Model:** reproduksi baseline dan RandomForest, lakukan rolling-origin validation dan calibration. Bandingkan top 5%, 10%, 20% risiko dengan biaya outreach dan opportunity value.

**Decision:** buat policy untuk high risk/high value, dan desain randomized experiment untuk incremental retention. Presentasikan caveat data agar manajemen tidak menganggap korelasi sebagai penyebab.

## Menggunakan pada kelas

Jalankan `build_case.py`, diskusikan 3 visual anomali, tugaskan kelompok menyusun hipotesis dan menyelesaikan analisis. Jangan tunjukkan formula generasi sintetis sebelum peserta merumuskan hipotesis karena formula berisi mekanisme simulasi yang dijadikan misteri pada case study.
