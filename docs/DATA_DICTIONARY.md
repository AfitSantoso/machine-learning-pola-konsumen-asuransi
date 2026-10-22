# Data dictionary dan kontrak waktu

Semua file `*_synthetic.csv` adalah data buatan. Setiap customer_id hanya memiliki satu polis.
Dua simulator (portofolio dan akuisisi) adalah dua latihan terpisah; jumlah polis baru mingguan
tidak direkonsiliasi dengan entry pada portofolio.

## Tabel polis

Sumber: `data/raw/policy_monthly_synthetic.csv`.
Hasil feature engineering: `data/processed/policy_features.csv`.

| Kolom | Tipe / satuan | Arti dan ketersediaan |
|---|---|---|
| customer_id | integer | ID sintetis; bukan fitur model |
| snapshot_date | tanggal | Awal bulan, polis masih aktif |
| feature_as_of | tanggal | Batas ketersediaan fitur, sehari sebelum snapshot |
| label_end_date | tanggal | Awal bulan berikutnya; outcome lengkap setelah interval selesai |
| entry_date | tanggal | Awal observasi bagi pelanggan lama, inception bagi pelanggan baru |
| is_inception_cohort | boolean | True hanya untuk cohort baru yang inception-nya teramati |
| age | tahun | Umur awal simulasi; dibuat tetap untuk menyederhanakan latihan |
| region | kategori | Jabodetabek, Jawa_lain, Sumatera, Kalimantan, Indonesia_timur |
| product | kategori | Kesehatan, Jiwa, Kendaraan |
| income_band | kategori | Low, Mid, High; tidak menunjukkan klasifikasi resmi |
| tenure_months | bulan | Usia polis pada snapshot |
| premium_idr | rupiah/bulan | Premi kontraktual saat snapshot; termasuk jadwal +10% |
| income_monthly_idr | rupiah/bulan | Pendapatan sintetis, bukan variabel yang selalu tersedia di dunia nyata |
| premium_change_pct | proporsi | 0 sebelum Jan 2025, 0.10 sesudahnya |
| economic_stress_index | indeks buatan | Konteks yang diasumsikan sudah diketahui saat snapshot; bukan IHK/BI-Rate |
| months_to_renewal | 0–11 bulan | 0 = bulan renewal; <=1 = window renewal |
| late_payments_3m | hitungan 0–3 | Kejadian keterlambatan dari maksimum 3 bulan historis |
| complaints_3m | hitungan 0–3 | Kejadian keluhan historis |
| claims_3m | hitungan 0–3 | Kejadian klaim historis |
| days_since_contact | hari | Recency kontak CRM (termasuk prospect), dibatasi 365 |
| app_sessions_1m | hitungan | Aktivitas aplikasi historis satu bulan; ada missingness |
| premium_to_income | rasio | premium_idr / income_monthly_idr; fitur turunan |
| renewal_payment_interaction | hitungan | 1(months_to_renewal<=1) x late_payments_3m |
| churn_next_month | 0/1 | Terminal cancel/lapse dalam [snapshot_date, label_end_date) |

Satu pelanggan dapat muncul berkali-kali. Setelah churn pertama ia tidak muncul lagi.
Model tidak memakai ID, entry_date, is_inception_cohort, tanggal outcome atau label sebagai fitur.
Premi adalah jadwal tagihan yang diketahui pada snapshot, sedangkan perilaku hanya dari masa lalu.
Riwayat sebelum inception dikeluarkan dari hitungan; pada awal keseluruhan simulasi belum
ada histori perilaku 3 bulan yang lengkap (warm-up). Pertimbangkan mengeluarkan awal seri
sebagai challenge robustness. Nasabah awal memiliki tenure yang beragam dan left truncation.

## Tabel akuisisi

Sumber: `data/raw/acquisition_weekly_synthetic.csv`, satu region x week.
Hasil fitur: `data/processed/acquisition_features.csv`, empat minggu awal setiap region dihapus
karena lag-4 belum tersedia.

| Kolom | Arti |
|---|---|
| week | Senin awal minggu prediksi |
| region | Wilayah sintetis |
| campaign_planned | Jadwal campaign minggu ini, diasumsikan diketahui sebelum minggu dimulai |
| inbound_leads | Lead pada minggu ini; tidak dipakai langsung untuk memprediksi outcome minggu sama |
| search_index | Indeks minat buatan minggu ini, bukan data Google Trends aktual |
| referrals | Referral masuk pada minggu ini |
| new_policies | Target: jumlah polis baru pada minggu ini |
| new_policies_lag1 / lag4 | Actual polis baru 1 / 4 minggu sebelumnya |
| leads_lag1 | Lead minggu sebelumnya |
| search_index_lag1 | Indeks pencarian minggu sebelumnya |
| referrals_lag1 | Referral minggu sebelumnya |
| campaign_lag2 | Jadwal campaign dua minggu sebelumnya |
| season_sin / season_cos | Encoding siklis minggu kalender yang diketahui sebelum outcome |

Outcome minggu ini menggunakan demand signals sebelumnya. `new_policies / inbound_leads`
bukan conversion rate cohort yang sah karena lead dan outcome tidak selalu berasal dari cohort
masuk yang sama. Simulasi tidak menyediakan unit lead-individu untuk conversion modeling.

## Feature contract dan batas produksi

`insurance_lab/config.py` adalah sumber daftar fitur. Imputer, scaler dan encoder fit pada train.
Untuk data nyata, setiap fitur juga membutuhkan event_time, ingestion_time, source_version,
serta aturan keterlambatan pelaporan. Data makro harus di-join berdasarkan publikasi yang sudah
tersedia saat snapshot, bukan berdasarkan kalender referensi atau nilai revisi terakhir.

## Output evaluasi

| File | Kegunaan |
|---|---|
| churn_metrics.csv | Baseline/logistik/HGB pada validation dan test |
| rolling_backtest.csv | Tiga fold historis per model |
| churn_metrics_by_month.csv | Diagnostik temporal model terpilih |
| subgroup_diagnostics.csv | Diagnostik region dan income, belum audit fairness menyeluruh |
| churn_test_predictions.csv | Skor, label, top10 bulanan, threshold tetap, skenario biaya dan review |
| permutation_importance_validation.csv | Rata-rata penurunan AP dan deviasi 4 permutation |
| cohort_retention.csv | Retention product-limit inception cohort, censoring ditandai kosong |
| acquisition_metrics.csv | MAE region-week, MAE total mingguan, WAPE dan deviance |
| acquisition_test_predictions.csv | Prediksi, residual dan flag investigasi |
| retention_sensitivity.csv | Skenario kapasitas dan efek retensi yang diasumsikan |
