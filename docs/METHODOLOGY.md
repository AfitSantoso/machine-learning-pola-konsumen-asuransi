# Metodologi analisis dan machine learning

## 1. Pertanyaan dan jenis bukti

EDA menjelaskan apa yang terlihat. Model supervised memperkirakan outcome dari sinyal yang
tersedia. Eksperimen membandingkan outcome dengan counterfactual. Ketiganya memiliki peran
berbeda. Tidak ada model dalam paket ini yang membuktikan dampak kausal kenaikan premi.

Harga naik untuk semua nasabah pada waktu yang sama. Karena tidak ada kelompok yang tidak
mengalami kenaikan, harga sangat berkaitan dengan kalender. Perubahan yang terlihat dapat
berasal dari faktor lain. Grafik yang tidak melonjak juga belum menyatakan efek harga nol.

## 2. Snapshot dan label

Snapshot pada awal bulan T memuat riwayat sampai akhir bulan T-1. Outcome adalah terminal
lapse/cancel selama [T, awal bulan T+1). Durasi bulan kalender bervariasi. Label pada snapshot
1 Juni matang pada 1 Juli. Pemisahan menggunakan embargo satu bulan untuk memberi jarak
antara label terakhir dan snapshot split berikutnya. Dalam produksi, tambahkan grace period,
reporting delay dan waktu finalisasi label yang sebenarnya.

Pada cohort baru, hitungan historis sebelum inception dikeluarkan. Pada pelanggan lama,
histori di awal seri memiliki warm-up pendek. Agregat historis dibuat langsung oleh simulator;
ETL dari raw transaction event adalah pengembangan yang perlu dilakukan untuk data nyata.

## 3. Audit, missing values dan preprocessing

Audit memeriksa kunci unik, fitur mendahului snapshot, label sesudah snapshot, premi positif,
label biner dan tidak ada snapshot setelah terminal churn. Null sengaja ditanam sekitar 1.2%
untuk sessions dan 0.6% untuk region. Realisasi tepatnya ada di `data_audit.json`.

Numerik memakai median imputation dan StandardScaler; kategori memakai most-frequent
imputation dan OneHotEncoder dengan unknown ignored. Semua fit hanya pada data training
di dalam Pipeline, termasuk di setiap rolling fold. Tidak ada SMOTE, oversampling, atau
class weight otomatis. Ini menjaga baseline probabilitas agar tidak sengaja bergeser akibat
penyeimbangan kelas; kualitas probabilitas tetap harus diperiksa lewat calibration dan Brier.

## 4. EDA yang dijalankan

- Tren churn rate dengan denominator polis aktif, jumlah churn, dan jumlah aktif.
- Distribusi dan missingness; segmen renewal x riwayat pembayaran beserta ukuran sampel.
- Tren sinyal historis sebelum label; tidak menyatakan temporal precedence sebagai kausalitas.
- Cohort inception dengan survival product-limit pada tenure, mengecualikan left-truncated
  pelanggan lama. Sel umur yang belum lengkap untuk seluruh kuartal inception disembunyikan.
- Akuisisi mingguan, campaign marker, lead/referral dan korelasi beberapa lag.

Retention pada akhir tenure t dihitung sebagai produk (1 - kejadian churn_t / at_risk_t).
Censoring administratif berbeda dari churn: belum cukup lama diamati bukan berarti bertahan
selamanya atau churn. Pelanggan awal tidak dianggap cohort inception Januari 2023.

## 5. Temporal evaluation

| Split | Periode | Fungsi |
|---|---|---|
| Train | Jan 2023–Jun 2025 | Fit model dan preprocessing |
| Gap | Jul 2025 | Embargo, tidak fit model |
| Validation | Agu–Des 2025 | Model selection dan threshold |
| Gap | Jan 2026 | Embargo |
| Test | Feb–Sep 2026 | Evaluasi terakhir |

Tiga rolling backtest (train end / validation): Jun 2024 / Agu–Okt 2024;
Sep 2024 / Nov 2024–Jan 2025; Des 2024 / Feb–Apr 2025. Masing-masing memiliki satu bulan gap.
Kandidat dan parameter fixed. Pilih model non-dummy berdasarkan AP validation terbesar,
dengan Brier terkecil sebagai tie breaker. Model pemenang tidak di-refit dengan validation,
sehingga threshold tetap sesuai model yang dievaluasi di validation.

Nasabah yang sama bisa muncul di train dan test pada bulan berbeda. Ini realistis untuk
scoring portofolio. Evaluasi untuk customer yang belum pernah diamati memerlukan group holdout
atau inception cohort terpisah. Periode sama tidak pernah dibagi acak antar customer/region.

## 6. Model churn

- DummyClassifier prior: baseline probabilitas konstan dari train.
- LogisticRegression: C=1, max_iter=600; termasuk fitur interaksi eksplisit.
- HistGradientBoostingClassifier: 130 iterasi, learning_rate=0.06, max_leaf_nodes=15,
  min_samples_leaf=55, l2_regularization=8, early_stopping=False.

HGB dipilih sebagai kandidat nonlinear efisien CPU. Logistik adalah baseline interpretabel.
Tidak ada keharusan model yang paling kompleks menang. Parameter dipatok untuk praktikum,
bukan hasil tuning test. Notebook challenge menyarankan ablation pada validation saja.

## 7. Metrik dan kapasitas

AP (`average_precision_score`) merupakan jumlah precision dibobot perubahan recall.
Ini berbeda dari area trapezoid di bawah PR curve. ROC-AUC mengukur ranking pasangan
positif-negatif. Brier adalah mean squared error dari probabilitas terhadap outcome biner.
Ketiganya perlu dibaca bersama prevalensi dan konteks biaya.

Untuk setiap bulan, urutkan skor lalu pilih ceil(10% x nasabah aktif), tie break customer_id.
Precision@10% = churn yang terpilih / seluruh yang terpilih.
Recall@10% = churn yang terpilih / seluruh churn.
Lift = precision@10% / prevalensi test. Akumulasi dilakukan sesudah pemilihan bulanan.
Perbedaan prevalensi antarbulan dapat memengaruhi lift agregat; lihat juga metrik per bulan.

Confusion matrix memakai threshold tetap berupa quantile 90% skor validation. Threshold tetap
ini berbeda dari kapasitas top10 per bulan, sehingga kedua aturan bisa memilih nasabah berbeda.
Prediksi positif dari threshold tidak berarti 10% nasabah setiap bulan.

Calibration diagram memakai delapan quantile bins pada test. Ini diagnostik, bukan proses
recalibration. Jika diperlukan, sediakan periode kalibrasi development terpisah atau temporal
cross-validation dan uji pada holdout baru; jangan fit calibrator pada final test.

Bootstrap resampling customer (120 kali) mempertahankan ketergantungan antar snapshot customer.
AP dihitung dengan bobot multiplicity customer; lift memakai ranking top10 model tetap.
Interval tidak mencakup seluruh ketidakpastian seleksi model atau shock waktu bersama.

## 8. Explainability

Permutation importance dihitung pada maksimum 4500 baris validation, empat pengulangan,
scoring AP. Penurunan AP mengukur ketergantungan model pada fitur dalam sampel tersebut.
Standar deviasi permutation bukan confidence interval kausal. Fitur berkorelasi dapat
menggantikan satu sama lain; interaksi yang diacak dapat menghasilkan kombinasi kurang realistis.

## 9. Akuisisi: forecast satu langkah dan residual

Data mingguan dibagi sebelum Jul 2025 untuk train, Agu–Des 2025 untuk validation dan
Feb–Sep 2026 untuk test, dengan gap yang sama secara kalender. Seluruh region pada minggu
yang sama masuk split yang sama. Lag dibangun groupby region lalu shift.

Bandingkan prediksi lag-1 dengan PoissonRegressor alpha=0.2, max_iter=500. Fitur meliputi
lag1/lag4 actual, lead/referral/search lag1, campaign lag2, campaign terencana dan musim.
Kolom activity minggu yang sama tidak menjadi fitur model. Pilih dari MAE total mingguan
validation. Metadata selected_model menjelaskan aturan yang terpilih. Artifact dan CLI inference memakai
aturan baseline lag1 bila baseline menang, atau pipeline Poisson bila Poisson menang.

Pada setiap origin test, actual minggu sebelumnya sudah tersedia. Ini rolling one-step
backtest, bukan satu forecast multi-step yang dibuat pada awal periode test. Untuk multi-step,
actual masa depan tidak boleh dipakai sebagai lag input.

MAE per region-week dan MAE total per minggu mengukur masalah berbeda. WAPE = jumlah error
absolut total mingguan / jumlah actual total mingguan. Flag residual memakai quantile
2.5% dan 97.5% dari validation; bukan interval prediksi terkalibrasi dan tidak menjamin 5% false alarm.

## 10. Decision economics dan eksperimen

Skenario: nilai margin terselamatkan = premi bulanan x 12 x 20%; biaya kontak Rp15.000;
rescue_fraction 5%, 10%, 20%, 30% dari pelanggan yang akan churn tanpa intervensi.
E[net] = P(churn) x rescue_fraction x margin - biaya kontak.
Formula menyederhanakan discount cost, competing outcomes, negative treatment effects dan CLV.
Tabel ini analisis sensitivitas hipotetis, bukan nilai keuntungan yang telah dicapai.

Randomisasikan treatment/control dalam kelompok eligible, tentukan horizon dan primary metric,
lakukan analisis intention-to-treat, lalu ukur incremental retention dan net margin.
Power calculation memerlukan baseline rate, minimum detectable effect, unit/cluster dan alpha.
Jangan mengklaim observasi retention sesudah outreach sebagai efek outreach tanpa kontrol.

## 11. Engineering dan cakupan implementasi

Implemented: simulator, audit, fitur, temporal train/validation/test, rolling backtest,
dua jalur model, artifact, inference churn, grafik, laporan offline, notebook dan deck.
Design only: integrasi CRM, live feature store, eksperimen langsung, monitoring terjadwal,
automatic retrain dan produksi. Semua file generated berada di output-dir yang eksplisit.
Manifest merekam versi dan checksum. Tidak ada pengiriman pesan, unggahan atau deployment.
