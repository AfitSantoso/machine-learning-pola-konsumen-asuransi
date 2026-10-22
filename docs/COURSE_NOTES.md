# Materi praktikum — dari pertanyaan bisnis ke model

Gunakan bersama slide dan notebook. Semua angka latihan berasal dari simulator.
Baca `METHODOLOGY.md` untuk batas evaluasi dan `DATA_DICTIONARY.md` untuk kontrak kolom.

## 1. Mengubah kebingungan bisnis menjadi pertanyaan analitis

"Mengapa nasabah keluar?" terlalu luas untuk langsung dijadikan model. Pecah menjadi:

1. Apa yang terjadi: kapan rate berubah, di wilayah/produk/cohort apa?
2. Apa yang dapat diprediksi: siapa akan lapse dalam satu bulan kalender berikutnya?
3. Apa yang belum diketahui: apakah harga, layanan, kondisi ekonomi atau proses administrasi?
4. Apa yang dapat dilakukan: tindakan mana meningkatkan retention dibanding layanan rutin?

Business brief harus memuat pemilik keputusan, kapasitas kontak, unit analisis, horizon,
biaya kesalahan dan metrik sukses. Prediksi churn yang akurat tetapi tidak mengubah keputusan
atau tidak menghasilkan manfaat incremental belum menyelesaikan masalah bisnis.

Contoh brief: "Tim layanan memiliki kapasitas menghubungi 10% polis aktif tiap bulan.
Buat ranking risiko lapse bulan berikutnya; evaluasi berapa churn yang ditemukan dan rancang
uji layanan agar diketahui apakah kontak benar-benar membantu nasabah bertahan."

## 2. Memahami pandas dan audit dasar

```python
from insurance_lab.data import simulate_policies, engineer_policy_features, audit_policy_data

raw = simulate_policies(n_customers=6000, seed=26)
df = engineer_policy_features(raw)
audit = audit_policy_data(df)
print(df.shape)
print(df.dtypes)
print(df.isna().mean().sort_values(ascending=False).head())
```

`DataFrame` adalah tabel. `shape` memberi jumlah baris/kolom; `dtypes` menunjukkan tipe.
`isna().mean()` adalah proporsi kosong. Audit menghentikan pipeline bila kunci duplikat,
timestamp salah, label bukan biner, atau customer masih muncul setelah terminal churn.

Jangan menghapus missing values secara otomatis. Hilangnya region mungkin menunjukkan
masalah channel, sedangkan tidak tercatatnya sesi aplikasi dapat memiliki arti bisnis lain.
Dalam simulator missingness dibuat acak; dalam data nyata mekanismenya perlu ditelusuri.

## 3. EDA yang menghubungkan visual dengan keputusan

### Rate dan denominator

```python
from insurance_lab.config import TARGET
monthly = df.groupby('snapshot_date').agg(
    active=(TARGET, 'size'), churned=(TARGET, 'sum'), rate=(TARGET, 'mean'))
monthly['rate_pct'] = monthly.rate * 100
monthly.rate_pct.plot(title='Churn (%) | data sintetis')
```

Jika 100 orang keluar dari 10000 nasabah, rate 1%. Jika bulan berikutnya hanya 80 keluar
dari 4000 aktif, jumlah churn turun tetapi rate naik menjadi 2%. Tanpa denominator,
presentasi dapat memberi arah keputusan yang salah.

### Segmentasi dan interaksi

```python
segments = df.assign(near_renewal=df.months_to_renewal.le(1),
                     payment_issue=df.late_payments_3m.gt(0))
summary = segments.groupby(['near_renewal', 'payment_issue'])[TARGET].agg(['size', 'mean'])
```

Interaksi berarti hubungan suatu fitur bergantung pada fitur lain. Tunggakan dapat menjadi
lebih penting dekat renewal. Periksa ukuran segmen serta kestabilannya antar waktu.
Selisih rate antarkelompok belum menjadi efek kausal karena kelompok berbeda dalam faktor lain.

### Cohort dan censoring

Cohort inception mengelompokkan nasabah menurut waktu mulai polis. Bandingkan mereka pada
umur polis yang sama. Nasabah yang baru masuk enam bulan tidak bisa dianggap gagal memiliki
retention 12 bulan; outcome 12 bulannya belum terlihat. Ini right censoring.

Pelanggan yang sudah bertahan sebelum awal data memiliki left truncation. Memasukkan mereka
sebagai cohort inception awal akan membuat retention tampak lebih baik karena pelanggan yang
sudah churn sebelum data tersedia tidak teramati. Paket hanya memakai cohort baru untuk heatmap.

### Lag

`campaign.shift(2)` menggeser campaign dua minggu ke depan agar campaign lama dibandingkan
penjualan sekarang. Korelasi lag membantu memilih data yang perlu diinvestigasi, tetapi jadwal
periodik, musim dan demand bersama dapat menghasilkan korelasi tanpa efek kausal.

## 4. Feature engineering dan leakage

Fitur harus mencerminkan informasi yang tersedia ketika keputusan dibuat.

```python
df['premium_to_income'] = df.premium_idr / df.income_monthly_idr
df['renewal_payment_interaction'] = df.months_to_renewal.le(1) * df.late_payments_3m
```

Rasio premi terhadap pendapatan memberi konteks keterjangkauan. Interaksi memberi model
logistik akses ke pola gabungan. Dalam data nyata, estimasi pendapatan harus diperiksa sumber,
izin penggunaan dan keandalannya. Uji ablation untuk mengetahui apakah fitur tersebut dibutuhkan.

Contoh leakage: memasukkan alasan cancel ke model yang memprediksi cancel; menghitung
`late_payments_3m` menggunakan transaksi setelah snapshot; memasukkan claim resolution yang
baru selesai akhir bulan; fit scaler pada semua data sebelum split.

Simulator menghasilkan agregat historis langsung. Untuk sistem nyata, bangun fitur dari event
log dengan event_time dan ingestion_time, lalu lakukan point-in-time join pada setiap snapshot.

## 5. Pipeline preprocessing

```python
from insurance_lab.modeling import candidate_models
from insurance_lab.config import FEATURES, TARGET, split_months

parts = split_months(df)
model = candidate_models(seed=26)['logistic_regression']
model.fit(parts['train'][FEATURES], parts['train'][TARGET])
probability = model.predict_proba(parts['validation'][FEATURES])[:, 1]
```

Model menerima fitur numerik dan kategori. Median imputer mengisi numerik, scaler mengubah
skala berdasarkan mean/std train, dan one-hot encoder membuat indikator kategori. Kategori baru
pada inference tidak menghentikan proses, tetapi perlu ditinjau sebagai indikasi perubahan data.

Pipeline menyimpan semua transformasi dengan estimator. Langkah ini mencegah transformasi
training berbeda dari transformasi inference dan menjaga imputasi tetap fit pada train.

## 6. Tiga kandidat churn

### Baseline prevalensi

Prediksi konstan `p = jumlah churn train / jumlah snapshot train`. Tidak memiliki kemampuan
ranking di luar ikatan skor. Baseline diperlukan agar model dinilai dari nilai tambahnya.

### Logistic regression

`z = beta0 + beta1*x1 + ... + betap*xp`, lalu `p = 1 / (1 + exp(-z))`.
Koefisien bekerja pada log-odds, bukan perubahan probability yang konstan. Probability bergantung
pada seluruh fitur. Koefisien prediktif juga bukan estimasi dampak intervensi tanpa asumsi kausal.

Kelebihan: efisien, stabil dan mudah dijadikan baseline. Batas: pola nonlinear dan interaksi
perlu dirancang atau ditangani model lain. Regularisasi membantu mengendalikan kompleksitas.

### Histogram gradient boosting

Model menambahkan pohon kecil bertahap untuk mengurangi kesalahan prediksi. Binning fitur
meningkatkan efisiensi. Kedalaman/leaf, minimum samples dan regularisasi menjaga agar model
menggeneralisasi. Model ini dapat mempelajari nonlinearitas tetapi tetap bisa kalah dari logistik.

Tidak ada alasan memakai GPU atau deep learning hanya karena proyek melibatkan machine learning.
Pilih kompleksitas berdasarkan struktur data, kinerja validasi dan kebutuhan operasional.

## 7. Evaluasi yang dapat dipertanggungjawabkan

Train dipakai belajar parameter. Validation memilih kandidat dan threshold. Test dipakai
sekali untuk mengukur hasil final setelah pilihan selesai. Rolling backtest menambah gambaran
kestabilan di beberapa periode historis. Pada panel, pisahkan berdasarkan tanggal, bukan baris.

- **Precision:** dari semua yang dipilih, berapa yang benar-benar churn?
- **Recall:** dari semua churn, berapa yang ditemukan?
- **Lift:** precision kelompok terpilih dibanding prevalensi populasi evaluasi.
- **AP:** ringkasan precision-recall dengan pembobot perubahan recall; makin besar makin baik.
- **ROC-AUC:** peluang skor churn lebih tinggi daripada skor non-churn pada pasangan acak.
- **Brier:** rata-rata `(p-y)^2`; makin kecil makin baik, tetapi perlu konteks baseline.

Contoh: dari 1000 aktif, 30 churn. Jika 100 dipilih dan 12 churn, precision 12%, recall 40%,
lift 4x dibanding prevalensi 3%. Ini belum berarti 12 nasabah dapat diselamatkan oleh kontak.

Calibration menilai apakah prediksi 10% benar-benar mengalami churn sekitar 10%. Reliability
diagram mengelompokkan prediksi lalu membandingkan rata-rata skor dengan rate aktual.
Grafik paket bersifat diagnostik. Calibrator tambahan harus dipelajari tanpa memakai final test.

## 8. Interpretasi dan ketidakpastian

Permutation importance menanyakan seberapa besar kinerja turun bila informasi suatu fitur diacak.
Ia menjelaskan ketergantungan model terhadap sinyal, bukan pengaruh mengubah variabel pada dunia nyata.
Fitur berkorelasi dan fitur interaksi memerlukan interpretasi hati-hati.

Customer bootstrap melakukan sampling ulang ID, mempertahankan snapshot yang berulang pada ID itu.
Ia lebih sesuai daripada menganggap semua baris independen, tetapi belum menangkap shock waktu
bersama antar nasabah. Jangan menampilkan interval sebagai kepastian yang menghapus semua risiko.

Error analysis membantu membuat backlog data: apakah false negative dipengaruhi faktor yang
belum dicatat? Apakah false positive sebenarnya berisiko tetapi bertahan karena keputusan lain?
Tidak semua prediksi salah dapat dijelaskan hanya dari fitur yang tersedia.

## 9. Model akuisisi dan prediksi hitungan

Poisson regression memakai `mu = exp(beta0 + beta*x)` untuk memprediksi mean hitungan positif.
Dalam model ini, seluruh fitur behavioral memakai lag historis. Campaign terencana boleh menjadi
fitur karena jadwalnya diasumsikan sudah diketahui sebelum minggu berjalan.

Bandingkan dengan `prediction_t = actual_(t-1)`. Evaluasi MAE region-week untuk ketepatan lokal,
dan MAE total mingguan untuk kapasitas nasional. WAPE adalah total absolute error dibagi total
actual. Pilih model sesuai level keputusan yang disepakati, bukan metrik yang kebetulan terbaik.

Forecast satu langkah berarti setiap minggu kita menerima actual minggu lalu dan memperbarui
prediksi minggu berikutnya. Jika membuat forecast delapan minggu sekali jalan, actual minggu
kedua sampai kedelapan belum tersedia; lag perlu diprediksi atau model harus dirancang berbeda.

Residual besar memicu investigasi tracking, event, channel, lead quality atau competitor.
Threshold residual dari validation hanya aturan triage, bukan penjelasan penyebab otomatis.

## 10. Menentukan tindakan dan mengukur manfaat

Skor risiko menjawab siapa mungkin churn tanpa mengetahui apakah ia akan merespons treatment.
Proporsi yang dapat diselamatkan harus diukur menggunakan eksperimen atau desain kausal memadai.

Contoh skenario: p(churn)=0.20, rescue_fraction=0.20, margin tahunan yang terselamatkan
Rp1.200.000, biaya kontak Rp15.000. Nilai harapan = 0.20 x 0.20 x 1.200.000 - 15.000 = Rp33.000.
Ini hanya perhitungan berdasarkan asumsi. Bila rescue_fraction lebih kecil atau probabilitas
tidak terkalibrasi, keputusan dapat berubah. Diskon dan biaya tambahan harus dimasukkan bila ada.

Experiment brief harus menetapkan eligibility, unit randomisasi, control, treatment, outcome,
horizon, guardrail, minimum detectable effect, rencana ukuran sampel dan analisis intention-to-treat.
Retention observed setelah campaign tidak cukup: nasabah tersebut mungkin bertahan tanpa campaign.

## 11. Menyampaikan hasil

Memo untuk manajemen berisi: masalah dan konteks, tiga temuan yang didukung data, kualitas prediksi,
asumsi/keterbatasan, serta satu keputusan atau eksperimen berikutnya. Tampilkan denominator,
periode, horizon dan keterangan sintetis pada visual. Hindari kalimat "AI menemukan penyebab"
ketika analisis hanya mengukur korelasi atau ranking risiko.

Latihan akhir: buka dashboard, pilih satu keputusan yang mungkin diambil, lalu jelaskan bukti
apa yang masih diperlukan sebelum tindakan dijalankan secara luas.
