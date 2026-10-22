# Laporan insight — Insurance Behavior Lab

**SELURUH ANGKA ADALAH SINTETIS.** Seed 26; 6,000 nasabah unik,
136,276 snapshot; Januari 2023–September 2026. Satu nasabah = satu polis.
Index ekonomi adalah variabel simulasi, bukan inflasi/BI-Rate aktual.

## Temuan deskriptif

1. Churn tertimbang jumlah polis aktif sebelum kenaikan (Okt–Des 2024): **2.19%**.
   Pada Jan–Feb 2025: **2.39%**. Pada Apr–Jun 2025: **6.71%**.
   Ini perbandingan deskriptif periode dengan populasi yang berubah. Tidak ada kelompok kontrol
   kenaikan harga; analisis ini tidak menyatakan dampak harga tidak signifikan secara statistik.
2. Periksa interaksi renewal dan tunggakan pada `segment_renewal_payment.csv`. Rate memakai
   denominator polis aktif, bukan hanya jumlah pelanggan keluar. Uji apakah pola stabil pada
   cohort dan bulan lain; pelanggan berisiko tinggi yang sudah keluar mengubah komposisi portofolio.
3. Campaign dan penjualan tidak selalu sejalan pada minggu yang sama. Plot lag dan sinyal
   pencarian/referral membantu investigasi. Korelasi lag tetap tidak membuktikan sebab-akibat.

## Hasil model churn

Model dipilih menggunakan **average precision validation**, bukan test: **logistic_regression**.
Training Jan 2023–Jun 2025; validation Agu–Des 2025; test Feb–Sep 2026.
Jul 2025 dan Jan 2026 menjadi embargo. Tiga rolling backtest berada sebelum final validation.
Model final tetap model yang dilatih pada train; tidak dilatih ulang dengan validation.

| Metrik test | Nilai |
|---|---:|
| Prevalensi churn | 3.61% |
| Average precision (AP) | 0.1936 |
| ROC-AUC | 0.7823 |
| Brier score, lebih kecil lebih baik | 0.0319 |
| Precision pada top 10% per bulan | 14.07% |
| Recall pada top 10% per bulan | 39.10% |
| Lift top 10% | 3.90x |

Interval bootstrap nasabah (120 pengulangan) AP:
0.1645–0.2223;
lift: 3.49–4.22x.
Interval ini bersyarat pada model tetap, tidak memasukkan seluruh ketidakpastian akibat shock waktu
atau pemilihan model. Top 10% diurutkan per bulan, bukan menggunakan informasi bulan mendatang.
Confusion matrix memakai threshold tetap 0.0524 yang ditentukan dari validation;
aturan ini berbeda dari kapasitas top 10% yang berubah mengikuti skor setiap bulan.

Sinyal importance validation teratas: **renewal_payment_interaction, late_payments_3m, days_since_contact, premium_to_income, months_to_renewal**.
Permutation importance mengukur perubahan AP ketika fitur diacak, dengan deviasi antar
pengulangan, bukan confidence interval kausal. Fitur berkorelasi dapat saling menggantikan.
Grafik kalibrasi adalah diagnostik; paket ini tidak mengklaim sudah melakukan recalibration.

## Model akuisisi

Model terpilih: **poisson_regression**, berdasarkan MAE total mingguan validation.
MAE total test **10.82 polis/minggu** dan WAPE **7.64%**.
Bandingkan dengan baseline lag-1 pada `acquisition_metrics.csv`. Ini prediksi satu minggu ke depan:
nilai aktual minggu sebelumnya tersedia kembali pada setiap origin. Ini bukan forecast multi-minggu
yang boleh mengakses actual masa depan. Region dari minggu sama selalu berada dalam split sama.
Flag residual menggunakan quantile 2.5%/97.5% residual validation; merupakan aturan investigasi,
bukan interval prediksi terkalibrasi atau detektor yang menjamin false-positive rate.

## Keputusan yang dapat diuji

- Prioritaskan tinjauan manusia terhadap nasabah berisiko: periksa keluhan, pembayaran, lalu renewal.
- Jalankan randomisasi treatment/control di kelompok eligible, dengan outcome retention setelah
  horizon label dan biaya per nasabah. Jangan otomatis memberi diskon kepada semua skor tinggi.
- Untuk akuisisi, tambahkan timestamp exposure, atribusi channel, biaya campaign, pencarian dan referral.
  Uji incrementality dengan holdout wilayah yang memperhitungkan spillover.
- Tabel sensitivitas memakai margin 20% x premi tahunan, biaya kontak Rp15.000, dan asumsi
  rescue fraction 5–30% dari calon churn. Nilai keuntungan bersifat hipotetis dan tidak terukur.
  Jangan memakai skenario ini sebagai estimasi dampak atau rekomendasi penetapan harga.

## Batas studi

Simulator sengaja menanam hubungan dan faktor tersembunyi. Ia mengajarkan investigasi, tidak
membuktikan bahwa mekanisme yang sama terjadi di Indonesia. Tidak ada randomisasi harga;
sebab churn individu belum diketahui. Umur dan kelompok pendapatan adalah fitur latihan,
bukan pembenaran keputusan harga/penolakan layanan. Diagnostik subgroup tersedia, belum menjadi
sertifikasi fairness. Grace period, multi-policy, reinstatement, dan competing risks disederhanakan.

## Artefak yang perlu dibaca

- `figures/executive_dashboard.png`: gambaran manajemen.
- `visual_atlas.pdf`: semua grafik dan diagram.
- `../artifacts/tables/`: metrik, prediksi, cohort, segmentasi dan skenario biaya.
- `../docs/INSTRUCTOR_GUIDE.md`: mekanisme simulasi dan jawaban fasilitator.
- `../docs/REFERENCES.md`: dokumentasi teknis dan sumber konteks ekonomi yang dapat ditambahkan.
