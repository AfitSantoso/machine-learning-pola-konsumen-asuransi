# Verifikasi paket

Dijalankan pada **22 Oktober 2026**, zona waktu Asia/Jakarta.
Runtime: Python 3.14.7, macOS ARM64. Versi library lengkap tercatat pada
`requirements-lock.txt` dan `artifacts/manifest.json`.

## Hasil eksekusi utama

```bash
python build_case.py --customers 6000 --seed 26
```

Berhasil menghasilkan:

- 6000 customer sintetis, 136276 snapshot, 45 bulan: Jan 2023–Sep 2026.
- 975 region-week acquisition mentah dan 955 baris fitur sesudah lag-4 tersedia.
- 44 slide PowerPoint dengan speaker notes dan PDF pendamping 44 halaman.
- 20 gambar/diagram dan atlas visual PDF 20 halaman.
- Pipeline churn dan acquisition tersimpan, metrik, prediksi, insight dan HTML offline.
- Dua notebook: guided end-to-end dan challenge; versi executed notebook disertakan.

Durasi run utama sekitar 11 detik pada lingkungan ini. Waktu laptop lain dapat berbeda;
instalasi dependency dan pembangunan font cache pertama tidak termasuk perkiraan tersebut.

## Pemeriksaan otomatis

```bash
python -m unittest discover -s tests -v
```

**12 pengujian lulus**, meliputi reproducibility seed, penolakan duplicate snapshot,
penolakan fitur masa depan, penolakan snapshot setelah terminal churn, label maturity dan
embargo, lag yang tidak melintasi region atau membaca current outcome, alokasi kapasitas per
bulan, model selection berdasarkan validation, rentang metrik, kesamaan prediksi sesudah
load model, jumlah slide, speaker notes dan batas koordinat elemen PowerPoint.

Pemeriksaan tambahan memverifikasi checksum artifact yang tercatat pada manifest,
jumlah halaman PDF, dan target tautan relatif pada README/dokumentasi/insight.

## Notebook

```bash
python scripts/execute_notebook.py
```

Notebook menjalankan **12 dari 12 sel kode tanpa error**. Metrik model dari notebook sama
dengan metrik CLI pada toleransi numerik `rtol=1e-10`, `atol=1e-12`.
Notebook executed memuat tabel dan visual hasil eksekusi sehingga dapat dibaca tanpa training ulang.

Sandbox lingkungan ini memerlukan izin untuk kernel Jupyter lokal karena memakai port loopback.
Itu kebutuhan eksekusi notebook, bukan kebutuhan koneksi cloud atau pengiriman data.

## Model tersimpan dan inference

```bash
python -m insurance_lab.score --model artifacts/models/churn_model.joblib \
  --input data/processed/scoring_example.csv \
  --output artifacts/tables/scoring_example_predictions.csv

python -m insurance_lab.score --model artifacts/models/acquisition_model.joblib \
  --input data/processed/acquisition_features.csv \
  --output artifacts/tables/acquisition_batch_predictions.csv
```

Berhasil menghasilkan 50 skor churn contoh dan 955 prediksi acquisition historis.
Pengujian integrasi juga membandingkan prediksi pipeline yang di-load dengan prediksi final test.

## Portabilitas direktori dan parameter

Pipeline juga dijalankan dari folder kerja `/private/tmp`, memakai path absolut entry point,
`--customers 1000 --seed 42 --output-dir /private/tmp/insurance_ml_portability --skip-slides`.
Run berhasil. Kode, dokumentasi dan script disalin ke output eksperimen agar notebook dan
referensi sumber tetap tersedia. Perintah tidak memerlukan path `/mnt/data` atau Node.js.

## Angka hasil yang dapat direproduksi

| Hasil test | Nilai |
|---|---:|
| Model churn dari validation | Logistic regression |
| Prevalensi churn test | 3.61% |
| Average precision | 0.1936 |
| ROC-AUC | 0.7823 |
| Precision top 10% per bulan | 14.07% |
| Recall top 10% per bulan | 39.10% |
| Lift top 10% | 3.90x |
| Model akuisisi dari validation | Poisson regression |
| MAE total mingguan | 10.82 polis |
| WAPE total mingguan | 7.64% |

Angka ini adalah hasil simulasi, bukan hasil perusahaan asuransi sesungguhnya.

## Pemeriksaan visual dan batas verifikasi

Preview slide pembuka, learning path, hasil model, serta dashboard dan cohort heatmap telah
ditinjau. Elemen slide diperiksa terhadap batas halaman. Teks/bentuk PowerPoint dapat diedit;
grafik adalah PNG yang dapat dibuat ulang dari kode. PDF dan preview berasal dari definisi
layout yang sama dengan PPTX; belum merupakan hasil render dari aplikasi Microsoft PowerPoint.
Pergantian font di aplikasi pembaca dapat sedikit mengubah tampilan.

Paket ini belum menguji deployment CRM, live data, randomized retention/campaign experiment,
recalibration tambahan, atau integrasi monitoring produksi. Komponen tersebut dibahas sebagai
rancangan pengembangan, bukan fitur produksi yang sudah diaktifkan.
