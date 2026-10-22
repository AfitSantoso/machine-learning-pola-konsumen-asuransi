# Insurance Behavior Lab — EDA sampai Machine Learning

Paket praktikum profesional berbahasa Indonesia untuk menyelidiki tiga pertanyaan:
premi bulanan naik 10% tetapi churn tidak langsung meningkat, churn melonjak beberapa waktu
kemudian, serta campaign sepi sementara pendaftaran ramai pada waktu lain.

**Semua data dan hasil adalah sintetis.** Paket ini mengajarkan investigasi, prediksi dan
pengujian keputusan; tidak mengklaim penyebab perilaku nasabah Indonesia atau dampak nyata harga.

## Mulai dari hasil yang sudah tersedia

- [PowerPoint lengkap dan editable](slides/Insurance_Consumer_Behavior_ML_Masterclass.pptx), dengan speaker notes pengajar.
- [PDF materi](slides/Insurance_Behavior_Lab.pdf), untuk dibaca tanpa PowerPoint.
- [Learning path 16 jam](docs/LEARNING_PATH.md) dan [diagram path](reports/figures/learning_path.png).
- [Materi praktikum terperinci](docs/COURSE_NOTES.md), dengan konsep, rumus dan contoh kode.
- [Notebook end-to-end](notebooks/01_guided_end_to_end.ipynb) dan [student challenge](notebooks/02_student_challenge.ipynb).
- [Laporan visual offline](reports/dashboard.html), [insight bisnis](reports/INSIGHTS.md), dan [atlas grafik PDF](reports/visual_atlas.pdf).
- [Metodologi](docs/METHODOLOGY.md), [data dictionary](docs/DATA_DICTIONARY.md), serta [panduan pengajar](docs/INSTRUCTOR_GUIDE.md).

![Learning path](reports/figures/learning_path.png)

## Instalasi

Python 3.11+ disarankan. Lingkungan yang benar-benar dipakai untuk verifikasi tercatat pada
`artifacts/manifest.json`; dependency snapshot tersedia di `requirements-lock.txt`.
CPU laptop dan RAM sekitar 8 GB cukup sebagai sasaran praktikum; tidak membutuhkan GPU.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_case.py
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python build_case.py
```

Untuk mengulang environment verifikasi pada versi Python/platform yang sesuai:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` memberi rentang lintas platform; `requirements-lock.txt` mencatat versi
lingkungan pengujian. File lock bukan jaminan kompatibilitas semua versi Python/platform.

## Menjalankan pipeline

```bash
# Semua tahap: data, EDA, model, grafik, insight, dashboard, notebook, PowerPoint dan PDF
python build_case.py --customers 6000 --seed 26

# Eksperimen terpisah agar hasil utama mudah dibandingkan
python build_case.py --customers 2000 --seed 42 --output-dir artifacts/experiment_42

# Hanya data, analisis, model, visual dan notebook; lewati pembuatan deck
python build_case.py --skip-slides

# Pengujian kontrak data, leakage, lag, split dan artifact
python -m unittest discover -s tests -v
```

Path mengikuti lokasi script, bukan `/mnt/data` atau folder kerja shell. `--output-dir`
menentukan lokasi hasil. Rerun menulis ulang artifact dengan nama sama di output-dir tersebut.
Dokumentasi dan kode sumber juga disalin ke output-dir eksperimen agar notebook dapat digunakan
di sana. Tidak ada pengunduhan data saat pipeline berjalan.
`--skip-slides` tidak memperbarui deck yang sudah ada; gunakan output-dir terpisah untuk eksperimen.

## Batch scoring dari model tersimpan

Churn, memakai contoh snapshot historis sintetis:

```bash
python -m insurance_lab.score \
  --model artifacts/models/churn_model.joblib \
  --input data/processed/scoring_example.csv \
  --output artifacts/tables/scoring_example_predictions.csv
```

Akuisisi, memakai fitur historis yang sudah dibangun:

```bash
python -m insurance_lab.score \
  --model artifacts/models/acquisition_model.joblib \
  --input data/processed/acquisition_features.csv \
  --output artifacts/tables/acquisition_batch_predictions.csv
```

Perintah akuisisi tersebut hanya demonstrasi inference pada file historis, bukan evaluasi baru.
CLI memeriksa kolom fitur dan memakai model yang terpilih dari validation, termasuk aturan lag-1
bila baseline menang. Artifact joblib hanya boleh dimuat dari sumber tepercaya.

## Menggunakan notebook

Buka `notebooks/01_guided_end_to_end.ipynb` di VS Code/Jupyter dan pilih kernel dari `.venv`.
Jalankan sel berurutan. Output notebook disimpan ke `artifacts/notebook_run/` agar hasil utama
mudah dibandingkan. Library opsional JupyterLab dapat dipasang bila ingin UI browser:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

Notebook challenge berisi soal tanpa solusi implementasi. Pengajar membaca panduan pengajar
setelah peserta menyusun hipotesis. Formula simulator sengaja transparan untuk debrief.

Versi notebook yang telah dijalankan tersedia sebagai
`notebooks/01_guided_end_to_end.executed.ipynb`. Untuk menjalankan ulang otomatis:

```bash
python scripts/execute_notebook.py
```

Eksekusi otomatis membutuhkan izin menjalankan kernel Jupyter lokal (port loopback).

## Struktur proyek

```text
build_case.py                 # Entry point tunggal
insurance_lab/
  config.py                   # Kontrak fitur dan temporal split
  data.py                     # Simulator, fitur dan audit
  modeling.py                 # Baseline, model, backtest dan evaluasi
  visuals.py                  # EDA, diagram dan diagnostic charts
  reporting.py                # Insight dan HTML offline dari hasil nyata run
  presentation.py             # PPTX editable, PDF materi dan preview
  notebooks.py                # Generator notebook praktikum
  score.py                    # Batch inference churn dan akuisisi
  run.py                      # Orkestrasi dan manifest
notebooks/                    # Notebook guided dan challenge
slides/                       # PPTX, PDF dan beberapa preview
reports/                      # Insight, dashboard, atlas dan figures
artifacts/models/             # Pipeline joblib dengan metadata
artifacts/tables/             # Prediksi, metrik, cohort, segmen dan sensitivitas
artifacts/manifest.json        # Versi runtime, seed dan checksum hasil
artifacts/data_audit.json      # Hasil audit data
artifacts/run_summary.json     # Angka untuk laporan dan slide
data/raw/                     # Dataset sintetis sumber
data/processed/               # Fitur dan input scoring
scripts/                      # Eksekusi dan validasi notebook
tests/                        # Pemeriksaan data dan artifact
docs/                         # Materi, metode dan referensi
legacy_original/              # Salinan versi awal sebelum perbaikan
```

## Metode yang sudah diimplementasikan

- Audit data; EDA timeline, cohort inception, segmentasi dan sinyal historis.
- Churn: baseline prevalensi, logistic regression, histogram gradient boosting.
- Train Jan 2023–Jun 2025; validation Agu–Des 2025; test Feb–Sep 2026, dengan embargo.
- Tiga rolling backtest, pemilihan model dari AP validation, final test terpisah.
- Average precision, ROC-AUC, Brier, calibration diagnostic, confusion matrix dan lift top10 bulanan.
- Permutation importance validation; interval ilustratif bootstrap customer; subgroup diagnostics.
- Akuisisi mingguan: baseline lag-1, Poisson regression, MAE/WAPE dan residual triage.
- Skenario biaya-manfaat, model tersimpan, batch inference, laporan dan materi otomatis.

Grafik calibration adalah diagnostik, bukan recalibration. SHAP, survival, causal uplift,
MLflow, live CRM dan production monitoring dibahas sebagai pengembangan, belum diimplementasikan.
Semua angka slide/insight berasal dari pipeline sehingga tidak tertinggal ketika seed berubah.

## Membaca hasil dengan benar

Teks dan bentuk pada PowerPoint dapat diedit; grafik tertanam sebagai PNG dan dapat dibuat ulang
melalui kode. PDF materi dibuat dari definisi layout yang sama; rendering Office dapat berbeda
sedikit bila font DejaVu Sans tidak tersedia dan diganti oleh PowerPoint.

Model prediktif membantu menemukan siapa berisiko dan kapan volume berubah. Feature importance,
korelasi lag, dan before-after tidak membuktikan mengapa. Harga naik serentak tanpa control;
paket ini tidak menyimpulkan efek harga tidak signifikan secara statistik.

Top10 dipilih per bulan; confusion matrix menggunakan threshold validation tetap. Acquisition
backtest adalah satu minggu ke depan pada setiap origin dengan actual minggu lalu tersedia.
Nilai keuntungan memakai asumsi efek intervensi, bukan keuntungan yang sudah tercapai.

Baca [insight hasil run](reports/INSIGHTS.md) untuk angka aktual dan keterbatasannya, serta
[referensi primer](docs/REFERENCES.md) untuk dokumentasi teknis dan sumber konteks Indonesia.

## Troubleshooting

| Masalah | Langkah |
|---|---|
| ModuleNotFoundError | Aktifkan `.venv`; pastikan interpreter notebook berasal dari environment sama |
| Versi Python tidak didukung dependency lock | Gunakan Python yang sesuai manifest atau install `requirements.txt` |
| Matplotlib tidak bisa menulis font cache | Set `MPLCONFIGDIR` ke direktori yang dapat ditulis sebelum menjalankan |
| Notebook kernel tidak ditemukan | Pilih `.venv/bin/python` / `.venv\Scripts\python.exe` di IDE |
| Hasil tidak identik | Cocokkan seed, jumlah nasabah, versi Python dan library pada manifest |
| Kategori baru saat inference | Encoder menerima unknown; tinjau apakah kategori baru adalah drift |
| Fitur input scoring kurang | Lengkapi feature contract; CLI menyebutkan kolom yang hilang |

Lihat `docs/VERIFICATION.md` untuk bukti eksekusi yang telah dilakukan.
