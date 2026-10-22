# Referensi dan penggunaan sumber

Ditinjau 22 Oktober 2026. Data dan angka dalam paket ini berasal dari simulator lokal,
bukan hasil pengunduhan statistik Indonesia. Referensi berikut membantu metodologi dan
pengembangan selanjutnya; tidak menunjukkan afiliasi dengan perusahaan teknologi tertentu.

## Metode teknis dari dokumentasi resmi

- [TimeSeriesSplit — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html):
  pemisahan waktu dan gap. Implementasi panel pada paket memakai mask tanggal eksplisit,
  agar semua nasabah/region pada timestamp sama tetap bersama, bukan TimeSeriesSplit per baris.
- [Average precision — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html):
  AP noninterpolated yang berbeda dari area trapezoid di bawah PR curve.
- [Probability calibration — scikit-learn](https://scikit-learn.org/stable/modules/calibration.html):
  reliability diagram, interpretasi probabilitas dan pemisahan data kalibrasi.
- [Permutation feature importance — scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html):
  evaluasi importance pada holdout dan keterbatasan fitur berkorelasi.
- [PoissonRegressor — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html):
  model count dengan log link dan regularisasi.
- [Common pitfalls — scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html):
  preprocessing yang konsisten dan pencegahan data leakage dengan Pipeline.

## Konteks Indonesia untuk pengayaan data nyata

- [Metadata Indeks Harga Konsumen — BPS](https://sirusa.web.bps.go.id/metadata/indikator/d568de59-cd83-b851-e5f8-e6f5c047f31f):
  konsep IHK, kelompok pengeluaran dan disaggregasi wilayah.
- [Indikator BI-Rate — Bank Indonesia](https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx):
  sumber resmi indikator suku bunga dan tanggal keputusan.

Jika ditambahkan ke proyek produksi, simpan tanggal referensi, tanggal rilis, nilai versi
pertama dan revisi. Join hanya menggunakan informasi yang telah tersedia saat scoring.
Jangan menganggap economic_stress_index dalam latihan sebagai salinan statistik BPS/BI.
