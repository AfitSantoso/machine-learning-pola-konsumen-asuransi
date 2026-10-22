# Panduan pengajar — buka setelah peserta menyusun hipotesis

## Tujuan studi kasus

Mengajarkan investigasi perilaku, disiplin waktu, perbandingan model dan desain keputusan.
Tidak ada satu jawaban sebab-akibat yang dapat ditemukan hanya dari model prediktif.
Peserta dinilai dari kekuatan bukti dan proses yang dapat direproduksi.

## Persiapan kelas

1. Ikuti setup README, jalankan pipeline dan pengujian pada laptop pengajar.
2. Buka deck PowerPoint atau PDF; speaker notes menyertakan arahan diskusi.
3. Distribusikan `notebooks/`, `data/`, `docs/DATA_DICTIONARY.md`, dan problem brief.
4. Pada latihan hipotesis awal, minta peserta menunda membaca `data.py` dan panduan ini.
   Kode tetap terbuka; penundaan ini hanya untuk menjaga proses discovery.
5. Bagi kelompok 3–4 orang: business analyst, data analyst, modeler dan reviewer.
6. Sediakan baseline hasil dan panduan troubleshooting agar waktu kelas untuk berpikir.

## Mekanisme yang sengaja ditanam di simulator

### Portofolio

- Data Jan 2023–Sep 2026; 6000 customer default, sebagian sudah memiliki polis saat data dimulai.
- Premi semua nasabah naik 10% pada Jan 2025. Koefisien langsung pada log-odds churn dibuat kecil
  (+0.08), sehingga simulator tidak memaksa lonjakan segera setelah harga naik.
- Pada Apr–Jun 2025 dan Apr–Jun 2026 tekanan ekonomi buatan dan probabilitas keluhan meningkat.
- Churn dipengaruhi tunggakan, keluhan, window renewal, recency kontak, rendahnya engagement,
  income segment dan interaksi renewal x tunggakan.
- Shock tertunda juga berinteraksi dengan sensitivitas laten customer. Risiko laten dan
  sensitivitas tidak diekspor sebagai fitur. Karena itu model tetap punya ketidakpastian.
- Simulasi menghasilkan covariates lebih dulu, baru outcome, lalu mengeluarkan customer churn.
  Agregat transaksi dibuat langsung, bukan diturunkan dari file transaksi individu.

Persamaan log-odds berada di `insurance_lab/data.py`. Ini **kunci mekanisme buatan**, bukan
penemuan kausal dari analisis mahasiswa. Bahkan jika model menemukan renewal sebagai fitur
penting, model observasional sendiri tidak membuktikan mengubah jadwal renewal akan menurunkan churn.

### Akuisisi

- Data region x minggu, Jan 2023–Sep 2026, lima region.
- Campaign terjadi dua minggu setiap 13 minggu. Efek simulator ke penjualan muncul pada lag 2.
- Demand laten meningkat pada Agu–Sep 2025 dan Jul–Agu 2026. Sinyalnya terlihat pada pencarian
  dan referral; sumber kejadian demand tidak dicatat.
- Mean count polis baru memakai activity minggu sebelumnya dan campaign dua minggu sebelumnya.
- Dataset acquisition tidak terhubung satu-ke-satu ke customer entry di simulator portofolio.

Peserta dapat melihat korelasi sinyal dengan penjualan tetapi tidak dapat menyimpulkan apa
kejadian dunia luar yang memicu demand. Jawaban yang baik mengusulkan instrumentasi dan eksperimen.

## Jawaban yang diharapkan dari tiga misteri

| Misteri | Jawaban yang sesuai bukti | Klaim yang perlu dikoreksi |
|---|---|---|
| Premi naik tetapi bertahan | Dampak segera tampak kecil; periksa nilai, friksi, renewal dan mix | Harga pasti tidak memengaruhi churn |
| Churn melonjak kemudian | Cari sinyal historis dan interaksi, validasi antar cohort/waktu | Model tahu alasan pasti setiap nasabah |
| Campaign sepi lalu ramai | Periksa response lag, organic demand dan attribution | Semua kenaikan berasal dari campaign |

Tekankan: kata "signifikan" perlu desain uji yang tepat. Paket ini tidak menguji efek kausal
harga dan tidak memiliki control harga. Pre/post rate saja tidak menyelesaikan confounding.

## Demonstrasi kelas yang disarankan

1. Tampilkan churn_trend sebelum menunjukkan fitur lain; kumpulkan hipotesis.
2. Tampilkan renewal_payment dan leading_signals; minta peserta memperbarui dugaan.
3. Tanyakan berapa banyak churn ditemukan bila hanya bisa menghubungi top10%.
4. Tampilkan hasil logistik dan boosting; minta alasan memilih model dari validation.
5. Baca false positives/negatives, lalu tunjukkan bahwa probabilitas bukan kepastian.
6. Tampilkan acquisition_trend lalu campaign_lag; diskusikan mengapa timing saja menyesatkan.
7. Akhiri dengan business_sensitivity dan rancangan randomisasi.

## Pertanyaan lisan dan poin jawaban

- **Mengapa random split bermasalah?** Calendar regime dan repeat observations bisa bercampur;
  kemampuan pada masa depan tidak terukur dengan jujur.
- **Mengapa gap?** Label butuh waktu matang; produksi juga membutuhkan reporting/grace period.
- **Apa arti AP?** Ringkasan precision-recall noninterpolated, berbeda dari trapezoidal PR-AUC.
- **Mengapa lift tinggi belum profit?** Biaya dan peluang menyelamatkan calon churn belum diketahui.
- **Mengapa Brier kecil belum cukup?** Outcome langka dan prediksi mendekati nol dapat memberi
  squared error kecil; bandingkan baseline dan reliability bins.
- **Mengapa region tidak menyatakan sebab?** Bisa menjadi proxy konteks dan komposisi nasabah.
- **Apa arti sel cohort kosong?** Follow-up belum lengkap, bukan retention nol.
- **Apakah forecast test memakai future actual?** Hanya minggu sebelumnya pada setiap origin;
  sah untuk one-step operational backtest, tidak sah jika diklaim multi-step dari satu origin.
- **Apa gunanya bootstrap customer?** Menjaga repeat rows satu nasabah, tetapi belum memasukkan
  shock waktu yang sama antar nasabah.

## Tantangan tambahan dan penilaian

Ablation fitur pendapatan harus menggunakan validation, bukan test. Penambahan kalibrasi
memerlukan pemisahan development tambahan. Fairness diperiksa lewat error dan manfaat layanan
per kelompok, bukan hanya kesamaan rata-rata skor. Surival/uplift/SHAP adalah advanced track,
bukan fitur yang diklaim sudah diimplementasikan pada paket inti.

Catat pertanyaan yang belum terjawab sebagai backlog instrumentasi. Mahasiswa yang menyatakan
"data ini belum cukup untuk mengetahui penyebab" lalu mengusulkan uji yang tepat menunjukkan
keterampilan analitis yang diharapkan.
