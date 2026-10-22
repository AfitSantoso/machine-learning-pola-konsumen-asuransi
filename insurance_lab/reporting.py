"""Data-driven narrative and a portable self-contained HTML report."""
import base64
import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import TARGET


def write_reports(churn, acquisition, monthly, audit, paths, out, seed):
    chosen = churn["metrics"].loc[(churn["metrics"].model == churn["selected_model"]) &
                                   (churn["metrics"].split == "test")].iloc[0]
    am = acquisition["metrics"].loc[(acquisition["metrics"].model == acquisition["selected_model"]) &
                                     (acquisition["metrics"].split == "test")].iloc[0]
    before = monthly.loc["2024-10-01":"2024-12-01"]
    immediate = monthly.loc["2025-01-01":"2025-02-01"]
    later = monthly.loc["2025-04-01":"2025-06-01"]
    rates = {name: float(part.churned.sum() / part.active.sum())
             for name, part in [("pre_hike", before), ("immediate", immediate), ("delayed", later)]}
    top_features = churn["importance"].head(5).feature.tolist()
    summary = {"synthetic": True, "seed": seed,
               "created_at": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
               "audit": audit, "churn_model": churn["selected_model"],
               "churn_test": chosen.drop(["model", "split", "selected_on_validation"]).to_dict(),
               "customer_bootstrap_intervals": churn["intervals"],
               "acquisition_model": acquisition["selected_model"],
               "acquisition_test": am.drop(["model", "split", "selected_on_validation"]).to_dict(),
               "descriptive_event_rates": rates, "validation_top_features": top_features,
               "validation_threshold": churn["threshold"]}
    (out / "artifacts/run_summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    text = f"""# Laporan insight — Insurance Behavior Lab

**SELURUH ANGKA ADALAH SINTETIS.** Seed {seed}; {audit['customers']:,} nasabah unik,
{audit['rows']:,} snapshot; Januari 2023–September 2026. Satu nasabah = satu polis.
Index ekonomi adalah variabel simulasi, bukan inflasi/BI-Rate aktual.

## Temuan deskriptif

1. Churn tertimbang jumlah polis aktif sebelum kenaikan (Okt–Des 2024): **{rates['pre_hike']:.2%}**.
   Pada Jan–Feb 2025: **{rates['immediate']:.2%}**. Pada Apr–Jun 2025: **{rates['delayed']:.2%}**.
   Ini perbandingan deskriptif periode dengan populasi yang berubah. Tidak ada kelompok kontrol
   kenaikan harga; analisis ini tidak menyatakan dampak harga tidak signifikan secara statistik.
2. Periksa interaksi renewal dan tunggakan pada `segment_renewal_payment.csv`. Rate memakai
   denominator polis aktif, bukan hanya jumlah pelanggan keluar. Uji apakah pola stabil pada
   cohort dan bulan lain; pelanggan berisiko tinggi yang sudah keluar mengubah komposisi portofolio.
3. Campaign dan penjualan tidak selalu sejalan pada minggu yang sama. Plot lag dan sinyal
   pencarian/referral membantu investigasi. Korelasi lag tetap tidak membuktikan sebab-akibat.

## Hasil model churn

Model dipilih menggunakan **average precision validation**, bukan test: **{churn['selected_model']}**.
Training Jan 2023–Jun 2025; validation Agu–Des 2025; test Feb–Sep 2026.
Jul 2025 dan Jan 2026 menjadi embargo. Tiga rolling backtest berada sebelum final validation.
Model final tetap model yang dilatih pada train; tidak dilatih ulang dengan validation.

| Metrik test | Nilai |
|---|---:|
| Prevalensi churn | {chosen.prevalence:.2%} |
| Average precision (AP) | {chosen.average_precision:.4f} |
| ROC-AUC | {chosen.roc_auc:.4f} |
| Brier score, lebih kecil lebih baik | {chosen.brier:.4f} |
| Precision pada top 10% per bulan | {chosen.precision_at_10pct:.2%} |
| Recall pada top 10% per bulan | {chosen.recall_at_10pct:.2%} |
| Lift top 10% | {chosen.lift_at_10pct:.2f}x |

Interval bootstrap nasabah (120 pengulangan) AP:
{churn['intervals']['average_precision']['lower_95']:.4f}–{churn['intervals']['average_precision']['upper_95']:.4f};
lift: {churn['intervals']['lift_at_10pct']['lower_95']:.2f}–{churn['intervals']['lift_at_10pct']['upper_95']:.2f}x.
Interval ini bersyarat pada model tetap, tidak memasukkan seluruh ketidakpastian akibat shock waktu
atau pemilihan model. Top 10% diurutkan per bulan, bukan menggunakan informasi bulan mendatang.
Confusion matrix memakai threshold tetap {churn['threshold']:.4f} yang ditentukan dari validation;
aturan ini berbeda dari kapasitas top 10% yang berubah mengikuti skor setiap bulan.

Sinyal importance validation teratas: **{', '.join(top_features)}**.
Permutation importance mengukur perubahan AP ketika fitur diacak, dengan deviasi antar
pengulangan, bukan confidence interval kausal. Fitur berkorelasi dapat saling menggantikan.
Grafik kalibrasi adalah diagnostik; paket ini tidak mengklaim sudah melakukan recalibration.

## Model akuisisi

Model terpilih: **{acquisition['selected_model']}**, berdasarkan MAE total mingguan validation.
MAE total test **{am.weekly_total_mae:.2f} polis/minggu** dan WAPE **{am.weekly_total_wape:.2%}**.
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
"""
    (out / "reports/INSIGHTS.md").write_text(text, encoding="utf-8")
    cards = [("Snapshot", f"{audit['rows']:,}"), ("Model churn", churn["selected_model"]),
             ("AP test", f"{chosen.average_precision:.3f}"), ("Lift top 10%", f"{chosen.lift_at_10pct:.2f}x")]
    sections = []
    captions = {
        "executive_dashboard": "Dashboard manajemen: pola volume, risiko dan sinyal prediktif.",
        "churn_trend": "Gunakan jumlah polis aktif sebagai denominator. Perubahan waktu belum mengisolasi efek harga.",
        "renewal_payment": "Segmen ini membantu membentuk hipotesis dan prioritas tinjauan layanan.",
        "acquisition_forecast": "Evaluasi satu minggu ke depan; bandingkan actual dengan prediksi.",
        "calibration": "Bandingkan probabilitas dengan frekuensi aktual sebelum memakai skor untuk perhitungan nilai.",
        "feature_importance": "Importance pada validation menunjukkan sinyal model, bukan penyebab.",
        "business_sensitivity": "Nilai harapan bergantung pada asumsi efek intervensi yang belum diuji.",
    }
    ordered = sorted(paths, key=lambda path: path.stem != "executive_dashboard")
    for path in ordered:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        caption = captions.get(path.stem, path.stem.replace("_", " ").capitalize())
        sections.append(f'<figure><img loading="lazy" src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    card_html = "".join(f'<article><small>{html.escape(label)}</small><strong>{html.escape(value)}</strong></article>' for label, value in cards)
    content = f"""<!doctype html><html lang="id"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Insurance Behavior Lab — Laporan</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#eef3f8;color:#112640;margin:0}}
header{{background:#112640;color:white;padding:48px max(6%,calc((100% - 1180px)/2))}}
header p{{max-width:900px;color:#c6d6e8}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.1}}
.tag{{color:#57e1cf;letter-spacing:.14em;font-size:12px;font-weight:700}}
main{{max-width:1180px;margin:32px auto;padding:0 20px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}}
article,figure,.note{{background:white;border-radius:12px;padding:22px;box-shadow:0 4px 18px #11264008}}
strong{{display:block;font-size:25px;overflow-wrap:anywhere}}small,figcaption{{color:#52657d}}figure{{margin:25px 0}}img{{width:100%;height:auto}}.note{{margin-top:24px}}
a{{color:#2563eb}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d9e3ec}}
</style><header><div class="tag">EDUCATIONAL CASE / SYNTHETIC DATA</div><h1>Ketika perilaku nasabah<br>terlihat tidak masuk akal.</h1><p>EDA, dua jalur model, validasi waktu, dan keputusan yang dapat diuji. Seluruh data dan angka merupakan simulasi pembelajaran.</p></header>
<main><section class="cards">{card_html}</section><div class="note"><b>Interpretasi utama</b><p>Churn sebelum kenaikan {rates['pre_hike']:.2%}, segera setelah kenaikan {rates['immediate']:.2%}, dan periode tertunda {rates['delayed']:.2%}. Ini deskripsi, bukan pengukuran dampak harga.</p><p>Model dipilih dari validation. AP dan lift mengukur prediksi; penyebab dan manfaat retensi memerlukan pengujian tambahan. Baca <a href="INSIGHTS.md">laporan insight</a> untuk asumsi dan keterbatasan.</p></div>{''.join(sections)}<div class="note">Dibuat dari pipeline Python dengan seed {seed}. HTML ini menyertakan semua gambar sehingga dapat dibuka offline.</div></main></html>"""
    (out / "reports/dashboard.html").write_text(content, encoding="utf-8")
    return summary
