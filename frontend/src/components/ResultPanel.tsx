import type { AIExplanation, Detection, ModelInfo, PredictionResponse } from "../types";
import { DetectionCanvas } from "./DetectionCanvas";
import { capabilityLabel, capabilityStatusLabel } from "../lib/capabilities";
import { detectionWarnings, emptyResultExplanation } from "../lib/resultMessages";
import { useState } from "react";
import { selectedDetection } from "../lib/objectSelection";
import { ObjectCropSelector } from "./ObjectCropSelector";
import { ColorIndicatorChart } from "./ColorIndicatorChart";

type ResultPanelProps = { imageUrl: string | null; model: ModelInfo | null; prediction: PredictionResponse | null; ai: AIExplanation | null; busy: boolean };

const CONDITION_LABELS: Record<string, string> = { maturity: "Kematangan", rot: "Busuk", disease: "Penyakit", grade: "Grade" };

function DetectionRow({ detection }: { detection: Detection }) {
  const conditions = Object.values(detection.conditions ?? {});
  const warnings = detectionWarnings(detection.warnings);
  return <li className="detection-row"><span className="object-number">{String(detection.id).padStart(2, "0")}</span><div><strong>{detection.fruit}</strong><span className="row-meta">{Math.round(detection.confidence * 100)}% yakin · {detection.review_status === "review" ? "perlu review" : "diterima"}</span>{warnings.length > 0 && <span className="condition-list">{warnings.map((warning) => <span className="condition-chip condition-review" key={warning}>{warning}</span>)}</span>}{conditions.length > 0 && <span className="condition-list">{conditions.map((condition) => <span key={condition.task} className={`condition-chip condition-${condition.status}`} title={condition.detail || undefined}>{CONDITION_LABELS[condition.task] ?? condition.task}: {condition.status === "available" ? `${condition.label_display} (${Math.round((condition.confidence ?? 0) * 100)}%)` : "perlu review"}</span>)}</span>}</div><span className="stage-pill">{conditions.length > 0 ? "dinilai per objek" : detection.stage_available ? detection.stage : "kematangan belum tersedia"}</span></li>;
}

export function ResultPanel({ imageUrl, model, prediction, ai, busy }: ResultPanelProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const enhancedUrl = prediction?.enhanced_image ?? imageUrl;
  const selected = selectedDetection(prediction?.detections ?? [], selectedId);
  const conditionActive = ["disease", "maturity", "rot", "grade"].some((task) => prediction?.capabilities[task]?.status === "available");
  return <section className="result-panel" aria-labelledby="result-title">
    <div className="panel-heading"><div><p className="section-kicker">Pembacaan</p><h2 id="result-title">Hasil pengamatan</h2></div>{busy && <span className="loading-label">memproses...</span>}</div>
    <div className="image-comparison"><figure><figcaption>Enhanced + bounding box</figcaption><div className="result-visual"><DetectionCanvas imageUrl={enhancedUrl} detections={prediction?.detections ?? []} /></div></figure>{prediction?.original_image && <figure><figcaption>Asli</figcaption><div className="result-visual"><img className="original-preview" src={prediction.original_image} alt="Foto asli sebelum enhancement" /></div></figure>}</div>
     {prediction && <p className="prediction-message">{prediction.detections.length === 0 ? emptyResultExplanation(prediction) : prediction.message}</p>}
    {prediction?.enhancement_status && <div className="result-note"><strong>{prediction.enhancement_status === "enhanced" ? "Gambar diperjelas sebelum deteksi." : "Gambar sudah cukup jelas."}</strong><span>{prediction.enhancement_steps.length ? `Tahap: ${prediction.enhancement_steps.join(", ")}.` : "Tidak ada filter yang dipaksakan."}</span></div>}
    {prediction?.quality && <div className="quality-grid"><div><span>Kecerahan</span><strong>{prediction.quality.brightness}</strong></div><div><span>Kontras</span><strong>{prediction.quality.contrast}</strong></div><div><span>Noise</span><strong>{prediction.quality.noise}</strong></div><div><span>Ketajaman</span><strong>{prediction.quality.blur_score}</strong></div><div><span>Silau</span><strong>{Math.round(prediction.quality.glare_ratio * 100)}%</strong></div><div><span>Resolusi</span><strong>{prediction.quality.width} x {prediction.quality.height}</strong></div></div>}
     {prediction && prediction.detections.length > 0 && <ul className="detection-list">{prediction.detections.map((detection) => <DetectionRow key={detection.id} detection={detection} />)}</ul>}
     {prediction && <ObjectCropSelector detections={prediction.detections} selectedId={selected?.id ?? null} onSelect={setSelectedId} />}
     {selected && <section className="selected-object" aria-labelledby="selected-object-title"><p className="section-kicker">Objek terpilih</p><h3 id="selected-object-title">Objek {selected.id}: {selected.fruit}</h3><p className="row-meta">{Math.round(selected.confidence * 100)}% yakin · hasil deteksi tersimpan untuk objek ini</p>{selected.color_indicator && <ColorIndicatorChart segments={selected.color_indicator.hue_segments} />}<div className="selected-status"><strong>Perlu verifikasi lanjutan oleh pengguna.</strong><span>Crop, bounding box, dan indikator warna membantu pemeriksaan awal. Sistem tidak mendiagnosis kematangan atau kebusukan dari HSV.</span></div>{selected.warnings.length > 0 && <ul>{selected.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}</section>}
    {prediction && <div className="capability-list"><strong>Status fitur inspeksi lanjutan</strong><span className="capability-intro">{conditionActive ? "Model kondisi aktif membuat penilaian per objek untuk komoditas yang tercantum di detail." : "Fitur di bawah ini belum membuat penilaian otomatis dari foto saat ini."}</span>{["disease", "maturity", "rot", "grade"].map((name) => { const capability = prediction.capabilities[name] ?? { status: "unavailable" as const, detail: "Belum tersedia." }; return <div className="capability-row" key={name}><div><span>{capabilityLabel(name)}</span><small>{capability.detail}</small></div><span className={`capability-status capability-${capability.status}`}>{capabilityStatusLabel(capability)}</span></div>; })}</div>}
    {prediction?.recommendations.length ? <div className="recommendation-list"><strong>Rekomendasi</strong><ul>{prediction.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}</ul></div> : null}
    {ai && <div className="ai-result"><strong>Status AI: {ai.status === "available" ? "tersedia" : ai.status === "error" ? "gagal, hasil CV tetap tersedia" : "belum aktif"}</strong><span>{ai.detail}</span>{ai.summary && <p>{ai.summary}</p>}{ai.uncertainties.length > 0 && <ul>{ai.uncertainties.map((item) => <li key={item}>{item}</li>)}</ul>}<small>{ai.disclaimer}</small></div>}
    {!prediction && !busy && <div className="result-note"><strong>Belum ada gambar.</strong><span>Ambil snapshot atau unggah gambar buah untuk memulai.</span></div>}
    {model && !model.model_available && <div className="model-warning"><strong>Model terlatih belum tersedia.</strong><span>Antarmuka siap, tetapi aplikasi tidak akan mengarang deteksi. Tambahkan bobot model saat tahap training selesai.</span></div>}
    {model?.model_available && model.mode === "demo" && <div className="model-warning"><strong>Mode demo aktif.</strong><span>Bobot fallback dapat membantu percobaan awal, tetapi belum dianggap model custom tiga kelas yang lolos gate evaluasi.</span></div>}
  </section>;
}
