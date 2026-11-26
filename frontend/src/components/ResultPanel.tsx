import { useState } from "react";
import type { AIExplanation, Detection, ModelInfo, PredictionResponse } from "../types";
import { DetectionCanvas } from "./DetectionCanvas";
import { capabilityLabel, capabilityStatusLabel } from "../lib/capabilities";
import { detectionWarnings, emptyResultExplanation } from "../lib/resultMessages";
import { selectedDetection } from "../lib/objectSelection";
import { ObjectCropSelector } from "./ObjectCropSelector";
import { ColorIndicatorChart } from "./ColorIndicatorChart";

type ResultPanelProps = {
  imageUrl: string | null;
  model: ModelInfo | null;
  prediction: PredictionResponse | null;
  ai: AIExplanation | null;
  busy: boolean;
  onExplain?: () => void;
  aiBusy?: boolean;
  aiMessage?: string | null;
};

const CONDITION_LABELS: Record<string, string> = {
  maturity: "Kematangan",
  rot: "Busuk",
  disease: "Penyakit",
  grade: "Grade",
};

function DetectionRow({ detection }: { detection: Detection }) {
  const conditions = Object.values(detection.conditions ?? {});
  const warnings = detectionWarnings(detection.warnings);
  return (
    <li className="detection-row">
      <span className="object-number">{String(detection.id).padStart(2, "0")}</span>
      <div>
        <strong>{detection.fruit}</strong>
        <span className="row-meta">
          {Math.round(detection.confidence * 100)}% keyakinan · {detection.review_status === "review" ? "perlu review" : "diterima"}
        </span>
        {warnings.length > 0 && (
          <span className="condition-list">
            {warnings.map((warning) => (
              <span className="condition-chip condition-review" key={warning}>
                {warning}
              </span>
            ))}
          </span>
        )}
        {conditions.length > 0 && (
          <span className="condition-list">
            {conditions.map((condition) => (
              <span
                key={condition.task}
                className={`condition-chip condition-${condition.status}`}
                title={condition.detail || undefined}
              >
                {CONDITION_LABELS[condition.task] ?? condition.task}:{" "}
                {condition.status === "available"
                  ? `${condition.label_display} (${Math.round((condition.confidence ?? 0) * 100)}%)`
                  : "perlu review"}
              </span>
            ))}
          </span>
        )}
      </div>
      <span className="stage-pill">
        {conditions.length > 0
          ? "dinilai per objek"
          : detection.stage_available
            ? detection.stage
            : "kematangan belum tersedia"}
      </span>
    </li>
  );
}

export function ResultPanel({
  imageUrl,
  model,
  prediction,
  ai,
  busy,
  onExplain,
  aiBusy = false,
  aiMessage = null,
}: ResultPanelProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"detections" | "quality" | "analysis">("detections");

  const enhancedUrl = prediction?.enhanced_image ?? imageUrl;
  const selected = selectedDetection(prediction?.detections ?? [], selectedId);
  const conditionActive = ["disease", "maturity", "rot", "grade"].some(
    (task) => prediction?.capabilities[task]?.status === "available",
  );

  return (
    <section className="result-panel" aria-labelledby="result-title">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Pembacaan</p>
          <h2 id="result-title">Hasil pengamatan</h2>
        </div>
        {busy ? (
          <span className="loading-label">memproses...</span>
        ) : (
          prediction && (
            <span className="offline-tag" style={{ fontSize: "11px" }}>
              {prediction.detections.length} objek ditemukan
            </span>
          )
        )}
      </div>

      {/* Segmented Tab Navigation */}
      <nav className="tab-nav" aria-label="Navigasi bagian hasil">
        <button
          type="button"
          className={`tab-button ${activeTab === "detections" ? "active" : ""}`}
          onClick={() => setActiveTab("detections")}
          aria-selected={activeTab === "detections"}
          role="tab"
        >
          <span>Deteksi & Objek</span>
          {prediction && (
            <span className="tab-badge" aria-label={`${prediction.detections.length} objek`}>
              {prediction.detections.length}
            </span>
          )}
        </button>
        <button
          type="button"
          className={`tab-button ${activeTab === "quality" ? "active" : ""}`}
          onClick={() => setActiveTab("quality")}
          aria-selected={activeTab === "quality"}
          role="tab"
        >
          <span>Kualitas Citra</span>
        </button>
        <button
          type="button"
          className={`tab-button ${activeTab === "analysis" ? "active" : ""}`}
          onClick={() => setActiveTab("analysis")}
          aria-selected={activeTab === "analysis"}
          role="tab"
        >
          <span>Kondisi & AI</span>
        </button>
      </nav>

      {/* Loading Skeleton */}
      {busy && (
        <div className="skeleton-container" aria-live="polite">
          <div className="skeleton-box" style={{ height: "260px" }} />
          <div className="skeleton-box" style={{ height: "64px" }} />
          <div className="skeleton-box" style={{ height: "96px" }} />
        </div>
      )}

      {/* Empty State when no prediction and not busy */}
      {!prediction && !busy && (
        <div className="result-content">
          <div className="result-visual">
            <div className="result-empty">
              <p style={{ margin: "0 0 6px", fontWeight: 600, color: "var(--color-text-primary)" }}>
                Belum ada pengamatan aktif
              </p>
              <span>Ambil snapshot dari kamera atau unggah foto untuk menjalankan deteksi dan quality check.</span>
            </div>
          </div>
          <div className="result-note">
            <strong>Siap menganalisis komoditas.</strong>
            <span>Sistem akan mendeteksi tomat, labu, dan lemon serta mengevaluasi metrik kecerahan dan kontras.</span>
          </div>
        </div>
      )}

      {/* Results Content by Active Tab */}
      {!busy && prediction && (
        <div className="result-content">
          {/* TAB 1: Deteksi & Objek */}
          {activeTab === "detections" && (
            <>
              <div className="image-comparison">
                <figure>
                  <figcaption>Enhanced + Bounding Box</figcaption>
                  <div className="result-visual">
                    <DetectionCanvas imageUrl={enhancedUrl} detections={prediction.detections} />
                  </div>
                </figure>
                {prediction.original_image && (
                  <figure>
                    <figcaption>Foto Asli</figcaption>
                    <div className="result-visual">
                      <img className="original-preview" src={prediction.original_image} alt="Foto asli sebelum enhancement" />
                    </div>
                  </figure>
                )}
              </div>

              <p className="prediction-message">
                {prediction.detections.length === 0 ? emptyResultExplanation(prediction) : prediction.message}
              </p>

              {prediction.detections.length > 0 && (
                <>
                  <ul className="detection-list" aria-label="Daftar buah terdeteksi">
                    {prediction.detections.map((detection) => (
                      <DetectionRow key={detection.id} detection={detection} />
                    ))}
                  </ul>

                  <ObjectCropSelector
                    detections={prediction.detections}
                    selectedId={selected?.id ?? null}
                    onSelect={setSelectedId}
                  />
                </>
              )}

              {selected && (
                <section className="selected-object" aria-labelledby="selected-object-title">
                  <p className="section-kicker">Objek Terpilih</p>
                  <h3 id="selected-object-title">
                    Objek {selected.id}: {selected.fruit}
                  </h3>
                  <p className="row-meta">
                    {Math.round(selected.confidence * 100)}% keyakinan · hasil deteksi terisolasi
                  </p>
                  {selected.color_indicator && <ColorIndicatorChart segments={selected.color_indicator.hue_segments} />}
                  <div className="selected-status">
                    <strong>Perlu verifikasi visual lanjutan oleh pengguna.</strong>
                    <span>
                      Crop dan indikator warna membantu pemeriksaan awal di kebun/lab. Sistem tidak mendiagnosis kematangan atau kebusukan dari HSV semata.
                    </span>
                  </div>
                  {selected.warnings.length > 0 && (
                    <ul style={{ margin: "8px 0 0", paddingLeft: "16px", color: "var(--color-status-warning)", fontSize: "12px" }}>
                      {selected.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  )}
                </section>
              )}
            </>
          )}

          {/* TAB 2: Kualitas Citra */}
          {activeTab === "quality" && (
            <>
              {prediction.enhancement_status && (
                <div className="result-note">
                  <strong>
                    {prediction.enhancement_status === "enhanced"
                      ? "Gambar diperjelas sebelum deteksi."
                      : "Gambar sudah cukup jelas."}
                  </strong>
                  <span>
                    {prediction.enhancement_steps.length
                      ? `Tahapan perbaikan: ${prediction.enhancement_steps.join(", ")}.`
                      : "Koreksi adaptif tidak mengubah parameter gambar secara berlebihan."}
                  </span>
                </div>
              )}

              {prediction.quality && (
                <div className="quality-grid" aria-label="Metrik kualitas citra">
                  <div className="quality-card">
                    <span>Kecerahan</span>
                    <strong>{prediction.quality.brightness}</strong>
                  </div>
                  <div className="quality-card">
                    <span>Kontras</span>
                    <strong>{prediction.quality.contrast}</strong>
                  </div>
                  <div className="quality-card">
                    <span>Noise</span>
                    <strong>{prediction.quality.noise}</strong>
                  </div>
                  <div className="quality-card">
                    <span>Ketajaman</span>
                    <strong>{prediction.quality.blur_score}</strong>
                  </div>
                  <div className="quality-card">
                    <span>Silau (Glare)</span>
                    <strong>{Math.round(prediction.quality.glare_ratio * 100)}%</strong>
                  </div>
                  <div className="quality-card">
                    <span>Resolusi</span>
                    <strong>
                      {prediction.quality.width} × {prediction.quality.height}
                    </strong>
                  </div>
                </div>
              )}

              {prediction.quality?.warnings && prediction.quality.warnings.length > 0 && (
                <div className="model-warning">
                  <strong>Peringatan Kualitas Citra:</strong>
                  <span>{prediction.quality.warnings.join(". ")}</span>
                </div>
              )}
            </>
          )}

          {/* TAB 3: Kondisi & AI */}
          {activeTab === "analysis" && (
            <>
              <div className="capability-list">
                <strong>Status Fitur Inspeksi Lanjutan</strong>
                <span className="capability-intro">
                  {conditionActive
                    ? "Model kondisi aktif membuat penilaian per objek untuk komoditas yang didukung."
                    : "Fitur di bawah ini belum membuat penilaian otomatis dari foto saat ini."}
                </span>
                {["disease", "maturity", "rot", "grade"].map((name) => {
                  const capability = prediction.capabilities[name] ?? {
                    status: "unavailable" as const,
                    detail: "Belum tersedia.",
                  };
                  return (
                    <div className="capability-row" key={name}>
                      <div>
                        <span>{capabilityLabel(name)}</span>
                        <small>{capability.detail}</small>
                      </div>
                      <span className={`capability-status capability-${capability.status}`}>
                        {capabilityStatusLabel(capability)}
                      </span>
                    </div>
                  );
                })}
              </div>

              {prediction.recommendations.length > 0 && (
                <div className="recommendation-list">
                  <strong>Rekomendasi Tindakan Panen</strong>
                  <ul>
                    {prediction.recommendations.map((recommendation) => (
                      <li key={recommendation}>{recommendation}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Integrated AI Assistant */}
              <section className="ai-section" aria-label="Asisten AI">
                <div className="ai-header">
                  <div>
                    <p className="section-kicker">Konsultasi Pakar</p>
                    <h3>Asisten AI (Opsional)</h3>
                    <p>
                      Kirimkan ringkasan metrik CV untuk interpretasi agronomis umum. Keputusan panen tetap di tangan pengguna.
                    </p>
                  </div>
                  {onExplain && (
                    <button
                      type="button"
                      className="button button-primary"
                      onClick={onExplain}
                      disabled={aiBusy}
                    >
                      {aiBusy ? "Meminta penjelasan..." : "Minta penjelasan AI"}
                    </button>
                  )}
                </div>

                {aiMessage && <p className="prediction-message">{aiMessage}</p>}

                {ai && (
                  <div className="ai-result">
                    <strong>
                      Status AI: {ai.status === "available" ? "tersedia" : ai.status === "error" ? "gagal, hasil CV tetap aman" : "belum aktif"}
                    </strong>
                    <span>{ai.detail}</span>
                    {ai.summary && <p>{ai.summary}</p>}
                    {ai.uncertainties.length > 0 && (
                      <ul>
                        {ai.uncertainties.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    )}
                    <small>{ai.disclaimer}</small>
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      )}

      {/* Persistent Model Status Notices (Crucial for test compatibility & transparency) */}
      {model && !model.model_available && (
        <div className="model-warning" style={{ marginTop: "16px" }}>
          <strong>Model terlatih belum tersedia.</strong>
          <span>
            Antarmuka siap, tetapi aplikasi tidak akan mengarang deteksi. Tambahkan bobot model saat tahap training selesai.
          </span>
        </div>
      )}
      {model?.model_available && model.mode === "demo" && (
        <div className="model-warning" style={{ marginTop: "16px" }}>
          <strong>Mode demo aktif.</strong>
          <span>
            Bobot fallback dapat membantu percobaan awal, tetapi belum dianggap model custom tiga kelas yang lolos gate evaluasi.
          </span>
        </div>
      )}
      {model?.model_available && model.mode === "trial" && (
        <div className="model-warning" style={{ marginTop: "16px" }}>
          <strong>Mode uji coba aktif ({model.name}).</strong>
          <span>
            {model.limitation || "Model sedang dipakai untuk uji coba. Hasilnya belum lolos gate evaluasi final."}
          </span>
        </div>
      )}
    </section>
  );
}
