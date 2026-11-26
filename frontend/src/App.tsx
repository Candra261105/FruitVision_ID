import { useEffect, useState } from "react";
import { explainPrediction, fetchModelInfo, predictImage } from "./api";
import { CameraPanel } from "./components/CameraPanel";
import { ResultPanel } from "./components/ResultPanel";
import { StatusBanner } from "./components/StatusBanner";
import type { AIExplanation, ModelInfo, PredictionResponse } from "./types";

export default function App() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiMessage, setAiMessage] = useState<string | null>(null);
  const [aiResult, setAiResult] = useState<AIExplanation | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchModelInfo(controller.signal)
      .then((info) => {
        setModel(info);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError("Backend belum berjalan. Jalankan run_all.cmd lalu muat ulang halaman.");
      });
    return () => controller.abort();
  }, []);

  async function handleImage(file: Blob) {
    setBusy(true);
    setError(null);
    setPrediction(null);
    setAiMessage(null);
    setAiResult(null);

    const nextUrl = URL.createObjectURL(file);
    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return nextUrl;
    });

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 8000);

    try {
      setPrediction(await predictImage(file, controller.signal));
    } catch (reason) {
      setError(
        reason instanceof DOMException && reason.name === "AbortError"
          ? "Proses melewati batas 8 detik. Coba gambar yang lebih kecil."
          : reason instanceof Error
            ? reason.message
            : "Gambar tidak dapat diproses.",
      );
    } finally {
      window.clearTimeout(timeout);
      setBusy(false);
    }
  }

  async function handleExplain() {
    if (!prediction || aiBusy) return;
    setAiBusy(true);
    setAiMessage(null);
    try {
      const explanation = await explainPrediction(prediction);
      setAiResult(explanation);
      setAiMessage(`${explanation.summary || explanation.detail} ${explanation.recommendations.join(" ")}`.trim());
    } catch (reason) {
      setAiMessage(reason instanceof Error ? reason.message : "Penjelasan AI tidak tersedia.");
    } finally {
      setAiBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="FruitVision ID beranda">
          <span className="brand-seed" />
          FruitVision <em>ID</em>
        </a>
        <div className="topbar-meta">
          <span className="offline-tag">lokal · offline</span>
        </div>
      </header>

      <section className="intro">
        <div className="intro-content">
          <p className="eyebrow">Inspeksi tanaman dan hasil panen</p>
          <h1>Lihat hasil panen lebih jelas.</h1>
          <p className="intro-copy">
            Ambil snapshot atau unggah foto. Sistem memperbaiki gambar secara konservatif sebelum mendeteksi tomat, labu, dan lemon.
          </p>
        </div>
        <div className="class-strip" aria-label="Daftar komoditas target">
          <span className="class-strip-label">Target:</span>
          <span className="class-tag">tomat</span>
          <span className="class-tag">labu</span>
          <span className="class-tag">lemon</span>
        </div>
      </section>

      {error && <StatusBanner kind="error">{error}</StatusBanner>}

      <div className="workspace">
        <CameraPanel busy={busy} onImage={(file) => void handleImage(file)} />
        <ResultPanel
          imageUrl={imageUrl}
          model={model}
          prediction={prediction}
          ai={aiResult}
          busy={busy}
          onExplain={() => void handleExplain()}
          aiBusy={aiBusy}
          aiMessage={aiMessage}
        />
      </div>

      <footer className="footer">
        <span>FruitVision ID · Laboratorium PCT106</span>
        <span>Deteksi 3 Komoditas + Evaluasi Kualitas Citra</span>
        <span>Kondisi aktif mengikuti komoditas dan status model yang ditampilkan</span>
      </footer>
    </main>
  );
}
