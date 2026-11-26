import { useEffect, useRef, useState } from "react";

type CameraPanelProps = {
  busy: boolean;
  onImage: (file: Blob) => void;
};

export function CameraPanel({ busy, onImage }: CameraPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);

  useEffect(() => () => streamRef.current?.getTracks().forEach((track: MediaStreamTrack) => track.stop()), []);

  async function startCamera() {
    if (starting) return;
    setCameraError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Browser ini tidak menyediakan akses kamera. Gunakan tombol unggah gambar.");
      return;
    }
    setStarting(true);
    streamRef.current?.getTracks().forEach((track: MediaStreamTrack) => track.stop());
    streamRef.current = null;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch {
      streamRef.current?.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      streamRef.current = null;
      setActive(false);
      setCameraError("Kamera tidak diizinkan atau sedang dipakai aplikasi lain. Gunakan unggah gambar.");
    } finally {
      setStarting(false);
    }
  }

  function capture() {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob((blob) => blob && onImage(blob), "image/jpeg", 0.9);
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track: MediaStreamTrack) => track.stop());
    streamRef.current = null;
    setActive(false);
  }

  return (
    <section className="camera-panel" aria-labelledby="camera-title">
      <div className="panel-heading">
        <div>
          <p className="section-kicker">Ruang Pengamatan</p>
          <h2 id="camera-title">Arahkan ke buah</h2>
        </div>
        <span className={`live-dot ${active ? "is-live" : ""}`}>
          {active ? "aktif" : "siap"}
        </span>
      </div>

      <div className="camera-frame">
        <video ref={videoRef} className={active ? "visible" : ""} muted playsInline aria-label="Pratinjau kamera" />
        {!active && (
          <div className="camera-empty">
            <span className="camera-mark" aria-hidden="true">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                <circle cx="12" cy="13" r="3" />
              </svg>
            </span>
            <p>Kamera belum dinyalakan</p>
            <small>Nyalakan kamera perangkat atau pilih file foto dari komputer.</small>
          </div>
        )}
      </div>

      <div className="camera-actions">
        {!active ? (
          <button className="button button-primary" onClick={() => void startCamera()} disabled={busy || starting}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polygon points="23 7 16 12 23 17 23 7" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
            {starting ? "Menyalakan..." : "Nyalakan kamera"}
          </button>
        ) : (
          <button className="button button-primary" onClick={capture} disabled={busy}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            Ambil snapshot
          </button>
        )}

        {active && (
          <button className="button button-quiet" onClick={stopCamera}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            </svg>
            Matikan
          </button>
        )}

        <label className="button button-quiet file-button" htmlFor="fruit-upload">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          Unggah gambar
          <input
            id="fruit-upload"
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onImage(file);
              event.target.value = "";
            }}
            disabled={busy}
          />
        </label>
      </div>

      {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}
    </section>
  );
}
