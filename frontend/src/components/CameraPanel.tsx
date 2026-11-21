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
      setCameraError("Browser ini tidak menyediakan akses kamera. Gunakan unggah gambar.");
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
          <p className="section-kicker">Ruang pengamatan</p>
          <h2 id="camera-title">Arahkan ke buah</h2>
        </div>
        <span className={`live-dot ${active ? "is-live" : ""}`}>{active ? "aktif" : "siap"}</span>
      </div>
      <div className="camera-frame">
        <video ref={videoRef} className={active ? "visible" : ""} muted playsInline aria-label="Pratinjau kamera" />
        {!active && <div className="camera-empty"><span className="camera-mark">◌</span><p>Kamera belum dinyalakan</p><small>Atau pilih gambar dari perangkat untuk mulai.</small></div>}
      </div>
      <div className="camera-actions">
        {!active ? <button className="button button-primary" onClick={() => void startCamera()} disabled={busy || starting}>{starting ? "Menyalakan..." : "Nyalakan kamera"}</button> : <button className="button button-primary" onClick={capture} disabled={busy}>Ambil snapshot</button>}
        {active && <button className="button button-quiet" onClick={stopCamera}>Matikan</button>}
        <label className="button button-quiet file-button" htmlFor="fruit-upload">
          Unggah gambar
          <input id="fruit-upload" type="file" accept="image/*" onChange={(event) => { const file = event.target.files?.[0]; if (file) onImage(file); event.target.value = ""; }} disabled={busy} />
        </label>
      </div>
      {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}
    </section>
  );
}
