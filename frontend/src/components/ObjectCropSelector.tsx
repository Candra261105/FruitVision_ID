import type { Detection } from "../types";

type ObjectCropSelectorProps = {
  detections: Detection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
};

export function ObjectCropSelector({ detections, selectedId, onSelect }: ObjectCropSelectorProps) {
  if (detections.length === 0) return null;
  return <div className="object-selector" aria-label="Pilih objek hasil deteksi">
    {detections.map((detection) => <button
      className={`object-card${detection.id === selectedId ? " is-selected" : ""}`}
      key={detection.id}
      type="button"
      onClick={() => onSelect(detection.id)}
      aria-pressed={detection.id === selectedId}
    >
      {detection.crop_image ? <img src={detection.crop_image} alt={`Crop ${detection.fruit} objek ${detection.id}`} /> : <span className="object-card-empty">tanpa crop</span>}
      <span><strong>Objek {detection.id}</strong><small>{detection.fruit} · {Math.round(detection.confidence * 100)}%</small></span>
      {detection.warnings.length > 0 && <span className="object-card-warning">review</span>}
    </button>)}
  </div>;
}
