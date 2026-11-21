import { useEffect, useRef } from "react";
import type { Detection } from "../types";
import { toCanvasBox } from "../lib/boxes";

type DetectionCanvasProps = { imageUrl: string | null; detections: Detection[] };

export function DetectionCanvas({ imageUrl, detections }: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;
    const image = new Image();
    image.onload = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const maxWidth = 900;
      const scale = Math.min(1, maxWidth / image.naturalWidth);
      canvas.width = image.naturalWidth * scale;
      canvas.height = image.naturalHeight * scale;
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      detections.forEach((detection, index) => {
        const box = toCanvasBox(detection.box, canvas.width, canvas.height);
        context.strokeStyle = "#f2c14e";
        context.lineWidth = 3;
        context.strokeRect(box.x, box.y, box.width, box.height);
        context.fillStyle = "#f2c14e";
        context.font = "600 14px system-ui";
        const label = `${index + 1} ${detection.fruit}`;
        const labelWidth = context.measureText(label).width + 14;
        context.fillRect(box.x, Math.max(0, box.y - 25), labelWidth, 25);
        context.fillStyle = "#142c27";
        context.fillText(label, box.x + 7, Math.max(17, box.y - 8));
      });
    };
    image.src = imageUrl;
  }, [detections, imageUrl]);

  if (!imageUrl) return <div className="result-empty">Hasil snapshot akan tampil di sini.</div>;
  return <canvas ref={canvasRef} className="result-canvas" aria-label="Gambar dengan bounding box deteksi" />;
}
