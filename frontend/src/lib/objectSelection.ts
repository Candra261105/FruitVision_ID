import type { Detection } from "../types";

export function selectedDetection(detections: Detection[], selectedId: number | null): Detection | null {
  if (selectedId === null) return null;
  return detections.find((detection) => detection.id === selectedId) ?? null;
}

export function colorSegmentTotal(segments: Record<string, number>): number {
  return Math.round(Object.values(segments).reduce((total, value) => total + value, 0) * 100) / 100;
}
