import { describe, expect, it } from "vitest";
import { selectedDetection, colorSegmentTotal } from "./objectSelection";
import type { Detection } from "../types";

const detection = (id: number): Detection => ({
  id,
  class_name: "tomato",
  fruit: "Tomat",
  confidence: 0.9,
  box: { x1: 0, y1: 0, x2: 0.5, y2: 0.5 },
  stage: null,
  stage_available: false,
  review_status: "accepted",
  warnings: [],
  conditions: {},
  crop_image: null,
  color_indicator: { hue_segments: { merah: 70, hijau: 20, lainnya: 10 }, saturation: 0.6, value: 0.7, status: "review", detail: "indikator" },
});

describe("object selection", () => {
  it("selects by stable detection id rather than array position", () => {
    expect(selectedDetection([detection(42), detection(7)], 7)?.id).toBe(7);
    expect(selectedDetection([detection(42)], 7)).toBeNull();
  });

  it("sums normalized color segments", () => {
    expect(colorSegmentTotal(detection(1).color_indicator?.hue_segments ?? {})).toBe(100);
  });
});
