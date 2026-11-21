import { describe, expect, it } from "vitest";
import { toCanvasBox } from "./boxes";

describe("toCanvasBox", () => {
  it("maps normalized coordinates to the displayed image", () => {
    expect(toCanvasBox({ x1: 0.1, y1: 0.2, x2: 0.5, y2: 0.8 }, 1000, 500)).toEqual({
      x: 100,
      y: 100,
      width: 400,
      height: 300,
    });
  });
});
