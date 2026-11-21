export type NormalizedBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type CanvasBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export function toCanvasBox(box: NormalizedBox, width: number, height: number): CanvasBox {
  const rounded = (value: number): number => Math.round(value * 1000) / 1000;
  return {
    x: rounded(box.x1 * width),
    y: rounded(box.y1 * height),
    width: rounded((box.x2 - box.x1) * width),
    height: rounded((box.y2 - box.y1) * height),
  };
}
