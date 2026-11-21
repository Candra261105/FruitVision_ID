import { describe, expect, it } from "vitest";
import { emptyResultExplanation, detectionWarnings } from "./resultMessages";

describe("result messages", () => {
  it("explains an empty demo result without claiming the object is absent", () => {
    expect(emptyResultExplanation({ mode: "demo", limitation: "Fallback demo aktif." })).toContain("fallback");
    expect(emptyResultExplanation({ mode: "demo", limitation: "Fallback demo aktif." })).toContain("review");
  });

  it("returns crop and edge warnings for the visible result panel", () => {
    expect(detectionWarnings(["objek menyentuh tepi"])).toEqual(["objek menyentuh tepi"]);
  });
});
