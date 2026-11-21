import { describe, expect, it } from "vitest";
import { capabilityLabel, capabilityStatusLabel } from "./capabilities";

describe("capability display labels", () => {
  it("shows user-facing names for advanced inspection features", () => {
    expect(capabilityLabel("disease")).toBe("Penyakit");
    expect(capabilityLabel("maturity")).toBe("Kematangan");
    expect(capabilityLabel("rot")).toBe("Busuk");
    expect(capabilityLabel("grade")).toBe("Grade kualitas");
  });

  it("shows unavailable features explicitly", () => {
    expect(capabilityStatusLabel({ status: "unavailable", detail: "" })).toBe("Belum tersedia");
    expect(capabilityStatusLabel({ status: "review_required", detail: "" })).toBe("Perlu review");
  });

  it("does not turn unavailable features into biological negative results", () => {
    expect(capabilityStatusLabel({ status: "unavailable", detail: "" })).not.toMatch(/sehat|tidak busuk|grade/i);
  });

  it("keeps rejected and blocked states distinct", () => {
    expect(capabilityStatusLabel({ status: "rejected", detail: "" })).toBe("Ditolak gate");
    expect(capabilityStatusLabel({ status: "blocked", detail: "" })).toBe("Terblokir");
  });
});
