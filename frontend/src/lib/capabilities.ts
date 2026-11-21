import type { CapabilityStatus } from "../types";

export const CAPABILITY_LABELS: Record<string, string> = {
  maturity: "Kematangan",
  rot: "Busuk",
  disease: "Penyakit",
  grade: "Grade kualitas",
};

export function capabilityLabel(name: string): string {
  return CAPABILITY_LABELS[name] ?? name;
}

export function capabilityStatusLabel(capability: CapabilityStatus): string {
  if (capability.status === "unavailable") return "Belum tersedia";
  if (capability.status === "review_required") return "Perlu review";
  if (capability.status === "rejected") return "Ditolak gate";
  if (capability.status === "blocked") return "Terblokir";
  return "Tersedia";
}
