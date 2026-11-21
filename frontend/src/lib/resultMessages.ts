export function emptyResultExplanation(input: { mode: string; limitation: string }): string {
  if (input.mode === "demo") {
    return "Tidak ada box dengan keyakinan cukup. Mode fallback demo masih terbatas, jadi hasil ini perlu review dan bukan bukti objek tidak ada.";
  }
  if (input.mode === "unavailable") {
    return "Deteksi belum dijalankan karena model belum tersedia. Periksa status model sebelum review gambar.";
  }
  return `${input.limitation} Coba foto dengan objek utuh, cukup terang, dan tidak terlalu kecil. Jika tetap kosong, tandai untuk review manual.`;
}

export function detectionWarnings(warnings: string[]): string[] {
  return warnings.filter(Boolean);
}
