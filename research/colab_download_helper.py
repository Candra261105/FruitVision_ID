"""Robust artifact packaging and safe download helper for Google Colab."""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence


def safe_download_artifacts(
    files_to_bundle: Sequence[Path],
    zip_dest: Path,
    drive_dest: Path | None = None,
) -> dict[str, Any]:
    """Bundle weights and charts into a zip and download or save to Drive safely."""
    status: dict[str, Any] = {
        "zip_created": False,
        "zip_path": str(zip_dest),
        "drive_copied": False,
        "direct_download": False,
        "errors": [],
    }

    # 1. Create consolidated ZIP archive
    try:
        zip_dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for item in files_to_bundle:
                if item.is_file():
                    z.write(item, arcname=item.name)
        status["zip_created"] = zip_dest.is_file() and zip_dest.stat().st_size > 0
    except Exception as err:
        status["errors"].append(f"ZIP error: {err}")

    # 2. Save to Google Drive if mounted
    if drive_dest and drive_dest.exists() and status["zip_created"]:
        try:
            drive_target = drive_dest / zip_dest.name
            shutil.copy2(zip_dest, drive_target)
            status["drive_copied"] = True
            print(f"[OK] Berhasil menyimpan salinan ke Google Drive: {drive_target}")
        except Exception as err:
            status["errors"].append(f"Drive copy error: {err}")

    # 3. Attempt direct browser download via google.colab.files
    try:
        from google.colab import files
        print(f"[*] Mencoba unduh otomatis via browser: {zip_dest.name} ...")
        files.download(str(zip_dest))
        time.sleep(2)
        status["direct_download"] = True
        print("[OK] Permintaan unduhan browser berhasil dikirim.")
    except Exception as err:
        # Expected in subshell non-interactive mode: AttributeError ('NoneType' object has no attribute 'kernel')
        status["errors"].append(f"Colab download skipped: {err}")
        print(f"\n[Info] Unduhan browser otomatis dilewati ({type(err).__name__}).")
        print(f"[Aman] File hasil akhir Anda telah dikemas rapi di:")
        print(f"  👉 {zip_dest.resolve()}")
        print("  (Anda dapat mengunduhnya dengan klik kanan file tersebut di panel file sebelah kiri Colab)")

    return status
