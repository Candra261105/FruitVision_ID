from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--output", type=Path, default=Path("test-results/fruitvision-smoke.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        executable = os.getenv(
            "FRUITVISION_CHROMIUM",
            str(Path(os.environ["LOCALAPPDATA"]) / "ms-playwright/chromium-1255/chrome-win64/chrome.exe"),
        )
        browser = playwright.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.get_by_role("heading", name=re.compile(r"Lihat (hasil panen|buah) lebih jelas")).wait_for()
        page.get_by_label("Unggah gambar").wait_for(state="attached")
        page.get_by_role("heading", name="Hasil pengamatan").wait_for()
        page.screenshot(path=str(args.output), full_page=True)
        page.screenshot(path=str(args.output), full_page=True)
        browser.close()
    print(f"Smoke test lulus: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
