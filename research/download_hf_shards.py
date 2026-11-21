from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

REPOS = {
    "grade": "Project-AgML/AFruitDB_fruit_grade_classification",
    "cucumber_disease": "Project-AgML/cucumber_disease_classification",
    "smartphone_vegetables": "LFTTim/multiple_smartphone_bangladesh_vegetable_detection",
    "tomato_detection_cc0": "Project-AgML/tomato_detection_kaggle",
}

DEST_ROOT = Path("data/advanced/source/hf_parquet")


def url_for(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"


def download_one(url: str, dest: Path, chunk: int = 4 * 1024 * 1024) -> tuple[Path, int, float]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "fruitvision-loop/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    began = time.time()
    with urllib.request.urlopen(req, timeout=120) as response:
        mode = "ab" if start and response.status == 206 else "wb"
        if mode == "wb":
            start = 0
        with part.open(mode) as handle:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                handle.write(block)
    part.replace(dest)
    return dest, dest.stat().st_size, time.time() - began


def list_parquet(repo: str) -> list[tuple[str, int]]:
    import json

    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/data"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return [(item["path"], int(item.get("size", 0))) for item in payload if item["path"].endswith(".parquet")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh shard parquet dataset HF secara resumable.")
    parser.add_argument("--dataset", choices=sorted(REPOS), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    repo = REPOS[args.dataset]
    target = DEST_ROOT / args.dataset
    files = list_parquet(repo)
    print(f"{args.dataset}: {len(files)} shard dari {repo}", flush=True)

    pending = []
    for path, expected in files:
        dest = target / Path(path).name
        if dest.exists() and dest.stat().st_size == expected:
            print(f"lewati (ukuran cocok) {dest.name} {dest.stat().st_size} B", flush=True)
            continue
        if dest.exists():
            print(f"hapus (ukuran salah {dest.stat().st_size} != {expected}) {dest.name}", flush=True)
            dest.unlink()
        pending.append((path, dest, expected))

    if not pending:
        print("semua shard sudah lengkap", flush=True)
        return 0

    failures = 0
    for path, dest, expected in pending:
        for attempt in range(1, 4):
            try:
                dest, size, seconds = download_one(url_for(repo, path), dest)
                if size != expected:
                    raise ValueError(f"ukuran akhir {size} != {expected}")
                speed = size / 1048576 / max(seconds, 0.001)
                print(f"OK {dest.name} {size} B {seconds:.1f}s {speed:.2f} MB/s", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"GAGAL percobaan {attempt} {dest.name}: {type(exc).__name__}: {exc}", flush=True)
                original = dest
                part = dest.with_suffix(dest.suffix + ".part")
                for leftover in (original, part):
                    if leftover.exists() and leftover.stat().st_size != expected:
                        leftover.unlink()
                if attempt == 3:
                    failures += 1
                else:
                    time.sleep(3)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
