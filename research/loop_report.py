from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = ROOT / "runs" / "loop"
MODELS = ROOT / "models"


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_iterations() -> list[dict]:
    path = LOOP_DIR / "iterations.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Susun laporan hasil loop FruitVision.")
    parser.add_argument("--out", type=Path, default=LOOP_DIR / "REPORT.md")
    args = parser.parse_args()

    rows = load_iterations()
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_target[str(row.get("target", "?"))].append(row)

    models = {}
    for task in ("detector", "maturity", "rot", "grade", "disease"):
        gate = read_json(MODELS / f"{task}.gate.json")
        models[task] = gate

    lines = [
        "# Laporan Loop FruitVision",
        "",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M')} · sumber data: `runs/loop/iterations.jsonl`",
        "",
        "## Status model",
        "",
        "| Model | Status gerbang | Metrik kunci | Catatan |",
        "|---|---|---|---|",
    ]
    for task, gate in models.items():
        status = gate.get("status", "belum ada")
        metrics = gate.get("metrics") or gate.get("evaluation") or {}
        if task == "detector":
            detail = f"mAP50 {metrics.get('map50', 0):.3f}" if metrics.get("map50") is not None else "-"
        else:
            f1 = metrics.get("macro_f1")
            detail = f"macro-F1 {f1:.3f} · support {metrics.get('support', '-')}" if isinstance(f1, (int, float)) else "-"
            if isinstance(f1, (int, float)):
                detail += f" · balanced {metrics.get('balanced_accuracy', 0):.3f}"
            reasons = gate.get("reasons") or []
            if reasons:
                detail += " · " + "; ".join(str(item) for item in reasons[:3])
        lines.append(f"| {task} | {status} | {detail} | {gate.get('dataset', '')} |")

    lines.extend(["", "## Riwayat iterasi", "", "| Target | Iterasi | Tahap | Hasil | Keterangan |", "|---|---|---|---|---|"])
    for target, entries in by_target.items():
        for index, entry in enumerate(entries, start=1):
            stage = entry.get("stage", "train")
            detail = str(entry.get("detail", "")).splitlines()
            first = detail[0] if detail else ""
            first = first.replace("|", "/")
            lines.append(f"| {target} | {index} | {stage} | {'LULUS' if entry.get('ok') else 'GAGAL'} | {first} |")

    failures = [row for row in rows if not row.get("ok")]
    lines.extend(["", "## Kegagalan dan tindakan", ""])
    if not failures:
        lines.append("Tidak ada iterasi yang gagal.")
    else:
        for row in failures:
            detail = str(row.get("detail", "")).replace("\n", " ")[:400]
            lines.append(f"- **{row.get('target')} / {row.get('stage', 'train')}**: {detail}")

    eligible = [task for task, gate in models.items() if gate.get("status") == "eligible"]
    lines.extend(
        [
            "",
            "## Kesimpulan jujur",
            "",
            f"Model aktif (lolos gerbang): {', '.join(eligible) if eligible else 'belum ada'}.",
            "",
            "Status `eligible` hanya berarti metrik test holdout melewati ambang yang ditetapkan. "
            "Model tidak diklaim berlaku di luar domain dataset sumbernya.",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"laporan ditulis: {args.out}")
    print("\n".join(lines[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
