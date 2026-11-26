"""Generate realistic multi-object fruit composites for training YOLO11."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from research.procedural_backgrounds import generate_texture, STYLES

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
# Prioritize bottleneck classes: Lemon 45%, Tomato 35%, Pumpkin 20%
CLASS_WEIGHTS = {2: 0.45, 0: 0.35, 1: 0.20}


def apply_feathering(crop: Image.Image, radius: float = 1.0) -> Image.Image:
    """Add soft alpha border to crop for seamless visual blending."""
    w, h = crop.size
    rgba = crop.convert("RGBA")

    # Create solid white mask with soft border
    mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(mask)
    margin = max(1, int(radius * 1.5))
    # Draw zero alpha on outer edge
    for i in range(margin):
        alpha_val = int(255 * (i / margin))
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=alpha_val)

    mask = mask.filter(ImageFilter.BoxBlur(radius=max(1, int(radius))))
    # Preserve existing alpha channel from rotation expansion
    r, g, b, orig_a = rgba.split()
    final_a = ImageChops.multiply(orig_a, mask)
    rgba.putalpha(final_a)
    return rgba


def _load_crops_by_class(manifest_path: Path) -> dict[int, list[dict[str, Any]]]:
    """Group harvested crops by class ID."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for item in data.get("crops", []):
        cid = int(item["class_id"])
        if cid in grouped and Path(item["crop_path"]).is_file():
            grouped[cid].append(item)
    return grouped


def _sample_crop(
    grouped_crops: dict[int, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    """Sample a crop according to CLASS_WEIGHTS (45% lemon, 35% tomato, 20% pumpkin)."""
    r = rng.random()
    if r < CLASS_WEIGHTS[2]:
        cid = 2
    elif r < CLASS_WEIGHTS[2] + CLASS_WEIGHTS[0]:
        cid = 0
    else:
        cid = 1

    pool = grouped_crops.get(cid)
    if not pool:
        # Fallback to any non-empty pool
        available = [k for k, v in grouped_crops.items() if v]
        if not available:
            raise ValueError("No valid crops available in manifest")
        cid = rng.choice(available)
        pool = grouped_crops[cid]

    return rng.choice(pool)


def generate_composites(
    manifest_path: Path,
    output_dir: Path,
    count: int = 1500,
    canvas_size: tuple[int, int] = (800, 800),
    seed: int = 20261126,
) -> dict[str, Any]:
    """Generate realistic composite images with procedural backgrounds and rich occlusion."""
    rng = random.Random(seed)
    grouped_crops = _load_crops_by_class(manifest_path)

    out_images = output_dir / "images" / "train"
    out_labels = output_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    canv_w, canv_h = canvas_size
    manifest_items: list[dict[str, Any]] = []

    scenarios = ("separated", "adjacent", "touching", "overlap", "clustered", "scale_jitter")

    for idx in range(count):
        style = rng.choice(STYLES)
        canvas = generate_texture(canv_w, canv_h, style, rng)

        scenario = scenarios[idx % len(scenarios)]
        num_fruits = rng.randint(2, 6) if scenario in ("clustered", "adjacent", "overlap") else rng.randint(2, 4)

        placed_boxes: list[tuple[int, float, float, float, float]] = []
        item_metadata: list[dict[str, Any]] = []

        # Determine anchor point for clustering
        cluster_cx = rng.randint(int(canv_w * 0.3), int(canv_w * 0.7))
        cluster_cy = rng.randint(int(canv_h * 0.3), int(canv_h * 0.7))

        for f_idx in range(num_fruits):
            crop_meta = _sample_crop(grouped_crops, rng)
            crop_img = Image.open(crop_meta["crop_path"]).convert("RGB")

            # Scale fruit according to scenario and class
            cid = crop_meta["class_id"]
            if scenario == "scale_jitter" and rng.random() < 0.4:
                # Small distant fruit
                target_dim = rng.randint(35, 75)
            elif cid == 1:
                # Pumpkin is naturally larger
                target_dim = rng.randint(140, 260)
            elif cid == 2:
                # Lemon
                target_dim = rng.randint(55, 120)
            else:
                # Tomato
                target_dim = rng.randint(65, 140)

            # Preserve aspect ratio
            aspect = crop_img.width / max(1, crop_img.height)
            if aspect >= 1.0:
                fw = target_dim
                fh = max(16, int(target_dim / aspect))
            else:
                fh = target_dim
                fw = max(16, int(target_dim * aspect))

            crop_resized = crop_img.resize((fw, fh), Image.Resampling.BILINEAR)

            # Convert to RGBA before rotation so outer corners are transparent (alpha 0)
            crop_rgba = crop_resized.convert("RGBA")
            rot_angle = rng.uniform(-25, 25)
            crop_rot = crop_rgba.rotate(rot_angle, expand=True, resample=Image.Resampling.BILINEAR)

            # Lighting adjustment
            bright_factor = rng.uniform(0.85, 1.18)
            crop_rot = ImageEnhance.Brightness(crop_rot).enhance(bright_factor)
            if rng.random() < 0.3:
                contrast_factor = rng.uniform(0.9, 1.15)
                crop_rot = ImageEnhance.Contrast(crop_rot).enhance(contrast_factor)

            feathered = apply_feathering(crop_rot, radius=rng.uniform(0.8, 1.5))
            cw, ch = feathered.size

            # Placement coordinate determination
            if scenario == "clustered":
                px = max(10, min(canv_w - cw - 10, cluster_cx + rng.randint(-120, 120) - cw // 2))
                py = max(10, min(canv_h - ch - 10, cluster_cy + rng.randint(-120, 120) - ch // 2))
            elif scenario in ("touching", "overlap") and placed_boxes:
                # Place near previous fruit
                prev_cid, prev_cx, prev_cy, prev_bw, prev_bh = placed_boxes[-1]
                prev_x = int(prev_cx * canv_w)
                prev_y = int(prev_cy * canv_h)
                offset_dist = rng.randint(int(cw * 0.4), int(cw * 0.85)) if scenario == "overlap" else int(cw * 0.95)
                px = max(10, min(canv_w - cw - 10, prev_x + rng.choice([-1, 1]) * offset_dist))
                py = max(10, min(canv_h - ch - 10, prev_y + rng.choice([-1, 1]) * offset_dist))
            else:
                # Separated / distributed
                px = rng.randint(15, max(16, canv_w - cw - 15))
                py = rng.randint(15, max(16, canv_h - ch - 15))

            # Paste with alpha
            canvas.paste(feathered, (px, py), feathered)

            # YOLO bounding box clamped via edge coordinates
            x1 = max(0.0, float(px) / canv_w)
            y1 = max(0.0, float(py) / canv_h)
            x2 = min(1.0, float(px + cw) / canv_w)
            y2 = min(1.0, float(py + ch) / canv_h)
            norm_w = max(0.001, x2 - x1)
            norm_h = max(0.001, y2 - y1)
            center_x = max(0.001, min(0.999, (x1 + x2) / 2.0))
            center_y = max(0.001, min(0.999, (y1 + y2) / 2.0))

            placed_boxes.append((cid, center_x, center_y, norm_w, norm_h))
            item_metadata.append(
                {
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "crop_source": crop_meta["crop_path"],
                    "box": [center_x, center_y, norm_w, norm_h],
                }
            )

        stem = f"synthetic_exp_{seed}_{idx:05d}"
        img_out_path = out_images / f"{stem}.jpg"
        lbl_out_path = out_labels / f"{stem}.txt"

        canvas.save(img_out_path, quality=92)
        lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in placed_boxes]
        lbl_out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest_items.append(
            {
                "image": str(img_out_path),
                "label": str(lbl_out_path),
                "scenario": scenario,
                "background_style": style,
                "objects": item_metadata,
            }
        )

    report = {
        "count": count,
        "canvas_size": list(canvas_size),
        "seed": seed,
        "items": manifest_items,
    }
    (output_dir / "synthetic_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate realistic fruit composites.")
    parser.add_argument("--manifest", type=Path, default=Path("data/crops/harvest_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/tomato-pumpkin-lemon-expanded"))
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()

    print(f"[*] Generating {args.count} realistic fruit composites at {args.size}x{args.size}...")
    report = generate_composites(
        manifest_path=args.manifest,
        output_dir=args.output,
        count=args.count,
        canvas_size=(args.size, args.size),
        seed=args.seed,
    )
    print(f"[OK] Generated {report['count']} composites in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
