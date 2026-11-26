"""AI crop refinement, elliptical feathering, and multi-source crop extractor."""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refine_and_feather_crop(img: Image.Image, feather_radius: int = 4) -> Image.Image:
    """Apply smooth antialiased oval/elliptical alpha mask to remove black corner artifacts."""
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size

    # Create oversized mask for smooth downsampling anti-aliasing
    scale = 2
    mask_w, mask_h = w * scale, h * scale
    mask = Image.new("L", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(mask)

    # Inset slightly to guarantee corners remain transparent after blur
    inset = feather_radius * scale
    draw.ellipse((inset, inset, mask_w - inset - 1, mask_h - inset - 1), fill=255)

    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius * scale))

    # Resize mask back to original dimensions
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)

    # Attach feathered alpha channel
    img_rgba = img_rgb.convert("RGBA")
    img_rgba.putalpha(mask)
    return img_rgba


def extract_tomato_ripeness_crops(
    zip_path: Path,
    output_dir: Path,
    limit: int = 4000,
    feather: bool = True,
) -> list[dict[str, Any]]:
    """Extract curated tomato crops from tomato_ripeness_crops.zip."""
    records: list[dict[str, Any]] = []
    if not zip_path.is_file():
        return records

    target_dir = output_dir / "tomato"
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if count >= limit:
                break
            if name.endswith("/") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Accept tomato crops across ripeness classes
            if "cls_tomato" not in name:
                continue

            try:
                data = z.read(name)
                with Image.open(io.BytesIO(data)) as img:
                    w, h = img.size
                    if w < 16 or h < 16:
                        continue

                    stem = Path(name).stem
                    out_name = f"tomato_ripeness_{stem}.png" if feather else f"tomato_ripeness_{stem}.jpg"
                    out_path = target_dir / out_name

                    if feather:
                        feathered = refine_and_feather_crop(img)
                        feathered.save(out_path, format="PNG")
                    else:
                        img.convert("RGB").save(out_path, format="JPEG", quality=95)

                    records.append(
                        {
                            "class": "tomato",
                            "class_name": "tomato",
                            "class_id": 0,
                            "crop_path": str(out_path),
                            "source": "tomato_ripeness_crops",
                            "filename": out_name,
                            "width": w,
                            "height": h,
                            "sha256": _sha256_bytes(data),
                        }
                    )
                    count += 1
            except Exception:
                continue

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine and extract fruit crops.")
    parser.add_argument(
        "--ripeness-zip",
        type=Path,
        default=Path("data/advanced/source/tomato_ripeness_crops.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/fruit_crops"),
    )
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()

    print(f"[*] Extracting tomato crops from {args.ripeness_zip}...")
    records = extract_tomato_ripeness_crops(args.ripeness_zip, args.output_dir, limit=args.limit)
    print(f"[OK] Extracted {len(records)} tomato crops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
