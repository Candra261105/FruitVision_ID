from __future__ import annotations

from PIL import Image

from app.inference.base import RawDetection


def demo_provider(_image: Image.Image) -> list[RawDetection]:
    """Return no fabricated detections; used only to exercise the web shell."""
    return []
