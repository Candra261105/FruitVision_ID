from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class RawDetection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    stage: str | None = None


class InferenceProvider(Protocol):
    def __call__(self, image: Image.Image) -> list[RawDetection]: ...
