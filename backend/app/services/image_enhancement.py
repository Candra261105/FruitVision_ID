from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter, ImageStat


@dataclass(frozen=True)
class ImageQualityReport:
    width: int
    height: int
    brightness: float
    contrast: float
    noise: float
    glare_ratio: float
    blur_score: float
    warnings: list[str]


@dataclass(frozen=True)
class EnhancementReport:
    status: str
    applied_steps: list[str]
    before: ImageQualityReport
    after: ImageQualityReport


def _quality(image: Image.Image) -> ImageQualityReport:
    gray = image.convert("L")
    histogram = gray.histogram()
    total = max(1, gray.width * gray.height)
    brightness = sum(value * count for value, count in enumerate(histogram)) / total
    cumulative = 0
    low = 0
    high = 255
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= total * 0.05:
            low = value
            break
    cumulative = 0
    for value in range(255, -1, -1):
        cumulative += histogram[value]
        if cumulative >= total * 0.05:
            high = value
            break
    contrast = float(high - low)
    variance = ImageStat.Stat(gray).var[0]
    residual = ImageStat.Stat(gray.filter(ImageFilter.MedianFilter(size=3))).var[0]
    noise = min(10.0, residual**0.5 / 10.0)
    edge_mean = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0]
    glare_ratio = sum(histogram[250:]) / total
    warnings: list[str] = []
    if brightness < 90:
        warnings.append("gelap")
    elif brightness > 175:
        warnings.append("terlalu_terang")
    if contrast < 18 and (brightness < 90 or brightness > 175):
        warnings.append("kontras_rendah")
    if noise > 1.5:
        warnings.append("noise_tinggi")
    if glare_ratio > 0.05:
        warnings.append("silau")
    return ImageQualityReport(
        width=gray.width,
        height=gray.height,
        brightness=round(brightness, 2),
        contrast=contrast,
        noise=round(noise, 2),
        glare_ratio=round(glare_ratio, 4),
        blur_score=round(edge_mean, 2),
        warnings=warnings,
    )


def inspect_image(image: Image.Image) -> ImageQualityReport:
    return _quality(image.convert("RGB"))


def enhance_image(image: Image.Image) -> tuple[Image.Image, EnhancementReport]:
    original = image.convert("RGB")
    before = _quality(original)
    enhanced = original
    applied_steps: list[str] = []

    if "gelap" in before.warnings:
        enhanced = enhanced.point(lambda value: min(255, int(255 * (value / 255) ** 0.65)))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        applied_steps.append("koreksi_cahaya")
    elif "terlalu_terang" in before.warnings:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(0.9)
        applied_steps.append("koreksi_cahaya")

    if "noise_tinggi" in before.warnings:
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        applied_steps.append("denoise")

    after = _quality(enhanced)
    return enhanced, EnhancementReport(
        status="enhanced" if applied_steps else "unchanged",
        applied_steps=applied_steps,
        before=before,
        after=after,
    )
