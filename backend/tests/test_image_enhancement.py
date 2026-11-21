from PIL import Image

from app.services.image_enhancement import enhance_image, inspect_image


def test_inspect_image_reports_dark_low_contrast_photo() -> None:
    report = inspect_image(Image.new("RGB", (32, 32), (35, 35, 35)))

    assert report.brightness < 90
    assert report.contrast < 10
    assert "gelap" in report.warnings


def test_enhance_image_returns_rgb_image_and_explains_applied_steps() -> None:
    original = Image.new("RGB", (32, 32), (35, 35, 35))

    enhanced, report = enhance_image(original)

    assert enhanced.mode == "RGB"
    assert enhanced.size == original.size
    assert report.status == "enhanced"
    assert "koreksi_cahaya" in report.applied_steps
    assert enhanced.getpixel((0, 0))[0] > original.getpixel((0, 0))[0]


def test_clear_photo_is_not_forced_through_filters() -> None:
    original = Image.new("RGB", (32, 32), (120, 120, 120))

    enhanced, report = enhance_image(original)

    assert report.status == "unchanged"
    assert report.applied_steps == []
    assert enhanced.tobytes() == original.tobytes()


def test_quality_report_includes_blur_and_resolution() -> None:
    report = inspect_image(Image.new("RGB", (32, 24), (120, 120, 120)))

    assert report.width == 32
    assert report.height == 24
    assert report.blur_score >= 0
