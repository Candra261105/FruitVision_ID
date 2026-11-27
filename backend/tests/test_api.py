from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.ai_explainer import AIRequest


client = TestClient(app)


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_model_state() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_model_info_exposes_training_classes() -> None:
    response = client.get("/api/model-info")

    assert response.status_code == 200
    classes = response.json()["classes"]
    assert "tomato" in classes
    assert len(classes) >= 3


def test_invalid_image_returns_422() -> None:
    response = client.post(
        "/api/predict/image",
        files={"file": ("broken.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 422
    assert "gambar" in response.json()["detail"].lower()


def test_valid_image_returns_explicit_model_status() -> None:
    response = client.post(
        "/api/predict/image",
        files={"file": ("fruit.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is True
    assert body["mode"] in {"demo", "custom"}
    assert isinstance(body["detections"], list)
    assert body["original_image"].startswith("data:image/jpeg;base64,")
    assert body["enhanced_image"].startswith("data:image/jpeg;base64,")
    assert "disease" in body["capabilities"]


def test_calibration_is_unavailable_in_mvp() -> None:
    response = client.get("/api/calibration-status")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_ai_explanation_is_explicitly_unavailable_without_backend_key() -> None:
    prediction = client.post(
        "/api/predict/image", files={"file": ("fruit.png", _image_bytes(), "image/png")}
    )
    response = client.post(
        "/api/explain", json={"inspection_id": prediction.json()["inspection_id"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert "diaktifkan" in body["detail"]


def test_ai_explanation_rejects_arbitrary_or_empty_cv_payload() -> None:
    response = client.post(
        "/api/explain",
        json={"inspection_id": "x" * 32},
    )

    assert response.status_code == 404


def test_ai_explanation_rejects_payload_without_cv_evidence() -> None:
    app.state.inspections["z" * 32] = AIRequest(detections=[], quality_warnings=[])
    response = client.post(
        "/api/explain", json={"inspection_id": "z" * 32}
    )

    assert response.status_code == 422


def test_oversized_upload_is_rejected() -> None:
    response = client.post(
        "/api/predict/image",
        files={"file": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 413


def test_image_dimensions_are_limited() -> None:
    buffer = BytesIO()
    Image.new("RGB", (8193, 1), "white").save(buffer, format="PNG")

    response = client.post(
        "/api/predict/image",
        files={"file": ("wide.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 413
