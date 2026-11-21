from app.services.ai_explainer import AIExplainer, AIRequest


def test_ai_explainer_is_unavailable_without_provider_key() -> None:
    result = AIExplainer(provider="openrouter", api_key=None).explain(
        AIRequest(detections=[], quality_warnings=["gelap"])
    )

    assert result.status == "unavailable"
    assert "API key" in result.detail


def test_ai_explainer_parses_provider_json_without_turning_it_into_diagnosis() -> None:
    captured: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = body
        return b'{"choices":[{"message":{"content":"{\\"summary\\":\\"Tomat terdeteksi.\\",\\"recommendations\\":[\\"Periksa manual.\\"],\\"uncertainties\\":[\\"Penyakit belum dapat ditentukan.\\"]}"}}]}'

    result = AIExplainer(
        provider="openrouter",
        api_key="secret-for-test",
        model="test/model",
        transport=transport,
    ).explain(AIRequest(detections=[{"class_name": "tomato", "confidence": 0.91}], quality_warnings=[]))

    assert result.status == "available"
    assert result.summary == "Tomat terdeteksi."
    assert result.recommendations == ["Periksa manual."]
    assert "secret-for-test" not in result.detail
    assert captured["headers"] == {
        "Authorization": "Bearer secret-for-test",
        "Content-Type": "application/json",
    }


def test_ai_explainer_provider_error_keeps_safe_fallback() -> None:
    def transport(_url: str, _headers: dict[str, str], _body: bytes) -> bytes:
        raise TimeoutError("provider timeout")

    result = AIExplainer(
        provider="groq",
        api_key="secret-for-test",
        transport=transport,
    ).explain(AIRequest(detections=[], quality_warnings=[]))

    assert result.status == "error"
    assert result.recommendations
    assert "diagnosis" in result.disclaimer.lower()


def test_ai_explainer_rejects_provider_lists_with_wrong_shape() -> None:
    def transport(_url: str, _headers: dict[str, str], _body: bytes) -> bytes:
        return b'{"choices":[{"message":{"content":"{\\"summary\\":\\"ok\\",\\"recommendations\\":\\"not-a-list\\",\\"uncertainties\\":[]}"}}]}'

    result = AIExplainer(api_key="secret", transport=transport).explain(
        AIRequest(detections=[{"class_name": "tomato", "confidence": 0.9}], quality_warnings=[])
    )

    assert result.status == "error"


def test_ai_explainer_rejects_non_string_provider_items() -> None:
    def transport(_url: str, _headers: dict[str, str], _body: bytes) -> bytes:
        return b'{"choices":[{"message":{"content":"{\\"summary\\":\\"ok\\",\\"recommendations\\":[1],\\"uncertainties\\":[]}"}}]}'

    result = AIExplainer(api_key="secret", transport=transport).explain(
        AIRequest(detections=[{"class_name": "tomato", "confidence": 0.9}], quality_warnings=[])
    )

    assert result.status == "error"
