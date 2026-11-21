from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib import request


@dataclass(frozen=True)
class AIRequest:
    detections: list[dict[str, object]]
    quality_warnings: list[str]


@dataclass(frozen=True)
class AIResult:
    status: str
    detail: str
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    disclaimer: str = "Hasil AI bukan diagnosis ahli atau keputusan grade pabrik."


class AIExplainer:
    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        transport: Callable[[str, dict[str, str], bytes], bytes] | None = None,
    ) -> None:
        self.provider = (provider or os.getenv("FRUITVISION_AI_PROVIDER", "openrouter")).lower()
        self.api_key = api_key if api_key is not None else self._key_from_environment()
        self.model = model or self._model_from_environment()
        self.transport = transport or self._default_transport

    def _key_from_environment(self) -> str | None:
        if self.provider == "groq":
            return os.getenv("GROQ_API_KEY")
        return os.getenv("OPENROUTER_API_KEY")

    def _model_from_environment(self) -> str:
        if self.provider == "groq":
            return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        return os.getenv("OPENROUTER_MODEL", "")

    def _endpoint(self) -> str:
        return "https://api.groq.com/openai/v1/chat/completions" if self.provider == "groq" else "https://openrouter.ai/api/v1/chat/completions"

    def unavailable(self, detail: str) -> AIResult:
        return AIResult(
            status="unavailable",
            detail=detail,
            recommendations=["Gunakan hasil detector dan lakukan pemeriksaan manual."],
        )

    @staticmethod
    def _default_transport(url: str, headers: dict[str, str], body: bytes) -> bytes:
        req = request.Request(url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=12) as response:
            return response.read()

    def explain(self, payload: AIRequest) -> AIResult:
        if not self.api_key:
            return AIResult(
                status="unavailable",
                detail="AI provider belum aktif karena API key belum diisi di backend.",
                recommendations=["Gunakan hasil detector dan lakukan pemeriksaan manual."],
            )
        if self.provider not in {"openrouter", "groq"}:
            return AIResult(
                status="unavailable",
                detail="Provider AI tidak didukung.",
                recommendations=["Gunakan hasil detector dan lakukan pemeriksaan manual."],
            )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": 350,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten penjelas inspeksi hasil panen. "
                            "Gunakan hanya bukti CV yang diberikan. Jangan mendiagnosis penyakit, "
                            "kematangan, busuk, ukuran, atau grade jika data itu tidak tersedia. "
                            "Balas JSON dengan summary, recommendations, uncertainties."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"detections": payload.detections, "quality_warnings": payload.quality_warnings},
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            raw = json.loads(self.transport(self._endpoint(), headers, body))
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict):
                raise ValueError("respons AI bukan object")
            recommendations = parsed.get("recommendations", [])
            uncertainties = parsed.get("uncertainties", [])
            if not isinstance(recommendations, list) or not isinstance(uncertainties, list):
                raise ValueError("field respons AI harus berupa list")
            if len(recommendations) > 8 or len(uncertainties) > 8:
                raise ValueError("respons AI terlalu panjang")
            if not all(isinstance(value, str) and value.strip() for value in [*recommendations, *uncertainties]):
                raise ValueError("item respons AI harus berupa string")
            return AIResult(
                status="available",
                detail="Penjelasan AI berhasil dibuat dari hasil CV.",
                summary=str(parsed.get("summary", "")),
                recommendations=recommendations,
                uncertainties=uncertainties,
            )
        except (KeyError, TypeError, ValueError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return AIResult(
                status="error",
                detail=f"AI provider gagal digunakan ({type(exc).__name__}). Hasil CV tetap tersedia.",
                recommendations=["Gunakan hasil detector dan lakukan pemeriksaan manual."],
            )
