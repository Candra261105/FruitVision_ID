import type { AIExplanation, ModelInfo, PredictionResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function parseError(response: Response): Promise<never> {
  try {
    const body = (await response.json()) as { detail?: string };
    throw new Error(body.detail ?? `Permintaan gagal (${response.status}).`);
  } catch (error) {
    if (error instanceof Error) throw error;
    throw new Error(`Permintaan gagal (${response.status}).`);
  }
}

export async function fetchModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const response = await fetch(`${API_URL}/api/model-info`, { signal });
  if (!response.ok) await parseError(response);
  return (await response.json()) as ModelInfo;
}

export async function predictImage(file: Blob, signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file, "snapshot.jpg");
  const response = await fetch(`${API_URL}/api/predict/image`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!response.ok) await parseError(response);
  return (await response.json()) as PredictionResponse;
}

export async function explainPrediction(prediction: PredictionResponse, signal?: AbortSignal): Promise<AIExplanation> {
  const response = await fetch(`${API_URL}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inspection_id: prediction.inspection_id }),
    signal,
  });
  if (!response.ok) await parseError(response);
  return (await response.json()) as AIExplanation;
}
