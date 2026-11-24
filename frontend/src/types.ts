export type Box = { x1: number; y1: number; x2: number; y2: number };

export type ConditionAssessment = {
  task: "maturity" | "rot" | "disease" | "grade";
  status: "available" | "review";
  label: string | null;
  label_display: string | null;
  confidence: number | null;
  detail: string;
};

export type ColorIndicator = {
  hue_segments: Record<string, number>;
  saturation: number;
  value: number;
  status: "available" | "review";
  detail: string;
};

export type Detection = {
  id: number;
  class_name: string;
  fruit: string;
  confidence: number;
  box: Box;
  stage: string | null;
  stage_available: boolean;
  review_status: "accepted" | "review";
  warnings: string[];
  conditions: Record<string, ConditionAssessment>;
  crop_image: string | null;
  color_indicator: ColorIndicator | null;
};

export type QualityReport = {
  width: number;
  height: number;
  brightness: number;
  contrast: number;
  noise: number;
  glare_ratio: number;
  blur_score: number;
  warnings: string[];
};

export type CapabilityStatus = {
  status: "available" | "unavailable" | "review_required" | "rejected" | "blocked";
  detail: string;
};

export type AIExplanation = {
  status: "available" | "unavailable" | "error";
  detail: string;
  summary: string;
  recommendations: string[];
  uncertainties: string[];
  disclaimer: string;
};

export type PredictionResponse = {
  model_available: boolean;
  mode: "custom" | "demo" | "test" | "unavailable";
  detections: Detection[];
  message: string;
  limitation: string;
  original_image: string | null;
  enhanced_image: string | null;
  enhancement_status: "enhanced" | "unchanged";
  enhancement_steps: string[];
  quality: QualityReport | null;
  capabilities: Record<string, CapabilityStatus>;
  recommendations: string[];
  ai: AIExplanation | null;
  inspection_id: string | null;
};

export type ModelInfo = {
  name: string;
  mode: string;
  model_available: boolean;
  classes: string[];
  stage_classes: string[];
  limitation: string;
};
