import type { DatasetPreview } from "@/types";

export const NEW_RUN_DRAFT_KEY = "pfas-aria:newRunDraft";

export type NewRunStep = "upload" | "configure" | "settings";

export type NewRunDraftV1 = {
  v: 1;
  preview: DatasetPreview;
  step: NewRunStep;
  outcome: string;
  features: string[];
  runName: string;
  maxRounds: number;
  hypPerRound: number;
  threshold: number;
};

const STEPS: NewRunStep[] = ["upload", "configure", "settings"];

function isStep(x: unknown): x is NewRunStep {
  return typeof x === "string" && (STEPS as readonly string[]).includes(x);
}

export function loadNewRunDraft(): NewRunDraftV1 | null {
  try {
    const raw = sessionStorage.getItem(NEW_RUN_DRAFT_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as Partial<NewRunDraftV1>;
    if (!d?.preview?.filename || !Array.isArray(d.preview.columns)) return null;
    const step = isStep(d.step) ? d.step : "configure";
    return {
      v: 1,
      preview: d.preview as DatasetPreview,
      step,
      outcome: typeof d.outcome === "string" ? d.outcome : "",
      features: Array.isArray(d.features) ? d.features.filter((x) => typeof x === "string") : [],
      runName: typeof d.runName === "string" ? d.runName : "",
      maxRounds: typeof d.maxRounds === "number" && Number.isFinite(d.maxRounds) ? d.maxRounds : 10,
      hypPerRound: typeof d.hypPerRound === "number" && Number.isFinite(d.hypPerRound) ? d.hypPerRound : 6,
      threshold: typeof d.threshold === "number" && Number.isFinite(d.threshold) ? d.threshold : 0.75,
    };
  } catch {
    return null;
  }
}

export function saveNewRunDraft(draft: NewRunDraftV1): void {
  try {
    sessionStorage.setItem(NEW_RUN_DRAFT_KEY, JSON.stringify(draft));
  } catch {
    /* quota or private mode */
  }
}

export function clearNewRunDraft(): void {
  try {
    sessionStorage.removeItem(NEW_RUN_DRAFT_KEY);
  } catch {
    /* ignore */
  }
}
