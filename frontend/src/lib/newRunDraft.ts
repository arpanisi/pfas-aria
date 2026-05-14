import type { DatasetPreview } from "@/types";

/** Bump when server preview contract changes (e.g. LabelEncoder rows) so stale session rows are dropped. */
export const NEW_RUN_DRAFT_KEY = "pfas-aria:newRunDraft:v4-string-dtype";

const LEGACY_DRAFT_KEYS = [
  "pfas-aria:newRunDraft",
  "pfas-aria:newRunDraft:v2-labelencode",
  "pfas-aria:newRunDraft:v3-catcols-only",
] as const;

export type NewRunStep =
  | "upload"
  | "configure"
  | "clean_preview"
  | "hypotheses";

export type NewRunDraftV1 = {
  v: 1;
  preview: DatasetPreview;
  step: NewRunStep;
  outcome: string;
  features: string[];
  runName: string;
  threshold: number;
  screeningRunId?: string | null;
};

const STEPS: readonly NewRunStep[] = [
  "upload",
  "configure",
  "clean_preview",
  "hypotheses",
];

function isStep(x: unknown): x is NewRunStep {
  return typeof x === "string" && (STEPS as readonly string[]).includes(x as NewRunStep);
}

/** Map legacy session drafts to current step names. */
function normalizeDraftStep(x: unknown): NewRunStep {
  if (x === "settings") return "hypotheses";
  if (x === "review") return "clean_preview";
  if (x === "variables") return "configure";
  if (isStep(x)) return x;
  return "configure";
}

export function loadNewRunDraft(): NewRunDraftV1 | null {
  try {
    const raw = sessionStorage.getItem(NEW_RUN_DRAFT_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw) as Partial<NewRunDraftV1 & { step?: string }>;
    if (!d?.preview?.filename || !Array.isArray(d.preview.columns)) return null;
    return {
      v: 1,
      preview: d.preview as DatasetPreview,
      step: normalizeDraftStep(d.step),
      outcome: typeof d.outcome === "string" ? d.outcome : "",
      features: Array.isArray(d.features) ? d.features.filter((x) => typeof x === "string") : [],
      runName: typeof d.runName === "string" ? d.runName : "",
      threshold: typeof d.threshold === "number" && Number.isFinite(d.threshold) ? d.threshold : 0.75,
      screeningRunId: typeof d.screeningRunId === "string" ? d.screeningRunId : null,
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
    for (const k of LEGACY_DRAFT_KEYS) {
      sessionStorage.removeItem(k);
    }
  } catch {
    /* ignore */
  }
}
