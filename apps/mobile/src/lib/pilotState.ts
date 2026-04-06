export type DraftCase = {
  siteId: string;
  assetId: string;
  question: string;
  operatorContext: string;
  expectedStateLabel: string;
  pendingCaseId?: string | null;
};

export function restoreDraftCase(defaultDraft: DraftCase, raw: string | null): DraftCase {
  if (!raw) {
    return defaultDraft;
  }
  try {
    return { ...defaultDraft, ...(JSON.parse(raw) as Partial<DraftCase>) };
  } catch {
    return defaultDraft;
  }
}

export function pendingCaseLabel(caseId: string | null | undefined): string | null {
  if (!caseId) {
    return null;
  }
  return `Analysis pending for ${caseId}. Retry upload when connectivity stabilizes.`;
}

export function pendingCaseStatusMessage(status: string): string {
  if (status === "pending_analysis") {
    return "This case is still waiting for a successful upload or analysis retry.";
  }
  return "This draft case still needs a successful media upload.";
}
