import { describe, expect, it } from "vitest";

import {
  pendingCaseLabel,
  pendingCaseStatusMessage,
  restoreDraftCase,
} from "./pilotState";

describe("pilotState", () => {
  it("restores draft values over defaults", () => {
    const result = restoreDraftCase(
      {
        siteId: "",
        assetId: "",
        panelFamily: "family-a",
        panelId: "",
        question: "",
        operatorContext: "",
        expectedStateLabel: "",
        pendingCaseId: null,
      },
      JSON.stringify({ siteId: "site-1", pendingCaseId: "case-123" }),
    );
    expect(result.siteId).toBe("site-1");
    expect(result.panelFamily).toBe("family-a");
    expect(result.pendingCaseId).toBe("case-123");
  });

  it("returns default draft for invalid json", () => {
    const fallback = restoreDraftCase(
      {
        siteId: "seed",
        assetId: "",
        panelFamily: "family-a",
        panelId: "",
        question: "",
        operatorContext: "",
        expectedStateLabel: "",
        pendingCaseId: null,
      },
      "{bad json",
    );
    expect(fallback.siteId).toBe("seed");
  });

  it("builds a pending label only when a case exists", () => {
    expect(pendingCaseLabel(null)).toBeNull();
    expect(pendingCaseLabel("case-42")).toContain("case-42");
  });

  it("maps pending and draft statuses to useful messages", () => {
    expect(pendingCaseStatusMessage("pending_analysis")).toContain("analysis retry");
    expect(pendingCaseStatusMessage("draft")).toContain("media upload");
  });
});
