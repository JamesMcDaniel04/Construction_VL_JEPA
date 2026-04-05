import { describe, expect, it } from "vitest";

import { helpfulRateLabel, inviteIssuedMessage } from "./pilotAdmin";

describe("pilotAdmin helpers", () => {
  it("formats helpfulness safely when there is no feedback yet", () => {
    expect(helpfulRateLabel(null)).toBe("No feedback yet");
  });

  it("formats helpfulness as a rounded percentage", () => {
    expect(helpfulRateLabel(0.534)).toBe("53%");
  });

  it("builds the one-time invite message", () => {
    expect(inviteIssuedMessage("Alex", "token-123")).toContain("Alex");
    expect(inviteIssuedMessage("Alex", "token-123")).toContain("token-123");
  });
});
