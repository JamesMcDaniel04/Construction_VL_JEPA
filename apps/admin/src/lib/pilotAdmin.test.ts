import { describe, expect, it } from "vitest";

import { helpfulRateLabel, inviteIssuedMessage } from "./pilotAdmin";

describe("pilotAdmin helpers", () => {
  it("formats helpfulness safely when there is no feedback yet", () => {
    expect(helpfulRateLabel(null)).toBe("No feedback yet");
  });

  it("formats helpfulness as a rounded percentage", () => {
    expect(helpfulRateLabel(0.534)).toBe("53%");
  });

  it("builds the invite status message", () => {
    expect(inviteIssuedMessage("Alex", "alex@example.com", "sent")).toContain("Alex");
    expect(inviteIssuedMessage("Alex", "alex@example.com", "sent")).toContain(
      "alex@example.com",
    );
  });
});
