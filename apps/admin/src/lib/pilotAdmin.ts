export function helpfulRateLabel(rate: number | null | undefined): string {
  if (rate == null) {
    return "No feedback yet";
  }
  return `${Math.round(rate * 100)}%`;
}

export function inviteIssuedMessage(displayName: string, token: string): string {
  return `Invite created for ${displayName}. Token: ${token}`;
}
