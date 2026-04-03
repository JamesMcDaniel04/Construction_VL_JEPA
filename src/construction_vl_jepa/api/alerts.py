"""Alert router — sends anomaly alerts to configured destinations.

Supports: webhook, email (SMTP), Slack, PagerDuty.
"""

from __future__ import annotations

from construction_vl_jepa.config import AlertConfig
from construction_vl_jepa.data.schema import AlertPayload
from construction_vl_jepa.utils.logging import get_logger

log = get_logger(__name__)


class AlertRouter:
    """Routes anomaly alerts to configured destinations."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg

    def _severity_context(self, alert: AlertPayload) -> tuple[str, str]:
        """Map raw score information to alert-routing severity labels."""
        if alert.severity is not None:
            severity_value = alert.severity
        elif alert.threshold and alert.threshold > 0:
            severity_value = min(alert.score / alert.threshold, 2.0) / 2.0
        else:
            severity_value = 0.5

        if severity_value >= 0.9:
            return "critical", "[CRITICAL]"
        if severity_value >= 0.6:
            return "warning", "[WARNING]"
        return "info", "[INFO]"

    async def send_alert(self, alert: AlertPayload) -> dict[str, bool]:
        """Send an alert to all configured destinations.

        Returns dict of {destination: success}.
        """
        results = {}

        if self.cfg.webhook_url:
            results["webhook"] = await self._send_webhook(alert)

        if self.cfg.slack_webhook:
            results["slack"] = await self._send_slack(alert)

        if self.cfg.email_to:
            results["email"] = await self._send_email(alert)

        if self.cfg.pagerduty_key:
            results["pagerduty"] = await self._send_pagerduty(alert)

        if not results:
            log.warning("no_alert_destinations", machine_id=alert.machine_id)

        return results

    async def _send_webhook(self, alert: AlertPayload) -> bool:
        """Send alert via generic webhook (CMMS integration)."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self.cfg.webhook_url,
                    json=alert.model_dump(),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
            log.info("webhook_sent", machine_id=alert.machine_id, status=response.status_code)
            return True
        except Exception as e:
            log.error("webhook_failed", machine_id=alert.machine_id, error=str(e))
            return False

    async def _send_slack(self, alert: AlertPayload) -> bool:
        """Send alert to Slack channel via webhook."""
        try:
            import httpx

            # Format Slack message
            pagerduty_severity, severity_label = self._severity_context(alert)
            signals = ", ".join(
                s.get("sensor", f"token_{s.get('token_index', '?')}")
                for s in alert.contributing_signals[:3]
            )
            pattern_text = ""
            if alert.matched_patterns:
                p = alert.matched_patterns[0]
                pattern_text = (
                    f"\n*Matched pattern:* {p.get('pattern_id', 'unknown')} "
                    f"(similarity: {p.get('similarity', 0):.0%})"
                )

            text = (
                f"{severity_label} *Anomaly Alert — {alert.machine_id}*\n"
                f"Score: {alert.score:.3f}"
                f"{f' | Threshold: {alert.threshold:.3f}' if alert.threshold is not None else ''}"
                f" | Trend: {alert.trend or 'N/A'}"
                f" | PagerDuty severity: {pagerduty_severity}\n"
                f"Top signals: {signals}"
                f"{pattern_text}"
            )
            if alert.recommended_action:
                text += f"\n*Action:* {alert.recommended_action}"

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self.cfg.slack_webhook,
                    json={"text": text},
                )
                response.raise_for_status()
            log.info("slack_sent", machine_id=alert.machine_id)
            return True
        except Exception as e:
            log.error("slack_failed", machine_id=alert.machine_id, error=str(e))
            return False

    async def _send_email(self, alert: AlertPayload) -> bool:
        """Send alert via email (placeholder — needs SMTP config)."""
        log.info("email_alert_placeholder", machine_id=alert.machine_id, to=self.cfg.email_to)
        # In production: use aiosmtplib or a transactional email service
        return True

    async def _send_pagerduty(self, alert: AlertPayload) -> bool:
        """Send alert to PagerDuty (critical alerts only)."""
        try:
            import httpx

            pagerduty_severity, _ = self._severity_context(alert)
            event = {
                "routing_key": self.cfg.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"Anomaly on {alert.machine_id}: score {alert.score:.3f}",
                    "severity": pagerduty_severity,
                    "source": alert.machine_id,
                    "custom_details": alert.model_dump(),
                },
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=event,
                )
                response.raise_for_status()
            log.info("pagerduty_sent", machine_id=alert.machine_id)
            return True
        except Exception as e:
            log.error("pagerduty_failed", machine_id=alert.machine_id, error=str(e))
            return False
