from __future__ import annotations

from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy, IssueEvidence


def test_rank_issues_uses_calibrated_weights() -> None:
    policy = CalibratedTriagePolicy(
        issue_weights={
            "stuck_breaker": [3.0] + [0.0] * 10,
            "bad_relay": [0.5] + [0.0] * 10,
        },
        issue_bias={"stuck_breaker": 0.0, "bad_relay": 0.0},
        default_issue_weights=[0.0] * 11,
        default_issue_bias=-1.0,
        escalation_weights=[0.0] * 9,
        escalation_bias=-1.0,
        escalation_thresholds={
            "escalate_to_senior_technician": 0.8,
            "escalate_for_visual_review": 0.5,
        },
    )

    ranked = policy.rank_issues(
        [
            IssueEvidence(
                issue_class="stuck_breaker",
                features={"incident_sum": 1.0},
                rationale="Incident evidence points at breaker B4.",
            ),
            IssueEvidence(
                issue_class="bad_relay",
                features={"incident_sum": 1.0},
                rationale="Relay evidence is weaker.",
            ),
        ],
        top_k=2,
    )

    assert ranked[0][0] == "stuck_breaker"
    assert ranked[0][1] > ranked[1][1]


def test_choose_escalation_uses_calibrated_thresholds() -> None:
    policy = CalibratedTriagePolicy(
        issue_weights={},
        issue_bias={},
        default_issue_weights=[0.0] * 11,
        default_issue_bias=-1.0,
        escalation_weights=[2.0] + [0.0] * 8,
        escalation_bias=-1.5,
        escalation_thresholds={
            "escalate_to_senior_technician": 0.85,
            "escalate_for_visual_review": 0.4,
        },
    )

    action, risk = policy.choose_escalation({"top_issue_confidence": 1.0})

    assert action == "escalate_for_visual_review"
    assert 0.4 <= risk < 0.85
