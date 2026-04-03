from __future__ import annotations

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    CorpusSourceType,
    IncidentRecord,
    MediaType,
    ReferenceState,
    TriageRequest,
    VisualObservation,
)


def test_triage_service_image_flow(triage_service) -> None:
    triage_service.add_document(
        CorpusDocument(
            document_id="doc-sop",
            source_type=CorpusSourceType.sop,
            title="Fault Light SOP",
            body=(
                "Inspect breaker B4 for a half-open position. "
                "Reset the overload relay after verifying incoming isolation. "
                "Check fuse F7 before re-energizing the panel."
            ),
            tags=["stuck_breaker"],
        )
    )
    triage_service.add_incident(
        IncidentRecord(
            incident_id="inc-1",
            title="Breaker B4 half-open with red fault light",
            summary="Red fault lamp illuminated while breaker B4 sat between positions.",
            issue_class="stuck_breaker",
            fix_summary="Reset breaker B4 and replace the scorched contact block.",
        )
    )
    triage_service.add_reference_state(
        ReferenceState(
            state_id="state-normal",
            media_type=MediaType.image,
            state_label="normal_panel_state",
            description="All lamps green with aligned breaker handles.",
            caption="Normal electrical panel with no fault indicators.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.1] * (3 * 16 * 16),
        )
    )
    triage_service.add_reference_state(
        ReferenceState(
            state_id="state-fault",
            media_type=MediaType.image,
            state_label="fault_light_on",
            description="Red fault lamp active around breaker B4.",
            caption="Electrical panel with red fault light illuminated near breaker B4.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.9] * (3 * 16 * 16),
        )
    )

    response = triage_service.analyze(
        TriageRequest(
            question="The red fault light is on near breaker B4. What should happen next?",
            expected_state_label="fault_light_on",
            observation=VisualObservation(
                observation_id="obs-1",
                media_type=MediaType.image,
                tensor_shape=[3, 16, 16],
                tensor_values=[0.92] * (3 * 16 * 16),
            ),
        )
    )
    assert response.issue_candidates[0].issue_class == "stuck_breaker"
    assert response.state_assessment.matched_state_label == "fault_light_on"
    assert response.state_assessment.matches_expected is True
    assert len(response.next_steps) >= 1
    assert len(response.similar_incidents) == 1


def test_triage_service_video_flow(triage_service) -> None:
    triage_service.add_reference_state(
        ReferenceState(
            state_id="video-fault",
            media_type=MediaType.video,
            state_label="fault_light_on",
            description="Fault lamp flashes during the clip.",
            caption="Short clip showing a red fault light pulsing on the panel.",
            tensor_shape=[3, 10, 16, 16],
            tensor_values=[0.8] * (3 * 10 * 16 * 16),
        )
    )
    response = triage_service.analyze(
        TriageRequest(
            question="Does this clip look normal?",
            observation=VisualObservation(
                observation_id="obs-video-1",
                media_type=MediaType.video,
                tensor_shape=[3, 10, 16, 16],
                tensor_values=[0.78] * (3 * 10 * 16 * 16),
            ),
        )
    )
    assert response.state_assessment.matched_state_label == "fault_light_on"
    assert response.escalation_recommendation in {
        "proceed_with_guided_inspection",
        "escalate_for_visual_review",
        "escalate_to_senior_technician",
    }
