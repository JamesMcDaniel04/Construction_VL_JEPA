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
            metadata={"issue_class": "stuck_breaker"},
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
    assert response.next_steps[0].step == "Inspect breaker B4 for a half-open position"
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


def test_triage_service_extracts_numbered_sop_steps_without_truncation(triage_service) -> None:
    triage_service.add_document(
        CorpusDocument(
            document_id="doc-numbered-sop",
            source_type=CorpusSourceType.sop,
            title="Neutral Lug Heat SOP",
            body=(
                "1. Measure phase-to-neutral voltage at terminal block N2.\n"
                "2. Tighten the neutral lug at N2 if movement is present.\n"
                "3. Escalate immediately if discoloration or heat damage is visible."
            ),
            tags=["loose_neutral"],
        )
    )
    triage_service.add_incident(
        IncidentRecord(
            incident_id="inc-neutral-1",
            title="Heat damage around neutral lug N2",
            summary="Discoloration and heat buildup were visible near neutral lug N2.",
            issue_class="loose_neutral",
            fix_summary="Tighten the lug and replace the terminal block if heat damage is present.",
        )
    )
    triage_service.add_reference_state(
        ReferenceState(
            state_id="state-neutral-hot",
            media_type=MediaType.image,
            state_label="heat_discoloration_visible",
            description="Heat staining around neutral lug N2.",
            caption="Panel with visible discoloration around neutral lug N2.",
            metadata={"issue_class": "loose_neutral"},
            tensor_shape=[3, 16, 16],
            tensor_values=[0.6] * (3 * 16 * 16),
        )
    )

    response = triage_service.analyze(
        TriageRequest(
            question="What should I do about the heat mark near neutral lug N2?",
            expected_state_label="normal_panel_state",
            observation=VisualObservation(
                observation_id="obs-neutral-1",
                media_type=MediaType.image,
                tensor_shape=[3, 16, 16],
                tensor_values=[0.62] * (3 * 16 * 16),
            ),
        )
    )

    steps = [item.step for item in response.next_steps]
    assert "Measure phase-to-neutral voltage at terminal block N2" in steps
    assert "Tighten the neutral lug at N2 if movement is present" in steps
    assert all(not step.endswith("present") or "movement is present" in step for step in steps)


def test_triage_service_prefers_incident_linked_sop_steps(triage_service) -> None:
    triage_service.add_document(
        CorpusDocument(
            document_id="doc-linked-sop",
            source_type=CorpusSourceType.sop,
            title="Breaker Recovery SOP",
            body=(
                "1. Inspect breaker B4 for a half-open position.\n"
                "2. Reset the overload relay only after lockout-tagout is verified.\n"
                "3. Verify fuse F7 before re-energizing the panel."
            ),
            tags=["stuck_breaker"],
        )
    )
    triage_service.add_document(
        CorpusDocument(
            document_id="doc-distractor",
            source_type=CorpusSourceType.manual,
            title="Fault Light Overview",
            body=(
                "The red fault light can appear during several panel conditions. "
                "Photograph the indicator and note the panel identifier before review."
            ),
            tags=["general_fault_light"],
        )
    )
    triage_service.add_incident(
        IncidentRecord(
            incident_id="inc-linked-1",
            title="Breaker B4 fault light with half-open handle",
            summary="A red fault lamp illuminated while breaker B4 sat between positions.",
            issue_class="stuck_breaker",
            fix_summary="Reset breaker B4 after isolation and verify the overload relay.",
            linked_document_ids=["doc-linked-sop"],
        )
    )
    triage_service.add_reference_state(
        ReferenceState(
            state_id="state-linked-fault",
            media_type=MediaType.image,
            state_label="fault_light_on",
            description="Red fault lamp active around breaker B4.",
            caption="Electrical panel with red fault light illuminated near breaker B4.",
            metadata={"issue_class": "stuck_breaker"},
            tensor_shape=[3, 16, 16],
            tensor_values=[0.91] * (3 * 16 * 16),
        )
    )

    response = triage_service.analyze(
        TriageRequest(
            question="The red fault light is on near breaker B4. What should happen next?",
            expected_state_label="fault_light_on",
            observation=VisualObservation(
                observation_id="obs-linked-1",
                media_type=MediaType.image,
                tensor_shape=[3, 16, 16],
                tensor_values=[0.92] * (3 * 16 * 16),
            ),
        )
    )

    assert response.issue_candidates[0].issue_class == "stuck_breaker"
    assert response.next_steps
    assert response.next_steps[0].step == "Inspect breaker B4 for a half-open position"
    assert response.next_steps[0].citations[0].document_id == "doc-linked-sop"
