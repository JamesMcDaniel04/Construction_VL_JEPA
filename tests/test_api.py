from __future__ import annotations

import importlib.util


def test_system_health(client) -> None:
    response = client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["components"]["metadata_store"]["metadata_store"] == "sqlalchemy"


def test_ingest_and_triage_image(client) -> None:
    doc_response = client.post(
        "/corpus/documents",
        json={
            "document_id": "doc-1",
            "source_type": "manual",
            "title": "Panel Fault Manual",
            "body": "Inspect breaker B4. Reset the overload relay. Verify fuse F7.",
            "tags": ["stuck_breaker"],
        },
    )
    assert doc_response.status_code == 200

    incident_response = client.post(
        "/corpus/incidents",
        json={
            "incident_id": "inc-1",
            "title": "Red fault light on breaker B4",
            "summary": "Breaker B4 was stuck and the red light remained active.",
            "issue_class": "stuck_breaker",
            "fix_summary": "Reset breaker B4 and replace the damaged contact block.",
        },
    )
    assert incident_response.status_code == 200

    state_response = client.post(
        "/reference-states",
        json={
            "state_id": "state-1",
            "media_type": "image",
            "state_label": "fault_light_on",
            "description": "Red fault light illuminated near breaker B4.",
            "caption": "Panel image with a red fault indicator next to breaker B4.",
            "tensor_shape": [3, 16, 16],
            "tensor_values": [0.9] * (3 * 16 * 16),
        },
    )
    assert state_response.status_code == 200

    triage_response = client.post(
        "/triage/analyze",
        json={
            "question": "What component is likely causing this red light?",
            "expected_state_label": "fault_light_on",
            "observation": {
                "observation_id": "obs-1",
                "media_type": "image",
                "tensor_shape": [3, 16, 16],
                "tensor_values": [0.88] * (3 * 16 * 16),
            },
        },
    )
    assert triage_response.status_code == 200
    data = triage_response.json()
    assert data["state_assessment"]["matched_state_label"] == "fault_light_on"
    assert data["issue_candidates"]
    assert data["next_steps"]
    assert data["similar_incidents"]


def test_video_triage_endpoint(client) -> None:
    client.post(
        "/reference-states",
        json={
            "state_id": "video-state",
            "media_type": "video",
            "state_label": "fault_light_on",
            "description": "Fault lamp active in clip.",
            "caption": "Short clip with a persistent red fault light.",
            "tensor_shape": [3, 10, 16, 16],
            "tensor_values": [0.8] * (3 * 10 * 16 * 16),
        },
    )
    response = client.post(
        "/triage/analyze",
        json={
            "question": "Does the current panel state match expected normal state?",
            "observation": {
                "observation_id": "obs-video",
                "media_type": "video",
                "tensor_shape": [3, 10, 16, 16],
                "tensor_values": [0.82] * (3 * 10 * 16 * 16),
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["state_assessment"]["matched_state_label"] == "fault_light_on"


def test_legacy_package_removed() -> None:
    assert importlib.util.find_spec("construction_vl_jepa") is None
