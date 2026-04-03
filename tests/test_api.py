from __future__ import annotations

import importlib.util

import numpy as np

from maintenance_triage_copilot.domain.models import ReferenceState, VisualObservation


def test_system_health(client) -> None:
    response = client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["components"]["metadata_store"]["metadata_store"] == "sqlalchemy"
    assert response.json()["components"]["projector"]["checkpoint_loaded"] is False
    assert "X-Request-Duration-Ms" in response.headers


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


def test_media_encode_returns_embedding_values(client, monkeypatch) -> None:
    service = client.app.state.service
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.5, dtype=np.float32),
    )

    response = client.post(
        "/media/encode",
        files={"file": ("panel.png", b"fake-image", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["modality"] == "image"
    assert payload["embedding_dim"] == 192
    assert len(payload["embedding_values"]) == 192


def test_media_triage_uses_embedding_without_reencoding_observation(client, monkeypatch) -> None:
    service = client.app.state.service
    service.add_reference_state(
        ReferenceState(
            state_id="media-state",
            media_type="image",
            state_label="fault_light_on",
            description="Red fault light illuminated near breaker B4.",
            caption="Panel image with a red fault indicator next to breaker B4.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.9] * (3 * 16 * 16),
        )
    )

    original_encode_observation = service.state.image_backbone.encode_observation

    def guard_encode(media: object) -> np.ndarray:
        if isinstance(media, VisualObservation) and media.embedding_values is not None:
            raise AssertionError("Uploaded media observation should not be re-encoded")
        return original_encode_observation(media)

    monkeypatch.setattr(service.state.image_backbone, "encode_observation", guard_encode)
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.2, dtype=np.float32),
    )

    response = client.post(
        "/media/triage",
        files={"file": ("panel.png", b"fake-image", "image/png")},
        data={
            "equipment_family": "electrical_panel_family_a",
            "question": "What component is likely causing this?",
        },
    )
    assert response.status_code == 200
    assert response.json()["state_assessment"]["matched_state_label"] == "fault_light_on"


def test_media_encode_round_trip_works_with_triage_analyze(client, monkeypatch) -> None:
    service = client.app.state.service
    service.add_reference_state(
        ReferenceState(
            state_id="roundtrip-state",
            media_type="image",
            state_label="normal_panel_state",
            description="All indicator lamps are green.",
            caption="Normal panel state with aligned breaker handles.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.1] * (3 * 16 * 16),
        )
    )
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.1, dtype=np.float32),
    )

    encode_response = client.post(
        "/media/encode",
        files={"file": ("panel.png", b"fake-image", "image/png")},
    )
    assert encode_response.status_code == 200
    embedding_values = encode_response.json()["embedding_values"]

    original_encode_observation = service.state.image_backbone.encode_observation

    def guard_encode(media: object) -> np.ndarray:
        if isinstance(media, VisualObservation) and media.embedding_values is not None:
            raise AssertionError("Precomputed observation embedding should not be re-encoded")
        return original_encode_observation(media)

    monkeypatch.setattr(service.state.image_backbone, "encode_observation", guard_encode)

    triage_response = client.post(
        "/triage/analyze",
        json={
            "question": "Is this panel normal?",
            "observation": {
                "observation_id": "obs-roundtrip",
                "media_type": "image",
                "embedding_values": embedding_values,
            },
        },
    )
    assert triage_response.status_code == 200
    assert triage_response.json()["state_assessment"]["matched_state_label"] == "normal_panel_state"


def test_legacy_package_removed() -> None:
    assert importlib.util.find_spec("construction_vl_jepa") is None
