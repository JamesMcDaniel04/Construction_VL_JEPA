from __future__ import annotations

import importlib.util
import io

import numpy as np
from PIL import Image

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    CorpusSourceType,
    IncidentRecord,
    ReferenceState,
    VisualObservation,
)


def test_system_health(client) -> None:
    response = client.get("/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["components"]["metadata_store"]["metadata_store"] == "sqlalchemy"
    assert response.json()["components"]["projector"]["checkpoint_loaded"] is True
    assert response.json()["components"]["manifest"]["validated"] is True
    assert response.json()["components"]["auth"]["mode"] == "hybrid_bearer"
    assert "X-Request-Duration-Ms" in response.headers
    assert "X-Request-ID" in response.headers


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
    assert response.headers["X-Audit-ID"].startswith("audit-")


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


def test_metrics_endpoint_returns_prometheus_payload(client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mtc_http_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_missing_bearer_token_returns_request_context(client) -> None:
    client.headers.pop("Authorization", None)
    response = client.post(
        "/corpus/documents",
        json={
            "document_id": "doc-auth",
            "source_type": "manual",
            "title": "Needs Auth",
            "body": "Inspect breaker B4.",
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token"
    assert "X-Request-ID" in response.headers
    assert "X-Request-Duration-Ms" in response.headers


def test_triage_audit_detail_includes_linked_assets(client, monkeypatch) -> None:
    service = client.app.state.service
    service.add_reference_state(
        ReferenceState(
            state_id="audit-state",
            media_type="image",
            state_label="fault_light_on",
            description="Red fault light illuminated near breaker B4.",
            caption="Panel image with a red fault indicator next to breaker B4.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.9] * (3 * 16 * 16),
        )
    )
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.2, dtype=np.float32),
    )

    triage_response = client.post(
        "/media/triage",
        files={"file": ("panel.png", b"fake-image", "image/png")},
        data={
            "equipment_family": "electrical_panel_family_a",
            "question": "What component is likely causing this?",
        },
    )
    assert triage_response.status_code == 200
    audit_id = triage_response.headers["X-Audit-ID"]

    list_response = client.get("/audit/triage")
    assert list_response.status_code == 200
    assert any(item["audit_id"] == audit_id for item in list_response.json()["items"])

    detail_response = client.get(f"/audit/triage/{audit_id}")
    assert detail_response.status_code == 200
    payload = detail_response.json()
    assert payload["audit"]["audit_id"] == audit_id
    assert payload["audit"]["principal"] == "test-client"
    assert len(payload["linked_assets"]) == 1
    assert payload["linked_assets"][0]["asset_type"] == "triage_upload"
    assert payload["linked_assets"][0]["object_uri"].startswith("memory://triage_upload/")
    assert payload["linked_assets"][0]["presigned_url"].startswith("memory://triage_upload/")


def test_auth_me_returns_pilot_identity(technician_client, admin_client) -> None:
    technician_response = technician_client.get("/auth/me")
    assert technician_response.status_code == 200
    assert technician_response.json()["principal_type"] == "human"
    assert technician_response.json()["role"] == "technician"
    assert technician_response.json()["organization_id"] == "org-1"
    assert technician_response.json()["email"] == "alex@example.com"

    admin_response = admin_client.get("/auth/me")
    assert admin_response.status_code == 200
    assert admin_response.json()["role"] == "admin"
    assert admin_response.json()["display_name"] == "Sam Supervisor"


def test_admin_can_invite_persisted_pilot_user(admin_client) -> None:
    invite_response = admin_client.post(
        "/admin/pilot-users/invite",
        json={
            "organization_id": "org-1",
            "role": "technician",
            "display_name": "Jordan Newhire",
            "email": "jordan@example.com",
        },
    )
    assert invite_response.status_code == 200
    payload = invite_response.json()
    assert payload["user"]["display_name"] == "Jordan Newhire"
    assert payload["invite_status"] == "sent"

    list_response = admin_client.get("/admin/pilot-users")
    assert list_response.status_code == 200
    assert any(item["display_name"] == "Jordan Newhire" for item in list_response.json()["items"])


def test_catalog_endpoints_scope_assets_and_sites(admin_client, technician_client) -> None:
    site_response = admin_client.post(
        "/catalog/sites",
        json={
            "site_id": "site-c",
            "name": "Line C",
            "code": "C1",
        },
    )
    assert site_response.status_code == 200

    asset_response = admin_client.post(
        "/catalog/assets",
        json={
            "asset_id": "panel-88",
            "site_id": "site-c",
            "display_name": "Panel 88",
            "panel_family": "family-b",
            "equipment_family": "electrical_panel_family_b",
            "panel_id": "panel-88",
        },
    )
    assert asset_response.status_code == 200

    sites = technician_client.get("/catalog/sites")
    assert sites.status_code == 200
    assert any(item["site_id"] == "site-c" for item in sites.json()["items"])

    assets = technician_client.get("/catalog/assets", params={"site_id": "site-c"})
    assert assets.status_code == 200
    assert assets.json()["items"][0]["asset_id"] == "panel-88"


def test_case_lifecycle_for_technician(client, technician_client, monkeypatch) -> None:
    service = technician_client.app.state.service
    service.add_document(
        CorpusDocument(
            document_id="case-doc-1",
            source_type=CorpusSourceType.manual,
            title="Panel Inspection SOP",
            body=(
                "Check breaker handle alignment. Verify the overload relay and inspect "
                "the contact block."
            ),
            tags=["stuck_breaker"],
        )
    )
    service.add_incident(
        IncidentRecord(
            incident_id="case-inc-1",
            title="Breaker fault with red lamp",
            summary="Red lamp stayed on after reset.",
            issue_class="stuck_breaker",
            fix_summary="Reset breaker and replace contact block.",
        )
    )
    service.add_reference_state(
        ReferenceState(
            state_id="case-state-1",
            media_type="image",
            state_label="fault_light_on",
            description="Red fault light illuminated.",
            caption="Electrical panel with a persistent red fault light.",
            tensor_shape=[3, 16, 16],
            tensor_values=[0.8] * (3 * 16 * 16),
        )
    )

    from maintenance_triage_copilot.api.routes import cases as cases_route

    monkeypatch.setattr(cases_route, "assess_image_capture", lambda _: [])
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.8, dtype=np.float32),
    )

    create_response = technician_client.post(
        "/cases",
        json={
            "site_id": "site-a",
            "asset_id": "panel-42",
            "question": "What is likely causing this red light?",
            "expected_state_label": "fault_light_on",
        },
    )
    assert create_response.status_code == 200
    case_id = create_response.json()["case_id"]
    assert create_response.json()["panel_family"] == "family-a"
    assert create_response.json()["panel_id"] == "panel-42a"

    analyze_response = technician_client.post(
        f"/cases/{case_id}/analyze",
        files={"file": ("panel.png", _valid_png_bytes(), "image/png")},
    )
    assert analyze_response.status_code == 200
    analyzed_case = analyze_response.json()
    assert analyzed_case["analysis"]["issue_candidates"]
    assert analyzed_case["analysis"]["safety_notices"]
    assert analyzed_case["analysis"]["uncertainty_summary"]
    assert analyze_response.headers["X-Audit-ID"].startswith("audit-")

    feedback_response = technician_client.post(
        f"/cases/{case_id}/feedback",
        json={"labels": ["helpful"], "comment": "Saved me a call to the senior tech."},
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["feedback"]["labels"] == ["helpful"]

    list_response = technician_client.get("/cases")
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["case_id"] == case_id
    assert list_response.json()["items"][0]["helpful"] is True

    detail_response = technician_client.get(f"/cases/{case_id}")
    assert detail_response.status_code == 200
    assert (
        detail_response.json()["analysis"]["state_assessment"]["matched_state_label"]
        == "fault_light_on"
    )


def test_case_analyze_rejects_low_quality_capture(technician_client, monkeypatch) -> None:
    create_response = technician_client.post(
        "/cases",
        json={
            "site_id": "site-b",
            "asset_id": "panel-77",
        },
    )
    case_id = create_response.json()["case_id"]

    from maintenance_triage_copilot.api.routes import cases as cases_route

    monkeypatch.setattr(
        cases_route,
        "assess_image_capture",
        lambda _: [
            "The panel image is too dark. Move closer or increase lighting before retrying."
        ],
    )

    response = technician_client.post(
        f"/cases/{case_id}/analyze",
        files={"file": ("panel.png", _valid_png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["hints"]


def test_admin_corpus_reference_and_dashboard(admin_client, monkeypatch) -> None:
    service = admin_client.app.state.service
    monkeypatch.setattr(
        service.state.image_backbone,
        "encode_raw_image",
        lambda _: np.full((192,), 0.2, dtype=np.float32),
    )

    doc_response = admin_client.post(
        "/corpus/documents",
        json={
            "document_id": "admin-doc-1",
            "source_type": "manual",
            "title": "Panel Manual",
            "body": "Inspect breaker alignment and verify lamp state.",
        },
    )
    assert doc_response.status_code == 200

    incident_response = admin_client.post(
        "/corpus/incidents",
        json={
            "incident_id": "admin-inc-1",
            "title": "Panel alarm case",
            "summary": "Alarm lamp on and breaker misaligned.",
            "issue_class": "alarm_lamp",
            "fix_summary": "Reset lamp and reseat breaker.",
        },
    )
    assert incident_response.status_code == 200

    state_upload = admin_client.post(
        "/reference-states/upload",
        files={"file": ("state.png", _valid_png_bytes(), "image/png")},
        data={
            "state_label": "normal_state",
            "description": "Normal green-light panel state.",
            "caption": "All lights green and breakers aligned.",
        },
    )
    assert state_upload.status_code == 200

    docs_list = admin_client.get("/corpus/documents")
    assert docs_list.status_code == 200
    assert docs_list.json()["items"][0]["document_id"] == "admin-doc-1"

    incidents_list = admin_client.get("/corpus/incidents")
    assert incidents_list.status_code == 200
    assert incidents_list.json()["items"][0]["incident_id"] == "admin-inc-1"

    states_list = admin_client.get("/reference-states")
    assert states_list.status_code == 200
    assert states_list.json()["items"][0]["state_label"] == "normal_state"

    dashboard = admin_client.get("/admin/dashboard")
    assert dashboard.status_code == 200
    assert "total_cases" in dashboard.json()


def _valid_png_bytes() -> bytes:
    image = Image.new("RGB", (512, 512), color=(245, 245, 245))
    for idx in range(0, 512, 32):
        for y in range(512):
            image.putpixel((idx, y), (20, 20, 20))
            image.putpixel((min(idx + 1, 511), y), (20, 20, 20))
        for x in range(512):
            image.putpixel((x, idx), (20, 20, 20))
            image.putpixel((x, min(idx + 1, 511)), (20, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
