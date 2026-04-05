from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import boto3
import httpx
import pytest
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.assets import checkpoint_sha256, directory_sha256
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy
from maintenance_triage_copilot.vendor.meta_ijepa import VisionTransformer
from maintenance_triage_copilot.vendor.meta_vjepa import VideoVisionTransformer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_smoke_assets(model_dir: Path) -> None:
    torch.manual_seed(0)
    text_dir = model_dir / "text-encoder"
    text_dir.mkdir(parents=True, exist_ok=True)
    projector_path = model_dir / "projector.pt"
    policy_path = model_dir / "triage-policy.json"
    image_path = model_dir / "image_backbone.pt"
    video_path = model_dir / "video_backbone.pt"

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    model.save(str(text_dir))

    projector = VisualTextProjector(192, 384, 384)
    torch.save(projector.state_dict(), projector_path)
    CalibratedTriagePolicy.bootstrap().save(policy_path)
    torch.save(
        {
            "encoder": VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=192,
                depth=2,
                num_heads=3,
            ).state_dict()
        },
        image_path,
    )
    torch.save(
        {
            "target_encoder": VideoVisionTransformer(
                img_size=32,
                patch_size=8,
                num_frames=8,
                tubelet_size=2,
                embed_dim=192,
                depth=2,
                num_heads=3,
            ).state_dict()
        },
        video_path,
    )

    manifest = {
        "version": "smoke-1",
        "assets": {
            "text_encoder": {
                "kind": "directory",
                "path": "text-encoder",
                "sha256": directory_sha256(text_dir),
            },
            "projector": {
                "kind": "file",
                "path": "projector.pt",
                "sha256": checkpoint_sha256(projector_path),
            },
            "triage_policy": {
                "kind": "file",
                "path": "triage-policy.json",
                "sha256": checkpoint_sha256(policy_path),
            },
            "image_backbone": {
                "kind": "file",
                "path": "image_backbone.pt",
                "sha256": checkpoint_sha256(image_path),
                "preset": "ijepa_vith14_224",
                "embedding_dim": 192,
                "checkpoint_keys": ["target_encoder", "encoder"],
                "smoke_stub": {
                    "input_size": 32,
                    "patch_size": 8,
                    "embed_dim": 192,
                    "depth": 2,
                    "num_heads": 3,
                },
            },
            "video_backbone": {
                "kind": "file",
                "path": "video_backbone.pt",
                "sha256": checkpoint_sha256(video_path),
                "preset": "vjepa_vith16_224_2x16x16",
                "embedding_dim": 192,
                "checkpoint_keys": ["target_encoder", "encoder"],
                "smoke_stub": {
                    "input_size": 32,
                    "patch_size": 8,
                    "num_frames": 8,
                    "tubelet_size": 2,
                    "embed_dim": 192,
                    "depth": 2,
                    "num_heads": 3,
                },
            },
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _make_png_bytes(color: tuple[int, int, int] = (220, 40, 40)) -> bytes:
    image = Image.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.integration
def test_docker_compose_smoke(tmp_path) -> None:
    docker = shutil.which("docker")
    if docker is None or os.getenv("RUN_DOCKER_SMOKE") != "1":
        pytest.skip("Set RUN_DOCKER_SMOKE=1 with docker available to run the compose smoke test")

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "models"
    _prepare_smoke_assets(model_dir)
    api_port = _free_port()
    minio_port = _free_port()
    project_name = f"mtc-smoke-{uuid.uuid4().hex[:8]}"
    auth_headers = {"Authorization": "Bearer smoke-token"}
    env = os.environ.copy()
    env["COMPOSE_PROJECT_NAME"] = project_name
    env["MTC_MODEL_DIR"] = str(model_dir)
    env["MTC_API_PORT"] = str(api_port)
    env["MTC_MINIO_PORT"] = str(minio_port)
    env["MTC_ALLOW_SMOKE_ASSETS"] = "1"
    env["MTC_SERVICE_TOKENS_JSON"] = json.dumps({"compose-smoke": "smoke-token"})
    up_cmd = [
        docker,
        "compose",
        "-f",
        "tests/integration/docker-compose.smoke.yml",
        "up",
        "-d",
        "--build",
    ]
    down_cmd = [
        docker,
        "compose",
        "-f",
        "tests/integration/docker-compose.smoke.yml",
        "down",
        "-v",
    ]

    try:
        subprocess.run(up_cmd, cwd=repo_root, env=env, check=True)
        base_url = f"http://127.0.0.1:{api_port}"
        deadline = time.time() + 240
        last_error = "service did not become healthy"
        health_response: httpx.Response | None = None
        while time.time() < deadline:
            try:
                candidate = httpx.get(f"{base_url}/system/health", timeout=5.0)
                if candidate.status_code == 200:
                    health_response = candidate
                    break
                last_error = f"unexpected status {candidate.status_code}"
            except Exception as exc:  # pragma: no cover - network timing dependent
                last_error = str(exc)
            time.sleep(2)
        else:  # pragma: no cover - exercised only when the stack fails
            raise AssertionError(last_error)

        assert health_response is not None
        health = health_response.json()
        assert health["components"]["projector"]["checkpoint_loaded"] is True
        assert health["components"]["triage_policy"]["checkpoint_loaded"] is True
        assert health["components"]["metadata_store"]["metadata_store"] == "sqlalchemy"
        assert health["components"]["vector_index"]["required"] == "true"
        assert health["components"]["vector_index"]["mode"] == "qdrant+memory"
        assert health["components"]["object_store"]["object_store"] == "s3"
        assert health["components"]["manifest"]["validated"] is True
        assert health["components"]["auth"]["mode"] == "bearer"

        metrics_response = httpx.get(
            f"{base_url}/metrics",
            headers=auth_headers,
            timeout=10.0,
        )
        assert metrics_response.status_code == 200
        assert "mtc_http_requests_total" in metrics_response.text

        doc_response = httpx.post(
            f"{base_url}/corpus/upload",
            headers=auth_headers,
            files={
                "file": (
                    "smoke-manual.txt",
                    b"Inspect breaker B4. Reset relay. Verify lamp state.",
                    "text/plain",
                )
            },
            data={
                "document_id": "doc-smoke",
                "source_type": "manual",
                "equipment_family": "electrical_panel_family_a",
                "title": "Smoke Manual",
                "tags": "stuck_breaker",
            },
            timeout=20.0,
        )
        assert doc_response.status_code == 200
        document_asset_id = doc_response.json()["asset_id"]

        incident_response = httpx.post(
            f"{base_url}/corpus/incidents",
            headers=auth_headers,
            json={
                "incident_id": "inc-smoke",
                "title": "Fault light on B4",
                "summary": "Breaker B4 stuck with red fault lamp.",
                "issue_class": "stuck_breaker",
                "fix_summary": "Reset breaker and inspect contact block.",
                "linked_document_ids": ["doc-smoke"],
            },
            timeout=10.0,
        )
        assert incident_response.status_code == 200

        state_response = httpx.post(
            f"{base_url}/reference-states",
            headers=auth_headers,
            json={
                "state_id": "state-smoke",
                "media_type": "image",
                "state_label": "fault_light_on",
                "description": "Red fault light illuminated near breaker B4.",
                "caption": "Panel state with red fault light.",
                "tensor_shape": [3, 16, 16],
                "tensor_values": [0.9] * (3 * 16 * 16),
            },
            timeout=10.0,
        )
        assert state_response.status_code == 200

        triage_response = httpx.post(
            f"{base_url}/media/triage",
            headers=auth_headers,
            files={"file": ("panel.png", _make_png_bytes(), "image/png")},
            data={
                "equipment_family": "electrical_panel_family_a",
                "question": "What is likely causing this panel fault?",
                "expected_state_label": "fault_light_on",
            },
            timeout=20.0,
        )
        assert triage_response.status_code == 200
        assert triage_response.headers["X-Audit-ID"].startswith("audit-")
        triage_payload = triage_response.json()
        assert triage_payload["issue_candidates"]
        assert triage_payload["state_assessment"]["matched_state_label"] == "fault_light_on"

        list_audits = httpx.get(
            f"{base_url}/audit/triage",
            headers=auth_headers,
            timeout=10.0,
        )
        assert list_audits.status_code == 200
        assert list_audits.json()["items"]

        audit_id = triage_response.headers["X-Audit-ID"]
        audit_detail = httpx.get(
            f"{base_url}/audit/triage/{audit_id}",
            headers=auth_headers,
            timeout=10.0,
        )
        assert audit_detail.status_code == 200
        detail_payload = audit_detail.json()
        linked_assets = detail_payload["linked_assets"]
        assert linked_assets
        assert {asset["asset_id"] for asset in linked_assets}
        assert detail_payload["audit"]["principal"] == "compose-smoke"

        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://127.0.0.1:{minio_port}",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            region_name="us-east-1",
        )
        objects = s3.list_objects_v2(Bucket="mtc-assets")
        keys = {item["Key"] for item in objects.get("Contents", [])}
        assert any(key.startswith("triage_upload/") for key in keys)
        assert any(key.startswith("corpus_upload/") for key in keys)
        assert document_asset_id
    finally:
        subprocess.run(down_cmd, cwd=repo_root, env=env, check=False)
