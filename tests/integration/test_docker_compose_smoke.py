from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest
import torch
from sentence_transformers import SentenceTransformer

from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_smoke_assets(model_dir: Path) -> None:
    text_dir = model_dir / "text-encoder"
    text_dir.mkdir(parents=True, exist_ok=True)
    projector_path = model_dir / "projector.pt"
    policy_path = model_dir / "triage-policy.json"

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    model.save(str(text_dir))

    projector = VisualTextProjector(192, 384, 384)
    torch.save(projector.state_dict(), projector_path)
    CalibratedTriagePolicy.bootstrap().save(policy_path)


@pytest.mark.integration
def test_docker_compose_smoke(tmp_path) -> None:
    docker = shutil.which("docker")
    if docker is None or os.getenv("RUN_DOCKER_SMOKE") != "1":
        pytest.skip("Set RUN_DOCKER_SMOKE=1 with docker available to run the compose smoke test")

    repo_root = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "models"
    _prepare_smoke_assets(model_dir)
    api_port = _free_port()
    project_name = f"mtc-smoke-{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env["COMPOSE_PROJECT_NAME"] = project_name
    env["MTC_MODEL_DIR"] = str(model_dir)
    env["MTC_API_PORT"] = str(api_port)

    up_cmd = [
        docker,
        "compose",
        "-f",
        "docker-compose.yml",
        "up",
        "-d",
        "--build",
    ]
    down_cmd = [
        docker,
        "compose",
        "-f",
        "docker-compose.yml",
        "down",
        "-v",
    ]

    try:
        subprocess.run(up_cmd, cwd=repo_root, env=env, check=True)
        base_url = f"http://127.0.0.1:{api_port}"
        deadline = time.time() + 180
        last_error = "service did not become healthy"
        while time.time() < deadline:
            try:
                response = httpx.get(f"{base_url}/system/health", timeout=5.0)
                if response.status_code == 200:
                    break
                last_error = f"unexpected status {response.status_code}"
            except Exception as exc:  # pragma: no cover - network timing dependent
                last_error = str(exc)
            time.sleep(2)
        else:  # pragma: no cover - exercised only when the stack fails
            raise AssertionError(last_error)

        health = response.json()
        assert health["components"]["projector"]["checkpoint_loaded"] is True
        assert health["components"]["triage_policy"]["checkpoint_loaded"] is True
        assert health["components"]["metadata_store"]["metadata_store"] == "sqlalchemy"
        assert health["components"]["vector_index"]["required"] == "true"
        assert health["components"]["vector_index"]["mode"] == "qdrant+memory"

        doc_response = httpx.post(
            f"{base_url}/corpus/documents",
            json={
                "document_id": "doc-smoke",
                "source_type": "manual",
                "title": "Smoke Manual",
                "body": "Inspect breaker B4. Reset relay. Verify lamp state.",
                "tags": ["stuck_breaker"],
            },
            timeout=10.0,
        )
        assert doc_response.status_code == 200

        incident_response = httpx.post(
            f"{base_url}/corpus/incidents",
            json={
                "incident_id": "inc-smoke",
                "title": "Fault light on B4",
                "summary": "Breaker B4 stuck with red fault lamp.",
                "issue_class": "stuck_breaker",
                "fix_summary": "Reset breaker and inspect contact block.",
            },
            timeout=10.0,
        )
        assert incident_response.status_code == 200

        state_response = httpx.post(
            f"{base_url}/reference-states",
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
            f"{base_url}/triage/analyze",
            json={
                "question": "What is likely causing this panel fault?",
                "expected_state_label": "fault_light_on",
                "observation": {
                    "observation_id": "obs-smoke",
                    "media_type": "image",
                    "tensor_shape": [3, 16, 16],
                    "tensor_values": [0.88] * (3 * 16 * 16),
                },
            },
            timeout=10.0,
        )
        assert triage_response.status_code == 200
        triage_payload = triage_response.json()
        assert triage_payload["issue_candidates"]
        assert triage_payload["state_assessment"]["matched_state_label"] == "fault_light_on"
    finally:
        subprocess.run(down_cmd, cwd=repo_root, env=env, check=False)
