"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from construction_vl_jepa.api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestSystemEndpoints:
    def test_system_health(self, client):
        response = client.get("/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_list_patterns(self, client):
        response = client.get("/system/patterns")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_trigger_retrain(self, client):
        response = client.post("/system/training/retrain")
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"


class TestMachineEndpoints:
    def test_list_machines_empty(self, client):
        response = client.get("/machines")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_register_and_get_machine(self, client):
        machine = {
            "machine_id": "CNC-042",
            "machine_type": "CNC Mill",
            "location": "Building A",
        }
        # Register
        response = client.post("/machines", json=machine)
        assert response.status_code == 201
        assert response.json()["machine_id"] == "CNC-042"

        # Get
        response = client.get("/machines/CNC-042")
        assert response.status_code == 200
        assert response.json()["machine_type"] == "CNC Mill"

    def test_get_nonexistent_machine(self, client):
        response = client.get("/machines/DOES-NOT-EXIST")
        assert response.status_code == 404

    def test_machine_health(self, client):
        # Register first
        client.post("/machines", json={
            "machine_id": "CNC-099",
            "machine_type": "Lathe",
        })
        response = client.get("/machines/CNC-099/health")
        assert response.status_code == 200
        assert response.json()["status"] in ("green", "yellow", "red", "unknown")

    def test_submit_feedback(self, client):
        client.post("/machines", json={
            "machine_id": "CNC-001",
            "machine_type": "Test",
        })
        response = client.post("/machines/CNC-001/feedback", json={
            "anomaly_id": "a-001",
            "is_true_positive": True,
            "notes": "Confirmed bearing wear",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
