from fastapi.testclient import TestClient
from mlops_project.api import app

client = TestClient(app)


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
