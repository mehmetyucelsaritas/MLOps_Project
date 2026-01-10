import os
import pytest
from fastapi.testclient import TestClient
from mlops_project.api import app

client = TestClient(app)

pytestmark = pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip model-dependent API tests in CI")


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
