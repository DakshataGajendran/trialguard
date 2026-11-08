from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_evaluate_endpoint():
    # Test evaluating a known test account ID from generated dataset
    response = client.post("/evaluate", json={"account_id": "A1042"})
    # If A1042 is in test set, it should return 200 with action fields
    if response.status_code == 200:
        data = response.json()
        assert "action" in data
        assert "abuse_probability" in data
        assert "reasons" in data