from fastapi.testclient import TestClient
from api.main import app
import random

client = TestClient(app)

def test_predict_route_success():
    fake_features = [random.random() for _ in range(46)]
    resp = client.post("/predict", json={"features": fake_features})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_variant"] in ["A", "B"]
    assert "prediction" in data

def test_predict_wrong_feature_count():
    resp = client.post("/predict", json={"features": [1,2,3]})
    assert "error" in resp.json()
