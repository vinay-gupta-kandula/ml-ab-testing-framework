import pytest
from fastapi.testclient import TestClient
from api.main import app, DB_PATH
import sqlite3
import os
import time

client = TestClient(app)

# Helper to generate correct feature count
def get_fake_features():
    return [0.0] * 46  # Must match the 46 features expected by your models

def test_predict_user_stickiness():
    """
    Critical Test: Validates deterministic traffic splitting.
    The same user_id must always be routed to the same model variant.
    """
    user_id = "test_user_789"
    payload = {"user_id": user_id, "features": get_fake_features()}
    
    # First request
    resp1 = client.post("/predict", json=payload)
    assert resp1.status_code == 200
    variant1 = resp1.json()["model_variant"]
    
    # Second request
    resp2 = client.post("/predict", json=payload)
    assert resp2.status_code == 200
    variant2 = resp2.json()["model_variant"]
    
    # Assert stickiness
    assert variant1 == variant2, f"Stickiness failed: User {user_id} assigned to both {variant1} and {variant2}"

def test_traffic_distribution():
    """
    Validates that the hashing algorithm distributes traffic across both models.
    """
    variants = set()
    for i in range(20):  # Test multiple unique IDs to see both variants
        payload = {"user_id": f"unique_user_{i}", "features": get_fake_features()}
        resp = client.post("/predict", json=payload)
        variants.add(resp.json()["model_variant"])
    
    assert "A" in variants and "B" in variants, "Traffic splitting failed to use both model variants."

def test_asynchronous_logging():
    """
    Validates that predictions are correctly persisted to the database.
    """
    user_id = "db_test_user"
    payload = {"user_id": user_id, "features": get_fake_features()}
    
    resp = client.post("/predict", json=payload)
    request_id = resp.json()["request_id"]
    
    # Give the BackgroundTask a moment to write to the DB
    time.sleep(0.5)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT model_variant FROM predictions WHERE request_id = ?", (request_id,))
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None, "Prediction was not found in the database (Logging failed)."
    assert row[0] == resp.json()["model_variant"]

def test_invalid_input_handling():
    """
    Validates that the API returns a 400 error for incorrect feature counts.
    """
    # Test with 3 features instead of 46
    payload = {"user_id": "bad_request_user", "features": [1.0, 2.0, 3.0]}
    resp = client.post("/predict", json=payload)
    
    # Your updated main.py now raises an HTTPException(400)
    assert resp.status_code == 400
    assert "detail" in resp.json()