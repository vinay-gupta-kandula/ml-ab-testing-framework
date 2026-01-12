from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid
import random
import joblib
import sqlite3
import pandas as pd
import os

app = FastAPI()

# Load models
model_a = joblib.load("api/models/model_A.pkl")
model_b = joblib.load("api/models/model_B.pkl")

# Load feature column names
feature_cols = joblib.load("api/models/feature_cols.pkl")

models = {"A": model_a, "B": model_b}

# Use SAME DB location inside and outside Docker
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT,
            model_variant TEXT,
            input_features TEXT,
            prediction REAL,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Request schema
class PredictionRequest(BaseModel):
    features: list[float]   # MUST be 46 values in correct order!

@app.post("/predict")
async def predict(request: PredictionRequest):
    start = time.time()
    request_id = str(uuid.uuid4())

    # Ensure correct feature count
    if len(request.features) != len(feature_cols):
        return {"error": f"Expected {len(feature_cols)} features, got {len(request.features)}"}

    # Convert to DataFrame with correct columns
    df_input = pd.DataFrame([request.features], columns=feature_cols)

    # Random traffic split
    variant = "A" if random.random() < 0.5 else "B"
    model = models[variant]

    prediction = float(model.predict_proba(df_input)[0][1])
    latency_ms = (time.time() - start) * 1000

    # Log prediction
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions VALUES (?, datetime('now'), ?, ?, ?, ?)",
        (request_id, variant, str(request.features), prediction, latency_ms)
    )
    conn.commit()
    conn.close()

    return {
        "request_id": request_id,
        "model_variant": variant,
        "prediction": prediction,
        "latency_ms": latency_ms
    }
