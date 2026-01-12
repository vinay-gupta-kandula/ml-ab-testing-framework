from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import hashlib
import joblib
import sqlite3
import pandas as pd
import time
import uuid
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
ml_models = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")


def init_db():
    """Guarantee DB + table exists before API or tests run."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
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
        logger.info("DB READY")
    except Exception as e:
        logger.error(f"DB init error: {e}")


def load_models():
    """Load ML models once before serving any requests."""
    try:
        model_dir = os.path.join(BASE_DIR, "models")
        ml_models["A"] = joblib.load(os.path.join(model_dir, "model_A.pkl"))
        ml_models["B"] = joblib.load(os.path.join(model_dir, "model_B.pkl"))
        ml_models["feature_cols"] = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        logger.info("MODELS LOADED")
    except Exception as e:
        logger.error(f"Model load error: {e}")


# 💥 Load DB + Models IMMEDIATELY at import time
init_db()
load_models()

app = FastAPI()


class PredictionRequest(BaseModel):
    user_id: str
    features: list[float]


def log_to_db(request_id, variant, features, prediction, latency):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions VALUES (?, datetime('now'), ?, ?, ?, ?)",
            (request_id, variant, str(features), prediction, latency)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Logging error: {e}")


@app.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    start = time.time()
    request_id = str(uuid.uuid4())

    # If models failed somehow, re-load
    if not ml_models:
        init_db()
        load_models()

    # Validate length
    if len(request.features) != len(ml_models["feature_cols"]):
        raise HTTPException(status_code=400, detail="Bad feature count")

    try:
        # Deterministic model choice
        hash_val = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
        variant = "A" if hash_val % 2 == 0 else "B"
        model = ml_models[variant]

        df = pd.DataFrame([request.features], columns=ml_models["feature_cols"])
        prediction = float(model.predict_proba(df)[0][1])
        latency_ms = (time.time() - start) * 1000

        background_tasks.add_task(log_to_db, request_id, variant, request.features, prediction, latency_ms)

        return {
            "request_id": request_id,
            "model_variant": variant,
            "prediction": prediction,
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
