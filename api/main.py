from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import time
import uuid
import hashlib
import joblib
import sqlite3
import pandas as pd
import os
import logging
from contextlib import asynccontextmanager

# Set up logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for pre-loading heavy ML models and DB setup."""
    try:
        # Load models once at startup to save latency per request
        ml_models["A"] = joblib.load("api/models/model_A.pkl")
        ml_models["B"] = joblib.load("api/models/model_B.pkl")
        ml_models["feature_cols"] = joblib.load("api/models/feature_cols.pkl")
        init_db()
        logger.info("Models and Database initialized successfully.")
    except Exception as e:
        logger.error(f"Critical Startup Error: {e}")
        # In a real production environment, you might want to prevent startup here
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Persistence Path configured via environment for Docker volume support
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")

def init_db():
    """Initializes the SQLite schema if it doesn't exist."""
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

class PredictionRequest(BaseModel):
    user_id: str = Field(..., example="user_123", description="Required for deterministic A/B splitting")
    features: list[float] = Field(..., description="List of numerical features for the model")

def log_to_db(request_id: str, variant: str, features: list, prediction: float, latency: float):
    """
    Writes prediction data to SQLite. 
    Uses check_same_thread=False for safety in FastAPI's multi-threaded background environment.
    """
    try:
        # check_same_thread=False prevents threading errors in background tasks
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions VALUES (?, datetime('now'), ?, ?, ?, ?)",
            (request_id, variant, str(features), prediction, latency)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Async Logging Error: {e}")

@app.post("/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    start = time.time()
    request_id = str(uuid.uuid4())

    # 1. Strict Input Validation (Fails fast if feature count is wrong)
    if "feature_cols" not in ml_models:
        raise HTTPException(status_code=503, detail="Models not loaded correctly.")
        
    if len(request.features) != len(ml_models["feature_cols"]):
        raise HTTPException(
            status_code=400, 
            detail=f"Feature mismatch. Expected {len(ml_models['feature_cols'])}, got {len(request.features)}"
        )

    try:
        # 2. Deterministic Traffic Splitting (User Stickiness)
        # Hash ensures the same user_id always sees the same model variant.
        hash_val = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
        variant = "A" if hash_val % 2 == 0 else "B"
        model = ml_models[variant]

        # 3. Model Inference
        df_input = pd.DataFrame([request.features], columns=ml_models["feature_cols"])
        prediction = float(model.predict_proba(df_input)[0][1])
        
        latency_ms = (time.time() - start) * 1000

        # 4. Non-blocking Background Logging
        # This keeps API response times fast by not waiting for DB I/O.
        background_tasks.add_task(
            log_to_db, request_id, variant, request.features, prediction, latency_ms
        )

        return {
            "request_id": request_id,
            "model_variant": variant,
            "prediction": prediction,
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        logger.error(f"Inference Error for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during model inference.")