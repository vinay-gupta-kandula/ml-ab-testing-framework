# 🚀 ML A/B Testing Framework

This project implements a **production-style A/B testing system** for machine learning models.  
It demonstrates how real-world companies test ML systems by routing live traffic to multiple models and collecting evidence to decide which is better.

---

## 🎯 What This Project Includes

* 🧠 Two trained ML models (Random Forest & Logistic Regression)
* ⚡ FastAPI prediction API with A/B split (50/50 traffic)
* 🗄️ SQLite logging of predictions + latency
* 📊 Offline statistical analysis (t-tests)
* 📺 Streamlit dashboard for experiment monitoring
* 📦 Docker containerization for reproducible deployment
* 🧪 PyTest suite validating API behavior

---

## 📁 Project Structure

```

ml-ab-testing-framework/
│
├── api/
│   ├── main.py               # FastAPI service + logging + A/B routing
│   ├── models/
│   │   ├── model_A.pkl       # Logistic Regression
│   │   ├── model_B.pkl       # Random Forest
│   │   └── feature_cols.pkl  # 46 feature columns
│   └── **init**.py
│
├── analysis/
│   ├── run_analysis.py       # Statistical evaluation + JSON export
│   ├── dashboard.py          # Streamlit UI for visual analytics
│   └── **init**.py
│
├── data/                     # Holds SQLite DB (persisted on disk)
│   └── telco/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── tests/
│   ├── test_api.py           # Validates API + logging
│   └── **init**.py
│
├── train_models.py           # Trains both models from dataset
├── Dockerfile
├── docker-compose.yml
├── README.md
├── METHODOLOGY.md
├── video.txt                 # contains public url to access the video 
└── submission.yml


````

---

# 🧠 STEP 1 — Train Models (Run Once)

```bash
python train_models.py
````

This loads the Telco dataset, generates features, and saves:

* `model_A.pkl`
* `model_B.pkl`
* `feature_cols.pkl`

---

# 🚢 STEP 2 — Build & Run with Docker

### Build

```bash
docker-compose build
```

### Run container (API + DB)

```bash
docker-compose up -d
```

### Verify running

```bash
docker ps
```

---

# 🏁 STEP 3 — Make Predictions Through API

Open:

```
http://localhost:8000/docs
```

Select **POST /predict**
Paste EXACTLY this JSON (46 features):

```json
{
  "features": [0,1,29.85,29.85,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1]
}
```

Each request is randomly routed:

* Model A (Logistic Regression)
* Model B (Random Forest)

Requests are logged in `/data/ab_test_logs.db`.

---

# 📊 STEP 4 — Run A/B Test Statistical Analysis

After sending several requests:

```bash
docker-compose exec api python analysis/run_analysis.py
```

This prints:

* Count of A vs B samples
* Mean prediction comparison
* Latency t-test (p-value)

Results saved to:

```
analysis/results.json
```

View it:

```bash
docker-compose exec api cat analysis/results.json
```

---

# 📺 STEP 5 — View A/B Dashboard (Streamlit)

Launch dashboard inside Docker:

```bash
docker-compose exec api streamlit run analysis/dashboard.py --server.address=0.0.0.0 --server.port=8501
```

Streamlit prints:

```
URL: http://0.0.0.0:8501
```

OPEN this in browser (not 0.0.0.0):

```
http://localhost:8501
```

OR

```
http://127.0.0.1:8501
```

Dashboard Shows:
✔ Logged requests
✔ A vs B request split
✔ Prediction averages
✔ p-value significance

---

# 🧪 STEP 6 — Run Tests

```bash
docker-compose exec api pytest -q
```

Expected output:

```
..
2 passed
```

---

# 🛑 STEP 7 — Shut Down System

```bash
docker-compose down
```

Database remains safe in:

```
./data/ab_test_logs.db
```

---

# ⚡ Local (Non-Docker) Mode

### Install dependencies

```bash
pip install -r api/requirements.txt
```

### Run API locally

```bash
uvicorn api.main:app --reload
```

Visit:

```
http://localhost:8000/docs
```

### Run analysis

```bash
python analysis/run_analysis.py
```

### Run Streamlit

```bash
streamlit run analysis/dashboard.py
```

Open:

```
http://localhost:8501
```

---

# 🌟 Tech Stack Summary

| Component                           | Role                 |
| ----------------------------------- | -------------------- |
| FastAPI                             | Serves ML inference  |
| Logistic Regression / Random Forest | Model variants       |
| SQLite                              | Persistent logging   |
| Pandas / NumPy                      | Feature handling     |
| SciPy                               | Statistical testing  |
| Streamlit                           | Experiment dashboard |
| Docker + Compose                    | Deployment           |
| Pytest                              | Automated testing    |


