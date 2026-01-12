# 🚀 ML A/B Testing Framework

This project implements a **production-grade A/B testing system** for machine learning models, similar to what real tech companies deploy.
It supports deterministic routing, sticky user assignment, persistent logging, automated statistics, and a live dashboard.

---

## 🎯 Key Features (Submission-Ready)

* 🧠 Two ML variants: Logistic Regression (Control) & Random Forest (Treatment)
* 🔀 **Deterministic traffic split using MD5 hashing** of `user_id`
* ♻️ Sticky assignment — same user always gets same model
* ⚡ FastAPI serving with background DB logging
* 🗄️ SQLite persistence mounted to disk
* 📈 Welch’s T-test statistical comparison
* 📺 Streamlit dashboard to monitor results
* 🧪 Automated pytest suite already passing
* 🐳 Fully containerized with Docker & Compose
* 🎛 submission.yml for automated build → deploy → test → analyze

---

## 📁 Project Structure

```
ml-ab-testing-framework/
│
├── api/
│   ├── main.py                  # FastAPI service + hashing + logging
│   ├── models/
│   │   ├── model_A.pkl
│   │   ├── model_B.pkl
│   │   └── feature_cols.pkl
│   └── __init__.py
│
├── analysis/
│   ├── run_analysis.py          # Welch’s T-test + JSON results
│   ├── dashboard.py             # Streamlit visual dashboard
│   └── __init__.py
│
├── data/                        # SQLite DB persists here
│   └── telco/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── tests/
│   ├── test_api.py              # Verifies model routing & DB logging
│   └── __init__.py
│
├── train_models.py              # Train + save models & feature columns
├── Dockerfile
├── docker-compose.yml
├── README.md
├── METHODOLOGY.md
├── submission.yml
└── video.txt
```

---

## 🧠 STEP 1 — Train Models (Run Locally Once)

```bash
python train_models.py
```

Outputs:

* `model_A.pkl`
* `model_B.pkl`
* `feature_cols.pkl`

---

## 🚢 STEP 2 — Build & Run via Docker

### Build images

```bash
docker-compose build
```

### Start stack (API + DB volume)

```bash
docker-compose up -d
```

Verify running:

```bash
docker ps
```

---

## 🏁 STEP 3 — Make Deterministic API Requests

Swagger docs:

```
http://localhost:8000/docs
```

### Required JSON (46 feature values + user_id):

```json
{
  "user_id": "customer_99",
  "features": [0,1,29.85,29.85,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1]
}
```

➡️ Same `user_id` → always same model
➡️ Different user_ids → balanced 50/50 split

All requests log to:

```
./data/ab_test_logs.db
```

---

## 📊 STEP 4 — Statistical Analysis

```bash
docker-compose exec api python analysis/run_analysis.py
```

Output includes:

* Request counts by model
* Mean prediction comparison
* Welch’s T-test p-value
* Winner indication

JSON results stored in:

```
analysis/results.json
```

Inspect:

```bash
docker-compose exec api cat analysis/results.json
```

---

## 📺 STEP 5 — View Streamlit Dashboard

Run UI:

```bash
docker-compose exec api streamlit run analysis/dashboard.py --server.address=0.0.0.0 --server.port=8501
```

Open in browser:

```
http://localhost:8501
```

Dashboard shows:
✔ Raw logs
✔ A/B request volume
✔ Mean predicted churn per model
✔ p-value significance

---

## 🧪 STEP 6 — Run Automated Tests

```bash
docker-compose exec api pytest -q
```

Expected:

```
..
2 passed
```

---

## 🛑 STEP 7 — Shut Down

```bash
docker-compose down
```

SQLite logs are preserved in:

```
./data/ab_test_logs.db
```

---

## ⚡ OPTIONAL — Run Without Docker

Install deps:

```bash
pip install -r api/requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```

Run analysis:

```bash
python analysis/run_analysis.py
```

Dashboard:

```bash
streamlit run analysis/dashboard.py
```

---

## 🌟 Core Tech Stack

| Component                           | Role                            |
| ----------------------------------- | ------------------------------- |
| FastAPI                             | Model serving, hashing, logging |
| Logistic Regression / Random Forest | Competing ML variants           |
| SQLite                              | Persistent A/B logging          |
| SciPy                               | Welch’s T-test                  |
| Streamlit                           | Live experiment dashboard       |
| Docker + Compose                    | Runtime consistency             |
| Pytest                              | Automated verification          |


