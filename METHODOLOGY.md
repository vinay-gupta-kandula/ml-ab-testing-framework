# 📑 A/B Testing Methodology

## 🎯 Goal
Evaluate which ML model (A: Logistic Regression or B: Random Forest)
provides better churn predictions under real traffic distribution.

## 🧪 Hypothesis
- **Null H0**: There is no difference between Model A and Model B.
- **Alternative H1**: One model performs significantly better.

## 📊 Metrics Collected
| Metric | Why |
|--------|-----|
| Prediction value | Proxy for model output |
| Request volume | Ensures even A/B traffic |
| Latency per request | Measures speed differences |

## 🔀 Traffic Routing
Incoming requests are randomly split:
- Model A → 50%
- Model B → 50%

This simulates real-world inference load balancing.

## 📈 Statistical Test
We apply a **two-sample t-test (unequal variance)** on request latency:

- If **p-value < 0.05** → statistically significant
- If **p-value >= 0.05** → no meaningful difference

## 🧠 Decision Criteria
- If Model B consistently shows lower latency & better prediction behavior → declare B winner
- If results inconclusive → continue collecting samples

## 🗄 Data Logging
Every prediction request logs:
- Model variant
- Timestamp
- Input features
- Prediction value
- Latency

Stored inside `data/ab_test_logs.db`.

## ✔ Final Notes
This framework can be extended with:
- More variants (A/B/C)
- Real ground-truth tracking
- Advanced significance tests
