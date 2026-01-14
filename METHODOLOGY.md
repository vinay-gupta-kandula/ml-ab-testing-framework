### 📌 **FINAL METHODOLOGY.md**



```markdown
# 📑 A/B Testing Methodology

## 🎯 Goal
The objective of this experiment is to evaluate two machine learning model variants for customer churn prediction and determine which is more suitable for production deployment.  
The framework measures both prediction behavior and operational performance under real inference traffic.

---

## 🧪 Experimental Design & Hypothesis

### Model Groups
| Group | Model Type | Purpose |
|------|------------|----------|
| **Model A** | Logistic Regression | Baseline / Control |
| **Model B** | Random Forest | Experimental Variant (Treatment) |

### Hypotheses
Let **μA** and **μB** represent the mean latency and prediction values of models A and B respectively.

* **Null Hypothesis (H₀):** μA = μB — No statistically significant difference between model behaviors.
* **Alternative Hypothesis (H₁):** μA ≠ μB — A significant difference exists in latency or prediction behavior.

**Significance level:** α = 0.05

---

## 📊 Metrics Collected

| Metric | Category | Description |
|--------|----------|-------------|
| **Mean Prediction** | Business Proxy | Average churn probability predicted per model variant. |
| **Mean Latency (ms)** | Operational | Time taken for inference from request receive to prediction return. |
| **Request Volume** | Integrity | Total requests per model ensuring correct A/B split. |

All values are logged asynchronously using FastAPI BackgroundTasks to maintain low latency.

---

## 🔀 Traffic Routing & User Stickiness

This framework uses **deterministic cohort assignment** rather than random sampling:

* Each incoming request includes a **user_id**
* The value is hashed using **MD5**
* The hash result is mapped via modulo to either:
  * **Model A**, or
  * **Model B**
* **User Stickiness:** The same user always receives predictions from the same model across requests
* Over many users, the routing naturally approaches a **50/50 balanced split**

This strategy mirrors experimentation platforms used in production (Netflix, Amazon, DoorDash), ensuring fairness and integrity.

---

## 📈 Statistical Rigor

Collected results are analyzed using **Welch’s T-Test**, which is the appropriate choice because:

✔ It does not assume equal variances between group distributions  
✔ It supports unequal sample sizes (real-world A/B traffic is rarely perfectly balanced)  
✔ It outputs a **p-value** testing significance of observed differences  

### Decision Rule
* If **p-value < 0.05**, we reject the null hypothesis (performance differs meaningfully)
* If **p-value ≥ 0.05**, differences are consistent with random variation

---

## 🏆 Winning Model Criteria

A model is considered superior when both conditions are satisfied:

1. **Statistical Significance** achieved (p < 0.05), AND  
2. Model shows improvement in at least one key metric:
   * Lower latency
   * Higher predicted churn probability (if aligned with business signal)
   * More stable score distribution

If conditions are not met, the experiment is classified **Inconclusive**, and additional traffic should be collected.

---

## 🗄 Data Persistence & Integrity

Every request logged includes:

* Request ID (UUID)
* Timestamp
* Determined Model Variant (A/B)
* Raw Feature Vector
* Prediction Output (probability)
* Latency (ms)

Logs persist to a mounted SQLite database:


/app/data/ab_test_logs.db



This ensures:
* Data survives Docker restarts  
* Reproducible offline analysis  
* Supports automated dashboards and statistical processing

---

## 🧩 Conclusion

This A/B testing framework provides:
✔ Deterministic sticky routing  
✔ Concurrent model serving  
✔ Persistent high-quality logging  
✔ Statistical inference for business decision making  
✔ Real-time monitoring via Streamlit visualizations  



