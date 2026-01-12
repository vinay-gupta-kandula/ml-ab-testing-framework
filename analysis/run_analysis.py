import sqlite3
import pandas as pd
from scipy import stats
import json
import os
import numpy as np

# Ensures consistency with the API's database location
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")

def calculate_confidence_interval(data, confidence=0.95):
    """Calculates the confidence interval for the mean of a sample."""
    if len(data) < 2:
        return 0.0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return float(h)

def run_analysis():
    # 1. Robust Data Loading
    if not os.path.exists(DB_PATH):
        print(f"\n[ERROR] Database not found at {DB_PATH}. Run API and send requests first.")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        print("\n[WARNING] Database is empty. No data to analyze.")
        return

    print("\n" + "="*30)
    print("   PRODUCTION A/B ANALYSIS   ")
    print("="*30 + "\n")

    # 2. Data Segmentation
    df_A = df[df["model_variant"] == "A"]
    df_B = df[df["model_variant"] == "B"]

    stats_results = {
        "summary": {
            "total_requests": len(df),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    variants = {"A": df_A, "B": df_B}
    
    for label, data in variants.items():
        count = len(data)
        mean_pred = float(data["prediction"].mean()) if count > 0 else 0.0
        mean_lat = float(data["latency_ms"].mean()) if count > 0 else 0.0
        ci_lat = calculate_confidence_interval(data["latency_ms"]) if count > 1 else 0.0

        stats_results[f"model_{label}"] = {
            "requests": count,
            "mean_prediction": round(mean_pred, 4),
            "mean_latency": round(mean_lat, 2),
            "latency_ci": round(ci_lat, 2)
        }
        print(f"Model {label}: {count} requests | Avg Prediction: {mean_pred:.4f} | Avg Latency: {mean_lat:.2f}ms")

    # 3. Statistical Rigor: Welch's T-Test
    # We use equal_var=False because variances between models in production are rarely equal.
    if len(df_A) > 1 and len(df_B) > 1:
        t_stat, p_val = stats.ttest_ind(df_A["latency_ms"], df_B["latency_ms"], equal_var=False)
        
        is_significant = p_val < 0.05
        stats_results["statistical_tests"] = {
            "test_type": "Welch's T-Test (Latency)",
            "p_value": float(p_val),
            "significant": bool(is_significant),
            "winner": "B" if (is_significant and df_B["latency_ms"].mean() < df_A["latency_ms"].mean()) else "A" if (is_significant) else "None"
        }
        
        sig_text = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
        print(f"\nLatency Welch's T-Test: p-value = {p_val:.4f} ({sig_text})")
    else:
        stats_results["statistical_tests"] = {"error": "Insufficient data for t-test"}
        print("\n[INFO] Not enough samples for statistical significance testing.")

    # 4. Persistence for Dashboard
    os.makedirs("analysis", exist_ok=True)
    output_path = "analysis/results.json"
    with open(output_path, "w") as f:
        json.dump(stats_results, f, indent=4)

    print(f"\n[SUCCESS] Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_analysis()