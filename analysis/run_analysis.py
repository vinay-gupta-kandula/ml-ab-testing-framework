import sqlite3
import pandas as pd
from scipy import stats
import json
import os

# Use same database path API writes to
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")

def run_analysis():
    # Load data
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        print("\nNo data found in database. Send some /predict requests first!\n")
        return

    print("\n=== A/B Test Results ===\n")

    # A/B counts
    df_A = df[df["model_variant"] == "A"]
    df_B = df[df["model_variant"] == "B"]

    count_A = len(df_A)
    count_B = len(df_B)

    print(f"Model A requests: {count_A}")
    print(f"Model B requests: {count_B}")

    # Stats only if both have data
    mean_pred_A = df_A["prediction"].mean() if count_A else 0
    mean_pred_B = df_B["prediction"].mean() if count_B else 0

    print(f"\nAvg prediction A: {mean_pred_A:.4f}")
    print(f"Avg prediction B: {mean_pred_B:.4f}")

    # Latency t-test only if >1 sample each
    if count_A > 1 and count_B > 1:
        t_stat, p_value = stats.ttest_ind(df_A["latency_ms"], df_B["latency_ms"], equal_var=False)
        print(f"\nLatency t-test p-value: {p_value:.4f}")
    else:
        p_value = None
        print("\nNot enough samples to compute t-test.")

    # Save results
    results = {
        "model_A": {"requests": count_A, "mean_pred": float(mean_pred_A)},
        "model_B": {"requests": count_B, "mean_pred": float(mean_pred_B)},
        "p_value_latency": float(p_value) if p_value is not None else None
    }

    # Save under analysis folder
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to analysis/results.json")

if __name__ == "__main__":
    run_analysis()
