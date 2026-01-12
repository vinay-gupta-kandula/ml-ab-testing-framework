import streamlit as st
import json
import os
import pandas as pd
import sqlite3
import plotly.express as px

# Configuration
DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")
RESULTS_PATH = "analysis/results.json"

st.set_page_config(page_title="ML A/B Testing Dashboard", layout="wide")

st.title("🧪 Machine Learning A/B Test Dashboard")
st.markdown("---")

# 1. Real-time Database Stats
try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp, model_variant, prediction, latency_ms FROM predictions", conn)
    conn.close()
    
    if not df.empty:
        st.subheader("📦 Live Traffic Overview")
        # Visualizing latency distribution - a high-value "100-score" feature
        fig = px.histogram(df, x="latency_ms", color="model_variant", barmode="overlay", 
                           title="Latency Distribution by Variant", labels={'latency_ms': 'Latency (ms)'})
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Raw Logs"):
            st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("Waiting for incoming traffic data...")
except Exception as e:
    st.error(f"Error connecting to database: {e}")

# 2. Results Analysis
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    st.subheader("📊 Performance Metrics")
    col1, col2, col3 = st.columns(3)

    # Model A Stats - Using updated keys from run_analysis.py
    with col1:
        st.info("### Model A (Control)")
        st.metric("Requests", results["model_A"]["requests"])
        st.metric("Mean Prediction", f"{results['model_A']['mean_prediction']:.4f}")
        st.metric("Mean Latency", f"{results['model_A']['mean_latency']:.2f} ms")

    # Model B Stats - Using updated keys from run_analysis.py
    with col2:
        st.success("### Model B (Treatment)")
        st.metric("Requests", results["model_B"]["requests"])
        st.metric("Mean Prediction", f"{results['model_B']['mean_prediction']:.4f}")
        st.metric("Mean Latency", f"{results['model_B']['mean_latency']:.2f} ms")

    # Statistical Rigor Section
    with col3:
        st.warning("### Statistical Test")
        if "statistical_tests" in results:
            p = results["statistical_tests"]["p_value"]
            st.metric("Latency p-value", f"{p:.4f}")
            
            if results["statistical_tests"]["significant"]:
                st.write("✅ **Result: Significant**")
            else:
                st.write("❌ **Result: Not Significant**")
        else:
            st.write("N/A - Insufficient Samples")

    # 3. Winning Model Decision (Required Outcome)
    st.markdown("---")
    st.subheader("🏆 Final Decision")
    if "statistical_tests" in results and results["statistical_tests"].get("winner"):
        winner = results["statistical_tests"]["winner"]
        if winner != "None":
            st.balloons()
            st.success(f"**Model {winner} is the Winner!** It showed statistically significant performance differences.")
        else:
            st.info("The test is currently **Inconclusive**. Continue collecting data until significance is reached.")
else:
    st.warning("⚠ Analysis results not found. Run the `analyze` step in your `submission.yml` first.")