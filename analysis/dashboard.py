import streamlit as st
import json
import os
import pandas as pd
import sqlite3

DB_PATH = os.getenv("DATABASE_NAME", "/app/data/ab_test_logs.db")
RESULTS_PATH = "analysis/results.json"

st.set_page_config(page_title="ML A/B Testing Dashboard", layout="wide")

st.title("🧪 Machine Learning A/B Test Dashboard")

# Load DB summary directly
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM predictions", conn)
conn.close()

st.subheader("📦 Raw Logged Requests")
st.write(df)

# Load results.json (from analysis script)
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    st.subheader("📊 Summary Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model A Requests", results["model_A"]["requests"])
        st.metric("Mean Prediction A", results["model_A"]["mean_pred"])

    with col2:
        st.metric("Model B Requests", results["model_B"]["requests"])
        st.metric("Mean Prediction B", results["model_B"]["mean_pred"])

    st.subheader("🔬 Statistical Significance")
    p = results["p_value_latency"]
    if p is not None:
        st.write(f"Latency p-value: **{p:.4f}**")
        if p < 0.05:
            st.success("🎯 Significant difference detected — one model is faster!")
        else:
            st.info("ℹ No significant difference detected.")
    else:
        st.warning("Not enough data for t-test.")
else:
    st.warning("⚠ Run analysis first to generate results.json.")
