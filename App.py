import streamlit as st
import numpy as np
import torch

from src.model import LogAnomalyModel
from src.utils import extract_structural_features

# Load model (for demo purposes, untrained instance)
model = LogAnomalyModel()
model.eval()

st.set_page_config(page_title="Log Anomaly Detection", layout="centered")

st.title("🧠 Log Anomaly Detection System")
st.write("Hybrid CNN + Transformer model with structural feature engineering")

# Input log text
log_input = st.text_area("Enter log message:")

if st.button("Detect Anomaly"):

    if log_input.strip() == "":
        st.warning("Please enter a log message.")
    else:
        # Dummy embedding (replace with real embedding in future)
        x = torch.randn(1, 10, 128)

        # Structural features
        features = extract_structural_features(log_input)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            score = model(x, features).item()

        st.subheader("Result")

        if score > 0.5:
            st.error(f"⚠️ Anomaly Detected (Score: {score:.3f})")
        else:
            st.success(f"✅ Normal Log (Score: {score:.3f})")

st.caption("Powered by CNN + Transformer Hybrid Model")
