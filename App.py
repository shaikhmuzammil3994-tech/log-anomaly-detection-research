import streamlit as st

st.title("Log Anomaly Detection")

st.write("Enter a log message to check anomaly score")

log = st.text_input("Log Input")

if st.button("Predict"):
    st.success("Anomaly Score: 0.91")
    st.warning("This is a demo model output")
