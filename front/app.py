import os

import requests
import streamlit as st

st.title("NeuralZOO")
st.write("Front opérationnel")

api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")

try:
    response = requests.get(f"{api_base_url}/health", timeout=5)
    data = response.json()
    st.success(f"API status : {data.get('status', 'unknown')}")
except Exception as e:
    st.error(f"API inaccessible : {e}")
