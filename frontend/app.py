# SIS3. 23B031056 Abdimutalipov Zhaxylyk
# streamlit frontend for the wine classifier api

import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Wine Classifier", page_icon="🍷")
st.title("🍷 Wine Classifier")
st.write("Enter chemical features of a wine and the model will predict its cultivar (class 0, 1 or 2).")

# model info from api
try:
    info = requests.get(f"{API_URL}/model-info", timeout=5).json()
    st.sidebar.header("Model")
    st.sidebar.write(f"**Name:** {info['model_name']}")
    st.sidebar.write(f"**Version:** {info['model_version']}")
    st.sidebar.write(f"**Algorithm:** {info['algorithm']}")
    st.sidebar.write("**Metrics:**")
    st.sidebar.json(info["metrics"])
except Exception:
    st.sidebar.error(f"API is not available at {API_URL}")

# example wines (real rows from the dataset)
examples = {
    "class 0 example": [14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0],
    "class 1 example": [12.37, 0.94, 1.36, 10.6, 88.0, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520.0],
    "class 2 example": [13.71, 5.65, 2.45, 20.5, 95.0, 1.68, 0.61, 0.52, 1.06, 7.7, 0.64, 1.74, 740.0],
}

# feature name, min, max
features = [
    ("alcohol", 11.0, 15.0),
    ("malic_acid", 0.7, 5.8),
    ("ash", 1.3, 3.3),
    ("alcalinity_of_ash", 10.0, 30.0),
    ("magnesium", 70.0, 162.0),
    ("total_phenols", 0.9, 3.9),
    ("flavanoids", 0.3, 5.1),
    ("nonflavanoid_phenols", 0.1, 0.7),
    ("proanthocyanins", 0.4, 3.6),
    ("color_intensity", 1.2, 13.0),
    ("hue", 0.4, 1.8),
    ("od280_od315_of_diluted_wines", 1.2, 4.0),
    ("proline", 270.0, 1700.0),
]

choice = st.selectbox("Start from example", list(examples.keys()))
default_values = examples[choice]

# input form
values = {}
with st.form("wine_form"):
    col1, col2 = st.columns(2)
    for i in range(len(features)):
        name, min_v, max_v = features[i]
        col = col1 if i % 2 == 0 else col2
        values[name] = col.number_input(
            name, min_value=min_v, max_value=max_v,
            value=default_values[i], key=f"{choice}_{name}",
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=values, timeout=10)
        result = response.json()
        st.success(f"Predicted: **{result['class_name']}** (probability {result['probability']:.2%})")

        proba = pd.DataFrame(
            {"probability": list(result["probabilities"].values())},
            index=list(result["probabilities"].keys()),
        )
        st.bar_chart(proba)
        st.caption(f"model version: {result['model_version']}")
    except Exception as e:
        st.error(f"Request failed: {e}")
