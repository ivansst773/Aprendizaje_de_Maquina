import pandas as pd
import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title="Dashboard TAM", layout="wide")

@st.cache_resource
def load_models():
    return {
        "Lasso": joblib.load("Parcial1_TAM_2025/dashboard_streamlit/lasso_model.pkl"),
        "Random Forest": joblib.load("Parcial1_TAM_2025/dashboard_streamlit/random_forest_model.pkl"),
        "Gaussian Process": joblib.load("Parcial1_TAM_2025/dashboard_streamlit/gpr_model.pkl")
    }

st.title("Comparación de Modelos Predictivos")

# Widgets de entrada
feature1 = st.number_input("Área construida (m²)", min_value=50, value=120)
feature2 = st.number_input("Habitaciones", min_value=1, value=3)

if st.button("Predecir"):
    try:
        model = load_models()[st.selectbox("Modelo:", list(load_models().keys()))]
        prediction = model.predict([[feature1, feature2]])[0]
        st.success(f"Precio estimado: ${prediction:,.2f} USD")
    except Exception as e:
        st.error(f"Error: {str(e)}")
