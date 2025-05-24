import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Simular Streamlit si no está disponible (para pruebas en Colab)
try:
    import streamlit as st
except ModuleNotFoundError:
    class StreamlitMock:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = StreamlitMock()
    print("⚠️ Streamlit no está instalado. Modo de prueba activado.")

# Configuración
if hasattr(st, 'set_page_config'):
    st.set_page_config(page_title="Dashboard TAM", layout="wide")

# Cargar modelos
@st.cache_resource
def load_models():
    modelos = {
        "Lasso": joblib.load("lasso_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Gaussian Process": joblib.load("gpr_model.pkl")
    }
    return modelos

# Interfaz
if hasattr(st, 'title'):
    st.title("Comparación de Modelos")
    model_choice = st.selectbox("Modelo:", list(load_models().keys()))
    if st.button("Predecir"):
        st.write("Predicción simulada (instala Streamlit para funcionalidad completa)")
