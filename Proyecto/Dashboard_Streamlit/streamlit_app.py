
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1) Cargar el modelo
@st.cache_resource  # guarda en caché la carga
def load_model(path="modelos/modelo_rf.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🩺 Clasificador PERG: Diagnóstico Oftalmológico")
st.markdown("Ingresa las mediciones y metadatos del paciente para predecir su diagnóstico.")

# 2) Sidebar: inputs del usuario
st.sidebar.header("Parámetros del paciente")
RE_1 = st.sidebar.number_input("RE_1", value=0.0)
LE_1 = st.sidebar.number_input("LE_1", value=0.0)
RE_2 = st.sidebar.number_input("RE_2", value=0.0)
LE_2 = st.sidebar.number_input("LE_2", value=0.0)
RE_3 = st.sidebar.number_input("RE_3", value=0.0)
LE_3 = st.sidebar.number_input("LE_3", value=0.0)
age_years = st.sidebar.number_input("Edad (años)", min_value=0, max_value=120, value=30)
sex = st.sidebar.selectbox("Sexo", ["Male", "Female"])

# 3) Preprocesamiento de inputs
#   – Codifica 'sex' si tu modelo lo requiere (por ejemplo LabelEncoder)
sex_code = 1 if sex=="Female" else 0

X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])

# 4) Botón de predicción
if st.button("🔍 Predecir diagnóstico"):
    pred_code = model.predict(X_new)[0]
    # Si tienes el LabelEncoder guardado, decodifícalo aquí
    # diagnosis = le_diag.inverse_transform([pred_code])[0]
    diagnosis = f"Clase {pred_code}"
    st.success(f"Diagnóstico predicho: **{diagnosis}**")

    # Muestra probabilidades (opcional)
    proba = model.predict_proba(X_new)[0]
    df_proba = pd.DataFrame([proba], columns=[f"Clase {i}" for i in range(len(proba))])
    st.dataframe(df_proba.T, height=300)
