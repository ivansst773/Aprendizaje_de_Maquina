import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# 1) Función para cargar modelos en caché
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# 2) Rutas de los 3 modelos en tu carpeta `modelos/`
MODEL_DIR = "modelos"
model_paths = {
    "Random Forest": os.path.join(MODEL_DIR, "modelo_rf.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "modelo_log.pkl"),
    "MLPClassifier": os.path.join(MODEL_DIR, "modelo_mlp.pkl"),
}

# 3) Carga todos los modelos
models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.error(f"❌ No se encontró {path}")

# 4) Interfaz Streamlit
st.title("🩺 Clasificador PERG: Diagnóstico Oftalmológico")

st.sidebar.header("Selecciona modelo y parámetros")

# 5) Selector de modelo
model_name = st.sidebar.selectbox("Modelo", list(models.keys()))
model = models.get(model_name)

# 6) Inputs del paciente
RE_1       = st.sidebar.number_input("RE_1", value=0.0)
LE_1       = st.sidebar.number_input("LE_1", value=0.0)
RE_2       = st.sidebar.number_input("RE_2", value=0.0)
LE_2       = st.sidebar.number_input("LE_2", value=0.0)
RE_3       = st.sidebar.number_input("RE_3", value=0.0)
LE_3       = st.sidebar.number_input("LE_3", value=0.0)
age_years  = st.sidebar.number_input("Edad (años)", min_value=0, max_value=120, value=30)
sex        = st.sidebar.selectbox("Sexo", ["Male", "Female"])
sex_code   = 1 if sex == "Female" else 0

# 7) Botón de predicción
if st.button("🔍 Predecir diagnóstico"):
    X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])
    pred_code = model.predict(X_new)[0]
    st.success(f"Diagnóstico predicho con {model_name}: Clase **{pred_code}**")

    # Mostrar probabilidades si dispones de predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        df_proba = pd.DataFrame(
            proba.reshape(1, -1),
            columns=[f"Clase {i}" for i in range(proba.shape[0])]
        ).T
        df_proba.columns = ["Probabilidad"]
        st.dataframe(df_proba, width=200)
