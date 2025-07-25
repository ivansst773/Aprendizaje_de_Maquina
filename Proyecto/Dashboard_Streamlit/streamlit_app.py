import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ——————————————————————
# 0) Ruta absoluta base desde donde corre el script
# ——————————————————————
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "modelos")

# ——————————————————————
# Debug: mostrar carpeta de trabajo y contenido
# ——————————————————————
st.write("📁 Directorio raíz del script:", BASE_DIR)
st.write("📁 Contenido raíz:", os.listdir(BASE_DIR))
if os.path.isdir(MODEL_DIR):
    st.write("📁 Contenido de `modelos/`:", os.listdir(MODEL_DIR))
else:
    st.error("❌ La carpeta `modelos/` NO existe en el deploy")

# ——————————————————————
# 1) Configuración de rutas
# ——————————————————————
model_paths = {
    "Random Forest":       os.path.join(MODEL_DIR, "modelo_rf.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "modelo_log.pkl"),
    "MLPClassifier":       os.path.join(MODEL_DIR, "modelo_mlp.pkl"),
}

# ——————————————————————
# 2) Función para cargar un modelo con caché
# ——————————————————————
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error al cargar {os.path.basename(path)}: {e}")
        return None

# ——————————————————————
# 3) Cargar todos los modelos disponibles
# ——————————————————————
models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        modelo = load_model(path)
        if modelo:
            models[name] = modelo
    else:
        st.error(f"❌ No se encontró el fichero de modelo: {path}")

if not models:
    st.stop()

# ——————————————————————
# 4) Interfaz principal
# ——————————————————————
st.title("🩺 Clasificador PERG: Diagnóstico Oftalmológico")
st.sidebar.header("Configuración de predicción")

# 5) Selector de modelo
model_name = st.sidebar.selectbox("🔍 Elige el modelo", list(models.keys()))
model = models[model_name]

# 6) Entradas del usuario
RE_1      = st.sidebar.number_input("RE_1",      value=0.0, format="%.4f")
LE_1      = st.sidebar.number_input("LE_1",      value=0.0, format="%.4f")
RE_2      = st.sidebar.number_input("RE_2",      value=0.0, format="%.4f")
LE_2      = st.sidebar.number_input("LE_2",      value=0.0, format="%.4f")
RE_3      = st.sidebar.number_input("RE_3",      value=0.0, format="%.4f")
LE_3      = st.sidebar.number_input("LE_3",      value=0.0, format="%.4f")
age_years = st.sidebar.number_input("Edad (años)", min_value=0, max_value=120, value=30)
sex       = st.sidebar.selectbox("Sexo", ["Male", "Female"])
sex_code  = 1 if sex == "Female" else 0

# 7) Botón de predicción
if st.sidebar.button("🔮 Predecir diagnóstico"):

    # Construir vector de características
    X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])

    # Predicción
    pred_code = model.predict(X_new)[0]
    st.success(f"✅ Diagnóstico predicho con **{model_name}**: Clase **{pred_code}**")

    # Mostrar probabilidades si están disponibles
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        df_proba = pd.DataFrame(proba.reshape(1, -1),
                                columns=[f"Clase {i}" for i in range(len(proba))]).T
        df_proba.columns = ["Probabilidad"]
        st.subheader("📊 Probabilidades del modelo")
        st.dataframe(df_proba, width=250)
