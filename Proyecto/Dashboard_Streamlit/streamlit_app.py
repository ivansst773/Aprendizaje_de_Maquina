import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ————————————————————————————————
# 0) Ruta absoluta base del script
# ————————————————————————————————
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "modelos")

# ————————————————————————————————
# 1) Cargar los modelos
# ————————————————————————————————
model_paths = {
    "Random Forest":       os.path.join(MODEL_DIR, "modelo_rf.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "modelo_log.pkl"),
    "MLPClassifier":       os.path.join(MODEL_DIR, "modelo_mlp.pkl"),
}

@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error al cargar {os.path.basename(path)}: {e}")
        return None

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        modelo = load_model(path)
        if modelo:
            models[name] = modelo
    else:
        st.error(f"❌ No se encontró el fichero: {path}")

if not models:
    st.stop()

# ————————————————————————————————
# 2) Interfaz principal con pestañas
# ————————————————————————————————
st.title("🩺 Clasificador PERG — Diagnóstico Automático")
tab1, tab2 = st.tabs(["📋 Explicación del Proyecto", "🔍 Predicción en Tiempo Real"])

# ————————————————————————————————
# 3) Pestaña Explicativa
# ————————————————————————————————
with tab1:
    st.subheader("🎯 Objetivo del Proyecto")
    st.markdown("""
    Automatizar el diagnóstico oftalmológico a partir de señales PERG y metadatos clínicos, comparando tres modelos de clasificación supervisada:
    `Random Forest`, `Regresión Logística Multiclase` y `MLPClassifier`.
    """)

    st.subheader("🏗️ Pipeline Técnico")
    st.markdown("""
    - Preprocesamiento con `StandardScaler`
    - División en `train/test` y normalización
    - Entrenamiento de modelos con scikit-learn
    - Evaluación con Accuracy, F1-macro, matriz de confusión
    - Exportación de artefactos `.pkl` para despliegue
    """)

    st.subheader("📊 Resultados Comparativos")
    st.image("modelos/tabla_comparacion_modelos.png", caption="Comparación de métricas", use_column_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("modelos/confusion_matrix_random_forest.png", caption="Random Forest", use_column_width=True)
    with col2:
        st.image("modelos/matriz_logistica.png", caption="Regresión Logística", use_column_width=True)
    with col3:
        st.image("modelos/matriz_mlp.png", caption="MLPClassifier", use_column_width=True)

    st.subheader("🤖 Análisis e Interpretación")
    st.markdown("""
    - **Random Forest** logró el mejor balance multiclase.
    - **MLPClassifier** respondió bien a patrones no lineales.
    - **Regresión Logística** ofrece trazabilidad y buena base interpretativa.
    - Todos los modelos fueron serializados con `pickle` para despliegue web.
    """)

    st.subheader("📦 Artefactos reproducibles")
    st.markdown("""
    Los modelos entrenados y escaladores fueron guardados en `.pkl` y versionados en GitHub.
    La app actual los carga en tiempo de ejecución desde la carpeta `modelos/`.
    """)

# ————————————————————————————————
# 4) Pestaña Interactiva para Predicción
# ————————————————————————————————
with tab2:
    st.sidebar.header("⚙️ Parámetros del Paciente")
    model_name = st.sidebar.selectbox("📌 Modelo a usar", list(models.keys()))
    model = models[model_name]

    RE_1      = st.sidebar.number_input("RE_1",      value=0.0, format="%.4f")
    LE_1      = st.sidebar.number_input("LE_1",      value=0.0, format="%.4f")
    RE_2      = st.sidebar.number_input("RE_2",      value=0.0, format="%.4f")
    LE_2      = st.sidebar.number_input("LE_2",      value=0.0, format="%.4f")
    RE_3      = st.sidebar.number_input("RE_3",      value=0.0, format="%.4f")
    LE_3      = st.sidebar.number_input("LE_3",      value=0.0, format="%.4f")
    age_years = st.sidebar.number_input("Edad (años)", min_value=0, max_value=120, value=30)
    sex       = st.sidebar.selectbox("Sexo", ["Male", "Female"])
    sex_code  = 1 if sex == "Female" else 0

    if st.sidebar.button("🔮 Predecir diagnóstico"):
        X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])
        pred_code = model.predict(X_new)[0]
        st.success(f"✅ Diagnóstico con **{model_name}**: Clase **{pred_code}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            df_proba = pd.DataFrame(
                proba.reshape(1, -1),
                columns=[f"Clase {i}" for i in range(len(proba))]).T
            df_proba.columns = ["Probabilidad"]
            st.subheader("📊 Distribución de Probabilidades")
            st.dataframe(df_proba, width=300)
