import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ————————————————————————————————
# 1. Ruta base y modelos
# ————————————————————————————————
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "modelos")

model_paths = {
    "Random Forest":       os.path.join(MODEL_DIR, "modelo_rf.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "modelo_log.pkl"),
    "MLPClassifier":       os.path.join(MODEL_DIR, "modelo_mlp.pkl"),
}

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = load_model(path)
    else:
        st.warning(f"❗ Fichero no encontrado: {path}")

# ————————————————————————————————
# 2. Título y descripción
# ————————————————————————————————
st.title("🩺 Clasificador PERG — Comparación Multimodelo")
st.markdown("Introduce los parámetros del paciente para ver cómo cada modelo clasifica su diagnóstico, con probabilidades visualizadas.")

# ————————————————————————————————
# 3. Entrada de paciente
# ————————————————————————————————
with st.sidebar:
    st.header("⚙️ Parámetros del Paciente")

    RE_1      = st.number_input("RE_1",      value=0.0, format="%.4f")
    LE_1      = st.number_input("LE_1",      value=0.0, format="%.4f")
    RE_2      = st.number_input("RE_2",      value=0.0, format="%.4f")
    LE_2      = st.number_input("LE_2",      value=0.0, format="%.4f")
    RE_3      = st.number_input("RE_3",      value=0.0, format="%.4f")
    LE_3      = st.number_input("LE_3",      value=0.0, format="%.4f")
    age_years = st.number_input("Edad (años)", min_value=0, max_value=120, value=30)
    sex       = st.selectbox("Sexo", ["Male", "Female"])
    sex_code  = 1 if sex == "Female" else 0

    if st.button("🔍 Comparar Modelos"):
        X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])

        # Tabla de resultados
        comparacion = []
        for name, model in models.items():
            pred = model.predict(X_new)[0]
            fila = {"Modelo": name, "Clase Predicha": pred}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                for i, p in enumerate(proba):
                    fila[f"Clase {i} (%)"] = round(p * 100, 2)
            comparacion.append(fila)

        df_comp = pd.DataFrame(comparacion)
        st.subheader("📋 Predicción de Cada Modelo")
        st.dataframe(df_comp)

        # Gráfica comparativa
        st.subheader("📈 Probabilidades por Modelo")
        fig, ax = plt.subplots(figsize=(8, 4))
        for fila in comparacion:
            nombre = fila["Modelo"]
            probs = [fila.get(f"Clase {i} (%)", 0) for i in range(len(proba))]
            ax.plot(range(len(probs)), probs, marker='o', label=nombre)

        ax.set_xlabel("Clases")
        ax.set_ylabel("Probabilidad (%)")
        ax.set_title("Distribución de Probabilidades")
        ax.legend()
        st.pyplot(fig)
