import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# ————————————————————————————————
# 1. Carga de Modelos
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
        st.warning(f"⚠️ No se encontró el modelo: {path}")

# ————————————————————————————————
# 2. Interfaz Principal con Pestañas
# ————————————————————————————————
st.title("🧠 App Educativa — Clasificación de Señales PERG")
tab1, tab2, tab3 = st.tabs(["📚 Explicación de Modelos", "🔍 Simulación Diagnóstico", "📊 Comparación Visual"])

# ————————————————————————————————
# 3. Pestaña Explicativa
# ————————————————————————————————
with tab1:
    st.header("📘 ¿Qué modelos estás usando?")
    st.markdown("""
    Esta app utiliza tres clasificadores entrenados sobre señales PERG y metadatos clínicos:
    
    #### 🎄 Random Forest
    - Basado en árboles de decisión que votan en conjunto
    - Resiste sobreajuste y se adapta a patrones complejos

    #### 📈 Regresión Logística Multiclase
    - Modelo lineal con alta interpretabilidad
    - Usa funciones logísticas para estimar probabilidades

    #### 🧠 MLPClassifier
    - Red neuronal multicapa
    - Detecta relaciones no lineales entre variables

    Cada modelo fue entrenado con `StandardScaler`, validado con `Accuracy`, `F1-macro` y serializado con `pickle`.
    """)

# ————————————————————————————————
# 4. Pestaña Interactiva
# ————————————————————————————————
with tab2:
    st.header("🩺 Diagnóstico del Paciente")

    RE_1      = st.number_input("RE_1",      value=0.0, format="%.4f")
    LE_1      = st.number_input("LE_1",      value=0.0, format="%.4f")
    RE_2      = st.number_input("RE_2",      value=0.0, format="%.4f")
    LE_2      = st.number_input("LE_2",      value=0.0, format="%.4f")
    RE_3      = st.number_input("RE_3",      value=0.0, format="%.4f")
    LE_3      = st.number_input("LE_3",      value=0.0, format="%.4f")
    age_years = st.number_input("Edad",      min_value=0, max_value=120, value=30)
    sex       = st.selectbox("Sexo", ["Male", "Female"])
    sex_code  = 1 if sex == "Female" else 0

    X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age_years, sex_code]])

    st.markdown("Haz click abajo para predecir con cada modelo:")
    if st.button("🚀 Generar Diagnóstico"):
        comparacion = []
        for name, model in models.items():
            pred = model.predict(X_new)[0]
            result = {"Modelo": name, "Predicción": pred}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                for i, p in enumerate(proba):
                    result[f"Clase {i} (%)"] = round(p*100,2)
            comparacion.append(result)

        df_comp = pd.DataFrame(comparacion)
        st.success("✅ Comparación de Modelos")
        st.dataframe(df_comp)

# ————————————————————————————————
# 5. Pestaña Visual Comparativa
# ————————————————————————————————
with tab3:
    st.header("📊 Distribución de Probabilidades por Modelo")
    if "comparacion" in locals():
        fig, ax = plt.subplots(figsize=(8,5))
        for fila in comparacion:
            modelo = fila["Modelo"]
            y = [fila.get(f"Clase {i} (%)", 0) for i in range(len(proba))]
            x = [f"Clase {i}" for i in range(len(proba))]
            ax.plot(x, y, marker='o', label=modelo)

        ax.set_ylabel("Probabilidad (%)")
        ax.set_title("Comparación Multiclase")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("⚠️ Genera primero una predicción para visualizar los resultados.")
