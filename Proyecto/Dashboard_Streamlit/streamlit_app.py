import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# 1. Carga de modelos
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
        st.warning(f"⚠️ No se encontró: {path}")

# 2. Interfaz principal
st.title("🧠 Clasificador Educativo PERG")
tab1, tab2, tab3 = st.tabs([
    "📘 Explicación de Modelos",
    "🩺 Diagnóstico del Paciente",
    "📊 Comparación de Resultados"
])

# 3. Explicación de modelos
with tab1:
    st.subheader("🎓 ¿Qué hacen estos modelos?")
    st.markdown("""
    **🔍 Regresión Logística**: modelo lineal que estima probabilidades usando funciones logísticas. Ideal para interpretar coeficientes.

    **🌲 Random Forest**: conjunto de árboles de decisión que votan en grupo. Captura relaciones complejas y es robusto a ruido.

    **🧠 MLPClassifier**: red neuronal multicapa que detecta patrones no lineales en los datos. Aprende representaciones internas.

    Todos fueron entrenados sobre señales PERG + edad + sexo. Cada uno tiene ventajas distintas según el tipo de clase que se desea detectar.
    """)

# 4. Diagnóstico personalizado
with tab2:
    st.subheader("🧪 Ingresa parámetros clínicos")
    RE_1 = st.number_input("RE_1", value=0.0)
    LE_1 = st.number_input("LE_1", value=0.0)
    RE_2 = st.number_input("RE_2", value=0.0)
    LE_2 = st.number_input("LE_2", value=0.0)
    RE_3 = st.number_input("RE_3", value=0.0)
    LE_3 = st.number_input("LE_3", value=0.0)
    age  = st.number_input("Edad", min_value=0, max_value=120, value=30)
    sex  = st.selectbox("Sexo", ["Male", "Female"])
    sex_code = 1 if sex == "Female" else 0

    X_new = np.array([[RE_1, LE_1, RE_2, LE_2, RE_3, LE_3, age, sex_code]])

    if st.button("📤 Predecir con todos los modelos"):
        comparacion = []
        for nombre, modelo in models.items():
            clase = modelo.predict(X_new)[0]
            fila = {"Modelo": nombre, "Clase predicha": clase}
            if hasattr(modelo, "predict_proba"):
                probs = modelo.predict_proba(X_new)[0]
                for i, p in enumerate(probs):
                    fila[f"Clase {i} (%)"] = round(p * 100, 2)
            comparacion.append(fila)

        st.success("✅ Resultados de los 3 modelos")
        df_comp = pd.DataFrame(comparacion)
        st.dataframe(df_comp)

        st.session_state["comparacion"] = comparacion
        st.session_state["X_new"] = X_new

# 5. Comparación visual
with tab3:
    st.subheader("📈 Gráfica de probabilidades por modelo")
    if "comparacion" in st.session_state:
        fig, ax = plt.subplots(figsize=(8,5))
        for fila in st.session_state["comparacion"]:
            modelo = fila["Modelo"]
            y = [fila.get(f"Clase {i} (%)", 0) for i in range(len(fila) - 2)]
            x = [f"Clase {i}" for i in range(len(y))]
            ax.plot(x, y, marker="o", label=modelo)

        ax.set_ylabel("Probabilidad (%)")
        ax.set_title("Distribución de probabilidades")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Primero genera una predicción en la pestaña anterior.")

tab4 = st.tab("🧾 Interpretación técnica de resultados")

with tab4:
    st.subheader("🧠 Análisis comparativo entre modelos")
    if "comparacion" in st.session_state and "X_new" in st.session_state:
        for fila in st.session_state["comparacion"]:
            st.markdown(f"### 🔬 Modelo: {fila['Modelo']}")
            st.markdown(f"**Clase predicha:** {fila['Clase predicha']}")
            st.markdown("**Probabilidades por clase:**")
            probs = [v for k, v in fila.items() if "Clase" in k and "(%)" in k]
            clases = [k for k in fila.keys() if "Clase" in k and "(%)" in k]

            df_probs = pd.DataFrame({
                "Clase": clases,
                "Probabilidad (%)": probs
            })
            st.dataframe(df_probs)

            interpretacion = ""
            max_prob = max(probs)
            if max_prob < 50:
                interpretacion = "🔎 El modelo muestra incertidumbre elevada (ninguna clase supera 50%). Puede ser útil revisar el preprocesamiento o entrenar con más datos."
            elif max_prob >= 90:
                interpretacion = "✅ Predicción muy segura. El modelo asigna alta probabilidad a una clase específica."
            else:
                interpretacion = "🤔 La predicción muestra cierto sesgo, pero no es completamente concluyente. Puede usarse como apoyo clínico."

            st.info(interpretacion)
    else:
        st.warning("Genera primero una predicción para ver la interpretación.")

