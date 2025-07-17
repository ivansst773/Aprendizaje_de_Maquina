import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ✅ Función necesaria para cargar gpr_model.pkl
def to_dense(X):
    return X.toarray()

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Dashboard TAM - Ames Housing", layout="wide")

# ===================== CARGA DE MODELOS ============================
@st.cache_resource
def cargar_modelos():
    base_path = os.path.dirname(__file__)
    modelos = {}
    try:
        modelos["Lasso"] = joblib.load(os.path.join(base_path, "lasso_model.pkl"))
        modelos["Random Forest"] = joblib.load(os.path.join(base_path, "random_forest_model.pkl"))
        modelos["Gaussian Process"] = joblib.load(os.path.join(base_path, "gpr_model.pkl"))
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
    return modelos

modelos = cargar_modelos()

# ===================== TÍTULO Y DESCRIPCIÓN ========================
st.title("🏠 Dashboard de Predicción y Comparación de Modelos - Ames Housing")
st.markdown("""
**Parcial 1 TAM 2025-1**  
Visualiza y compara el desempeño de los mejores modelos de regresión para el Ames Housing Dataset.
""")

# ===================== CARGA Y VISTA DEL DATASET ==================
st.header("1. Exploración del Dataset")
uploaded_file = st.file_uploader("Sube tu archivo CSV de prueba (sin columna 'SalePrice')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    st.markdown("*Vista previa de tus datos de entrada.*")
    st.write(df.describe(include='all'))
else:
    st.info("No se subió archivo. Se usará el dataset por defecto desde GitHub.")
    url_default = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
    df = pd.read_csv(url_default).drop(columns=["SalePrice"])
    st.dataframe(df.head())
    st.markdown("*Vista previa del dataset Ames Housing cargado automáticamente.*")

# ===================== ANÁLISIS EXPLORATORIO BÁSICO ===============
st.header("2. Análisis Exploratorio Rápido")
st.bar_chart(df.select_dtypes(np.number))

# ===================== VALORES REALES PARA COMPARAR ===============
st.header("3. Comparación con Valores Reales (opcional)")
real_file = st.file_uploader("Sube el archivo CSV con valores reales ('SalePrice')", type=["csv"], key="real")
real_values = None
if real_file is not None:
    real_df = pd.read_csv(real_file)
    if "SalePrice" in real_df.columns:
        real_values = real_df["SalePrice"].values
        st.success("Archivo de valores reales cargado.")
    else:
        st.error("El archivo debe contener la columna 'SalePrice'.")

# ===================== PREDICCIÓN Y COMPARACIÓN DE MODELOS ========
st.header("4. Evaluación y Comparación de Modelos")
if modelos:
    seleccion = st.multiselect(
        "Selecciona los modelos a comparar:",
        list(modelos.keys()),
        default=list(modelos.keys())
    )

    resultados = {}
    for modelo_nombre in seleccion:
        modelo = modelos[modelo_nombre]
        try:
            pred = modelo.predict(df)
            resultados[modelo_nombre] = pred
            st.subheader(f"Predicciones con {modelo_nombre}")
            st.write(pred[:5])
        except Exception as e:
            st.error(f"Error al predecir con {modelo_nombre}: {e}")

    if real_values is not None:
        st.subheader("Métricas de Desempeño (MAE, MSE, R2, MAPE)")
        metricas = []
        for modelo_nombre, pred in resultados.items():
            mae = mean_absolute_error(real_values, pred)
            mse = mean_squared_error(real_values, pred)
            r2 = r2_score(real_values, pred)
            mape = np.mean(np.abs((real_values - pred) / real_values)) * 100
            metricas.append({
                "Modelo": modelo_nombre,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
                "MAPE (%)": mape
            })
        met_df = pd.DataFrame(metricas)
        st.dataframe(met_df)

        # 📊 Visualización comparativa de métricas
        st.subheader("Visualización Comparativa de Métricas")
        metricas_plot = met_df.set_index("Modelo")[["MAE", "MSE", "R2", "MAPE (%)"]]
        st.bar_chart(metricas_plot)

        # 🔍 Dispersión Real vs Predicho
        st.subheader("Dispersión: Valores Reales vs Predichos")
        for modelo_nombre, pred in resultados.items():
            fig, ax = plt.subplots()
            ax.scatter(real_values, pred, alpha=0.5)
            ax.plot([real_values.min(), real_values.max()], [real_values.min(), real_values.max()], 'r--')
            ax.set_xlabel("SalePrice Real")
            ax.set_ylabel("SalePrice Predicho")
            ax.set_title(f"{modelo_nombre}")
            st.pyplot(fig)
    else:
        st.info("Sube el archivo con valores reales para obtener métricas de desempeño.")

    # Descarga de resultados
    if resultados:
        for modelo_nombre, pred in resultados.items():
            df[f"{modelo_nombre}_Predicted"] = pred
        st.download_button(
            label="Descargar archivo con predicciones",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_comparadas.csv",
            mime="text/csv"
        )

# ===================== COMENTARIOS Y AYUDA ========================
st.markdown("""--- Reemplazo de app.py con nueva versión funcional de Streamlit ---""")
