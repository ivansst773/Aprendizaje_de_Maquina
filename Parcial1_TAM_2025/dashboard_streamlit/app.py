import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dashboard TAM", layout="wide")

# -----------------------
# Cargar modelos
# -----------------------
@st.cache_resource
def load_models():
    lasso = joblib.load("lasso_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    gpr = joblib.load("gpr_model.pkl")
    return {
        "Lasso": lasso,
        "Random Forest": rf,
        "Gaussian Process": gpr
    }

models = load_models()

# -----------------------
# Título
# -----------------------
st.title("🏠 Dashboard de Predicción - Ames Housing")
st.markdown("Sube un archivo CSV con las mismas columnas que el dataset original (excepto la columna 'SalePrice').")

# -----------------------
# Subida de archivo CSV
# -----------------------
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Vista previa del dataset:")
    st.dataframe(input_df.head())

    # Selección de modelo
    selected_model_name = st.selectbox("Selecciona el modelo para predecir:", list(models.keys()))
    selected_model = models[selected_model_name]

    # Botón para predecir
    if st.button("Predecir precios"):
        try:
            y_pred = selected_model.predict(input_df)
            input_df["Predicted_Price"] = y_pred

            st.subheader("🔮 Predicciones:")
            st.dataframe(input_df[["Predicted_Price"]].head())

            st.download_button(
                label="📥 Descargar resultados",
                data=input_df.to_csv(index=False).encode("utf-8"),
                file_name="predicciones_ames.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error al predecir: {e}")

else:
    st.info("Por favor, sube un archivo CSV con los datos de entrada.")
