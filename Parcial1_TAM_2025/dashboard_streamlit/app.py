import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Configurar página en Streamlit
st.set_page_config(page_title="Dashboard TAM", layout="wide")

# ================================
# 🏗️ Cargar modelos entrenados
# ================================
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    model_paths = {
        "Lasso": os.path.join(base_path, "lasso_model.pkl"),
        "Random Forest": os.path.join(base_path, "random_forest_model.pkl"),
        "Gaussian Process": os.path.join(base_path, "gpr_model.pkl")
    }

    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)  # Cargar pipeline completo
        except FileNotFoundError:
            st.error(f"Error: No se encontró el archivo {path}.")
        except Exception as e:
            st.error(f"Error al cargar {name}: {str(e)}")
    
    return models

st.title("Comparación de Modelos Predictivos")

# ================================
# 📊 Definir características esperadas por el modelo
# ================================
expected_features = [
    'Bedroom AbvGr', 'Land Contour', 'Sale Type', 'Garage Qual', 'Open Porch SF', 'Yr Sold',
    'BsmtFin Type 1', 'Heating', 'Garage Type', 'Enclosed Porch', 'Misc Val', 'Overall Qual',
    'Roof Matl', 'Bsmt Half Bath', '3Ssn Porch', 'Sale Condition', 'Bsmt Full Bath', '2nd Flr SF',
    'Exter Cond', 'Mas Vnr Area', 'MS Zoning', 'Bldg Type', 'Mo Sold', 'Paved Drive', 'Exterior 2nd',
    'Overall Cond', 'Central Air', 'Exterior 1st', 'Low Qual Fin SF', 'Garage Cars', 'BsmtFin Type 2',
    'Screen Porch', 'Bsmt Exposure', 'Kitchen Qual', 'Lot Area', 'Condition 2', 'Garage Finish',
    'Order', 'Garage Yr Blt', 'Street', 'Exter Qual', 'Functional', 'MS SubClass', 'Lot Frontage',
    'Full Bath', 'Condition 1', 'Electrical', '1st Flr SF', 'BsmtFin SF 1', 'Pool Area', 'Year Built',
    'Heating QC', 'Foundation', 'TotRms AbvGrd', 'Lot Config', 'Half Bath', 'Gr Liv Area', 'Garage Area',
    'Bsmt Unf SF', 'Garage Cond', 'Kitchen AbvGr', 'Bsmt Cond', 'Roof Style', 'BsmtFin SF 2', 'Lot Shape',
    'Year Remod/Add', 'Bsmt Qual', 'Total Bsmt SF', 'Wood Deck SF', 'Land Slope', 'Neighborhood',
    'House Style', 'Utilities', 'PID', 'Fireplaces'
]

# ================================
# 🏠 Capturar entrada del usuario
# ================================
user_input = {}
for feature in expected_features:
    user_input[feature] = st.text_input(f"Ingrese {feature}", "0")

# Convertir a DataFrame
X_input = pd.DataFrame([user_input])

# ================================
# 🚀 Cargar modelos
# ================================
models = load_models()

# ================================
# 🎯 Aplicar preprocesamiento antes de la predicción
# ================================
if st.button("Predecir") and models:
    try:
        model_name = st.selectbox("Modelo:", list(models.keys()))
        model = models.get(model_name)

        if model:
            X_processed = model.named_steps["preprocess"].transform(X_input)  # Aplicar preprocesamiento
            prediction = model.named_steps["model"].predict(X_processed)[0]  # Hacer predicción
            st.success(f"📈 Precio estimado: ${prediction:,.2f} USD")
            
            # ================================
            # 📊 Gráfico de distribución de precios
            # ================================
            real_prices = [200000, 220000, 250000, 270000, 300000]  # Casas reales
            fig, ax = plt.subplots()
            ax.hist(real_prices, bins=5, color='skyblue', alpha=0.7, label="Precios reales")
            ax.axvline(prediction, color="red", linestyle="dashed", linewidth=2, label="Predicción")
            ax.set_title("Comparación de Precios")
            ax.set_xlabel("Precio en USD")
            ax.set_ylabel("Frecuencia")
            ax.legend()
            st.pyplot(fig)

            # ================================
            # 🔥 Optimización del modelo con GridSearchCV
            # ================================
            if model_name == "Random Forest":  # Solo optimizamos Random Forest
                param_grid_rf = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5],
                }
                grid_rf = GridSearchCV(model, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_rf.fit(X_processed, prediction)  # Ajustar modelo
                best_params = grid_rf.best_params_

                # Mostrar resultados de optimización
                st.write("🔧 **Optimización del modelo:**")
                st.write(f"Mejor número de árboles (n_estimators): {best_params['model__n_estimators']}")
                st.write(f"Mejor profundidad máxima (max_depth): {best_params['model__max_depth']}")
                st.write(f"Mejor mínimo de división de nodos (min_samples_split): {best_params['model__min_samples_split']}")

        else:
            st.error("Error: No se encontró el modelo seleccionado.")

    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")