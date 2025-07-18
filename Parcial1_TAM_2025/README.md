# Parcial 1 - Teoría de Aprendizaje de Máquina

## Estructura del repositorio
- `Parcial1_TAM_2025/Parte_Teorica/`: Contiene el PDF con las soluciones teóricas.
- `Parcial1_TAM_2025/Parte_Practica/`: Incluye el notebook `Punto_2_Modelos_Regresion.ipynb` con la implementación.
- `Parcial1_TAM_2025/dashboard_streamlit/`: Dashboard interactivo con los modelos.

## Ejecución local
```bash
# Instalar dependencias
pip install -r requirements.txt

# Lanzar el dashboard
streamlit run Parcial1_TAM_2025/dashboard_streamlit/app.py
```

## Visualización en línea

👉 [Abrir Dashboard TAM - Ames Housing](https://aprendizajedemaquina-xkdhztxgv56qmmty6sdtfy.streamlit.app/)


## Datos utilizados

- **Dataset:** [Ames Housing Dataset en Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

Este conjunto de datos contiene información detallada sobre viviendas en Ames, Iowa, con más de 80 variables que describen características físicas, ubicación, calidad de construcción, y más.  
Fue diseñado como reemplazo del clásico Boston Housing Dataset para tareas de regresión.

- **Tamaño:** ~2,900 registros
- **Variables destacadas:** `OverallQual`, `GrLivArea`, `GarageCars`, `YearBuilt`, `Neighborhood`, `SalePrice`
- **Objetivo:** Predecir el precio de venta (`SalePrice`) a partir de las demás características

Este dataset es ideal para aplicar técnicas de regresión, ingeniería de características, y evaluación de modelos.

📌 Conclusiones y Recomendaciones
Este proyecto permitió aplicar técnicas de regresión supervisada sobre el dataset Ames Housing, evaluando el desempeño de tres modelos distintos:

Lasso Regression: útil para selección de variables, aunque puede subestimar precios altos.

Random Forest: mostró buen desempeño general, robusto ante ruido y no linealidades.

Gaussian Process Regression: ofrece predicciones suaves, pero es más costoso computacionalmente.

Recomendaciones para mejorar el modelo:
Realizar validación cruzada para obtener métricas más robustas.

Aplicar ingeniería de características sobre variables categóricas como Neighborhood.

Probar modelos adicionales como XGBoost o LightGBM.

Implementar una sección de interpretabilidad (SHAP, Permutation Importance) en el dashboard.
