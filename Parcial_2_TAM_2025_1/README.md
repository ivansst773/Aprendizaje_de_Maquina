
# 📊 Parcial 2 TAM 2025-1 – Universidad Nacional de Colombia

Este repositorio contiene la entrega completa del Parcial 2 del curso **Teoría de Aprendizaje de Máquina**, correspondiente al semestre 2025-1, bajo la dirección del profesor **Andrés Marino Álvarez Meza, Ph.D.**

---

## 🧭 Estructura del proyecto

```plaintext
Parcial_2_TAM_2025_1/
├── Parte_Teorica/
│   └── Solucion_Teorica.pdf
├── Parte_Practica/
│   └── Parcial_2_TAM_2025_1.ipynb
├── dashboard_streamlit/
│   ├── app.py
│   ├── modelo_lr.pkl
│   ├── modelo_rf.pkl
│   ├── modelo_dl.h5
├── video_explicativo.mp4 (opcional)
└── README.md
📚 Componentes de la solución
a) Modelos y formulación matemática
Formulación teórica y problema de optimización de 11 modelos de clasificación:

PCA, UMAP, Naive Bayes, SGD, Logistic Regression, LDA, KNN, SVC, Random Forest, Gaussian Process, Deep Learning

Presentado en PDF adjunto (Parte_Teorica/)

b) Proyección de datos USPS
Proyecciones con PCA y UMAP en 2D y 3D

Visualización con etiquetas y superposición de imágenes

Análisis del parámetro n_neighbors de UMAP

c) Clasificación supervisada
Modelos utilizados:

modelo_lr.pkl → Logistic Regression

modelo_rf.pkl → Random Forest

modelo_dl.h5 → CNN con Keras

Evaluación con Accuracy, F1-score macro, Precision

Curvas ROC para la clase 0

d) Dashboard interactivo
📎 Accede aquí 👉 Streamlit Cloud – USPS Dashboard

Visualización de proyecciones PCA/UMAP

Comparador de clasificadores con pestañas

Métricas y curvas ROC en tiempo real

e) Video explicativo
📺 YouTube (link de presentación): https://youtube.com/tu_video

Explicación del dashboard

Resumen de puntos clave del parcial

Introducción a atención y modelos Transformer

⚙️ Requisitos de ejecución
pip install -r requirements.txt

Dependencias principales:

streamlit

scikit-learn

umap-learn

matplotlib

pandas

tensorflow

joblib

📤 Entrega
✅ Enviado vía GitHub y correo electrónico amalvarezme@unal.edu.co

📆 Fecha límite: 17 de julio de 2025

📁 Adjuntos: PDF teórico, notebook, dashboard, video

👨‍💻 Autor
IAN [Tu nombre completo aquí] Curso TAM – Universidad Nacional de Colombia – sede Manizales


---

## 🧩 2. README del dashboard (ubicación: `/dashboard_streamlit/README.md`)

Este archivo explica cómo usar el dashboard y los modelos cargados.

```markdown
# 🧠 USPS Dashboard – Parcial 2 TAM 2025

Dashboard interactivo desarrollado en Streamlit para el Parcial 2 TAM. Permite visualizar proyecciones PCA y UMAP, y comparar tres clasificadores supervisados sobre el dataset USPS.

---

## ⚙️ Ejecución local

```bash
streamlit run app.py

📦 Archivos incluidos
Archivo	Descripción
app.py	Código principal del dashboard
modelo_lr.pkl	Modelo Logistic Regression entrenado
modelo_rf.pkl	Modelo Random Forest entrenado
modelo_dl.h5	Modelo CNN entrenado en Keras
🧪 Funcionalidades del dashboard
Visualización de proyecciones PCA y UMAP (configurables)

Comparación de clasificadores con métricas y curvas ROC

Navegación intuitiva mediante pestañas

🧠 Modelos utilizados
sklearn.linear_model.LogisticRegression

sklearn.ensemble.RandomForestClassifier

tensorflow.keras.Sequential (CNN con capas Dense, Dropout)

Todos los modelos fueron entrenados en Colab y exportados como .pkl y .h5.

🔎 Dataset utilizado
USPS – fetch_openml(name='USPS', version=1)

Preprocesado y proyectado en espacios de menor dimensión

📌 Requisitos

pip install streamlit scikit-learn umap-learn matplotlib pandas tensorflow joblib

🚀 Despliegue automático (Streamlit Cloud)
Disponible en: 🔗 https://share.streamlit.io/ivansst773/Aprendizaje_de_Maquina/...

👨‍💻 Autor
Edgar Ivan Calpa 


