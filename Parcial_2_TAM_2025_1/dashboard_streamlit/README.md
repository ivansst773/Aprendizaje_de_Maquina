
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
