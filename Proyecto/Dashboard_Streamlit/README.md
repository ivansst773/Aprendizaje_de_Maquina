🧠 Clasificador Educativo PERG
Dashboard interactivo para el análisis, comparación y explicación de modelos de clasificación aplicados al diagnóstico oftalmológico basado en señales PERG (Potencial Evocado de Retina). Permite al usuario:

📘 Comprender cómo funcionan los modelos empleados

🩺 Ingresar datos clínicos personalizados

📊 Visualizar las predicciones y probabilidades por modelo

🧾 Interpretar técnicamente los resultados

🔗 Acceso a la app
Accede directamente desde Streamlit Cloud:

👉 Abrir aplicación ( https://share.streamlit.io/ivansst773/aprendizaje_de_maquina/main/Proyecto/Dashboard_Streamlit/streamlit_app.py)
aprendizajedemaquina-hgj8adafx7dgmegxqp7apj

🚀 ¿Qué modelos se usan?
La aplicación compara tres clasificadores entrenados con scikit-learn:

Modelo	Descripción
🌲 Random Forest	Varios árboles de decisión que votan en grupo. Resistente al ruido y útil para relaciones no lineales.
📈 Regresión Logística	Modelo lineal e interpretable, ideal como línea base.
🧠 MLPClassifier	Red neuronal multicapa que aprende relaciones complejas y no lineales.
Todos fueron entrenados sobre un conjunto de datos clínico con variables PERG, edad y sexo, normalizados con StandardScaler.

🧪 Interacción y funcionalidades
La app se estructura en 4 pestañas:

📘 Explicación de modelos Breve descripción técnica y funcionamiento de cada clasificador.

🩺 Diagnóstico del paciente Ingreso manual de variables PERG + edad + sexo. Predicción simultánea con los 3 modelos.

📊 Comparación visual de resultados Gráfica generada automáticamente para mostrar cómo cada modelo distribuye las probabilidades entre las clases.

🧾 Interpretación técnica Evaluación de la confianza de la predicción según el porcentaje máximo de probabilidad. Mensajes explicativos integrados.

📁 Estructura del repositorio

Proyecto/Dashboard_Streamlit/
├── modelos/                 # Carpeta con modelos .pkl serializados
│   ├── modelo_rf.pkl
│   ├── modelo_log.pkl
│   └── modelo_mlp.pkl
├── requirements.txt         # Librerías necesarias para ejecución
├── streamlit_app.py         # Script principal de la app
└── README.md                # Este archivo
⚙️ Reproducibilidad local
Para correr la app en tu máquina local:
git clone https://github.com/ivansst773/Aprendizaje_de_Maquina.git
cd Aprendizaje_de_Maquina/Proyecto/Dashboard_Streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py

🔒 Versiones y compatibilidad
Modelos entrenados con scikit-learn==1.6.1

Despliegue en scikit-learn==1.7.1 (verificado compatible)

pickle fue usado para serializar los clasificadores

🤝 Créditos y autoría
App desarrollada por Edgar Ivan Calpa Cuacialpud como parte de su entrega de aprendizaje automático. Resalta por su enfoque educativo, estructura reproducible y claridad técnica.
