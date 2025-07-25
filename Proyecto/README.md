
🧠 Proyecto Final - Teoría de Aprendizaje de Máquina 2025-1
Diagnóstico Oftalmológico Asistido por Machine Learning sobre Señales PERG

Este repositorio documenta el desarrollo completo de un sistema de clasificación multiclase aplicado al diagnóstico oftalmológico, utilizando señales PERG y variables demográficas, con entrenamiento, evaluación y despliegue web de tres modelos comparativos.

📁 Estructura del Repositorio
Proyecto/
├── parte_practica/
│   ├── Proyecto_TAM.pdf         # Documento técnico completo del proyecto
│   ├── Notebooks/               # Scripts de entrenamiento, comparación y exportación
├── Dashboard_Streamlit/
│   ├── streamlit_app.py         # App interactiva para evaluación clínica
│   ├── modelos/                 # Modelos serializados para despliegue
│   ├── README.md                # README específico del dashboard
│   └── requirements.txt         # Requisitos para ejecución local

🎯 Objetivo General
Desarrollar e implementar una herramienta educativa y técnica que permita comparar clasificadores entrenados sobre señales PERG, interpretando sus predicciones de forma visual y comprensible para usuarios no expertos.

🧪 Contenido técnico incluido
Motivación clínica y relevancia del uso de ML en diagnóstico temprano

Delimitación del problema y pregunta de investigación

Revisión de literatura y estado del arte de métodos similares

Entrenamiento y evaluación de modelos Random Forest, Regresión Logística y MLPClassifier

Métricas cuantitativas y visualización comparativa de resultados

Interpretabilidad de predicciones mediante análisis de probabilidades

Despliegue web interactivo usando Streamlit con selector de modelo

🔗 Acceso directo a la app
👉 Abrir aplicación Streamlit Interactúa con los modelos, simula casos clínicos y explora sus resultados de forma visual.
(https://share.streamlit.io/ivansst773/aprendizaje_de_maquina/main/Proyecto/Dashboard_Streamlit/streamlit_app.py)


🔍 Para evaluadores y colaboradores
La app permite probar casos extremos (valores fuera de rango, datos incompletos)

Los artefactos .pkl están validados y cargados condicionalmente

El documento técnico Proyecto_TAM.pdf cubre todos los ítems exigidos en la guía oficial del curso, incluyendo referencias académicas y discusión crítica de resultados

Repositorio compatible con Colab y Kaggle

🤝 Créditos
Este proyecto fue desarrollado por Edgar Ivan Calpa Cuacialpud como entrega final del curso Teoría de Aprendizaje de Máquina 2025-1 en la Universidad Nacional de Colombia. Destaca por su enfoque reproducible, claridad técnica y componente interactivo educativo.

