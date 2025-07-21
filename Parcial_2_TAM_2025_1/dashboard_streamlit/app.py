import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import joblib
from tensorflow.keras.models import load_model
import os

# 🧠 Cargar USPS desde OpenML
@st.cache_data
def cargar_datos():
    usps = fetch_openml(name="USPS", version=1)
    X = usps.data / 255.0
    y = usps.target.astype(int)
    return X, y

X, y = cargar_datos()

# 🛠️ Configurar página de Streamlit
st.set_page_config(page_title="USPS Dashboard", layout="wide")
st.title("📊 Parcial 2 TAM 2025 – USPS Dashboard")

# 📂 Mostrar archivos que detecta el entorno
st.sidebar.write("📂 Archivos detectados por Streamlit:")
st.sidebar.write(os.listdir())

# ============================
# 🎯 Proyecciones PCA y UMAP
# ============================

st.sidebar.header("Configuración de proyección")
dim = st.sidebar.slider("Dimensiones", min_value=2, max_value=3, value=2)
n_neighbors = st.sidebar.slider("Vecinos (UMAP)", 2, 30, 10)

def plot_proyeccion(X_proj, labels, titulo, eje_x, eje_y):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.legend(*scatter.legend_elements(), title="Etiqueta")
    ax.set_title(titulo)
    ax.set_xlabel(eje_x)
    ax.set_ylabel(eje_y)
    st.pyplot(fig)

st.subheader("🔸 Proyección con PCA")
pca_visual = PCA(n_components=dim)
X_pca = pca_visual.fit_transform(X)
plot_proyeccion(X_pca, y, "Proyección PCA", "PC1", "PC2")

st.subheader("🔹 Proyección con UMAP")
umap = UMAP(n_components=dim, n_neighbors=n_neighbors)
X_umap = umap.fit_transform(X)
plot_proyeccion(X_umap, y, "Proyección UMAP", "UMAP1", "UMAP2")

# ============================
# 🧠 Clasificación Supervisada
# ============================streamlit run Parcial_2_TAM_2025_1/dashboard_streamlit/app.py

st.subheader("🔍 Comparación de modelos supervisados")

# 📁 Cargar modelos con verificación
MODELO_PATH = os.path.dirname(__file__)

if os.path.exists(os.path.join(MODELO_PATH, "modelo_lr.pkl")):
    modelo_lr = joblib.load(os.path.join(MODELO_PATH, "modelo_lr.pkl"))
else:
    modelo_lr = None
    st.error("❌ No se encontró el archivo 'modelo_lr.pkl'. Verifica que esté en la carpeta del dashboard y que lo hayas subido vía Git.")

if os.path.exists(os.path.join(MODELO_PATH, "modelo_rf.pkl")):
    modelo_rf = joblib.load(os.path.join(MODELO_PATH, "modelo_rf.pkl"))
else:
    modelo_rf = None
    st.error("❌ No se encontró el archivo 'modelo_rf.pkl'. Verifica que esté en la carpeta del dashboard y que lo hayas subido vía Git.")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar PCA para clasificación (por ejemplo, 64 componentes)
pca_clasificacion = PCA(n_components=64)
X_pca_train = pca_clasificacion.fit_transform(X_train)
X_pca_test = pca_clasificacion.transform(X_test)

# Cargar el modelo CNN
modelo_dl = load_model(os.path.join(MODELO_PATH, "modelo_dl.h5"))

# Preparar datos de test para la CNN
X_test_cnn = X_test.to_numpy().reshape(-1, 8, 8, 1)

# 📊 Función para mostrar métricas y curva ROC
def mostrar_resultados(nombre, y_true, y_pred, y_score):
    report = classification_report(y_true, y_pred, output_dict=True)
    st.write({
        "Accuracy": round(report["accuracy"], 3),
        "F1-score (Macro)": round(report["macro avg"]["f1-score"], 3),
        "Precision (Macro)": round(report["macro avg"]["precision"], 3)
    })
    fpr, tpr, _ = roc_curve((y_true == 0), y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title(f"Curva ROC - {nombre}")
    ax.set_xlabel("Falsos Positivos")
    ax.set_ylabel("Verdaderos Positivos")
    ax.legend()
    st.pyplot(fig)

# ============================
# 📂 Comparación en pestañas
# ============================

tab1, tab2, tab3 = st.tabs(["🔸 Logistic Regression", "🔹 Random Forest", "🔻 CNN"])

with tab1:
    st.markdown("### 📌 Logistic Regression (64D PCA)")
    if modelo_lr is not None:
        y_pred = modelo_lr.predict(X_pca_test)
        y_score = modelo_lr.predict_proba(X_pca_test)[:, 0]
        mostrar_resultados("Logistic Regression", y_test, y_pred, y_score)
    else:
        st.warning("⚠️ El modelo Logistic Regression no está disponible.")

with tab2:
    st.markdown("### 📌 Random Forest (64D PCA)")
    if modelo_rf is not None:
        y_pred = modelo_rf.predict(X_pca_test)
        y_score = modelo_rf.predict_proba(X_pca_test)[:, 0]
        mostrar_resultados("Random Forest", y_test, y_pred, y_score)
    else:
        st.warning("⚠️ El modelo Random Forest no está disponible.")

with tab3:
    st.markdown("### 📌 CNN (Entrada imagen 8x8)")
    if modelo_dl is not None:
        y_pred_full = modelo_dl.predict(X_test_cnn, batch_size=32).argmax(axis=1)
        y_score_full = modelo_dl.predict(X_test_cnn, batch_size=32)[:, 0]
        y_pred = y_pred_full[:len(y_test)]
        y_score = y_score_full[:len(y_test)]
        mostrar_resultados("CNN", y_test, y_pred, y_score)
    else:
        st.warning("⚠️ El modelo CNN no está disponible.")
