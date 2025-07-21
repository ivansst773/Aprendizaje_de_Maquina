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
# ============================

st.subheader("🔍 Comparación de modelos supervisados")

# Cargar modelos
modelo_lr = joblib.load("modelo_lr.pkl")
modelo_rf = joblib.load("modelo_rf.pkl")
modelo_dl = load_model("modelo_dl.h5")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔄 PCA para LR y RF (n_components debe coincidir con el entrenamiento)
pca_clasificacion = PCA(n_components=64)
X_pca_train = pca_clasificacion.fit_transform(X_train)
X_pca_test = pca_clasificacion.transform(X_test)

# 🧪 Preparar datos para CNN
X_test_np = X_test.to_numpy()
X_test_cnn = X_test_np.reshape(-1, 16, 16, 1)

# Función para mostrar métricas y curva ROC
def mostrar_resultados(nombre, y_pred, y_score):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write({
        "Accuracy": round(report["accuracy"], 3),
        "F1-score (Macro)": round(report["macro avg"]["f1-score"], 3),
        "Precision (Macro)": round(report["macro avg"]["precision"], 3)
    })
    fpr, tpr, _ = roc_curve((y_test == 0), y_score)
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
    y_pred = modelo_lr.predict(X_pca_test)
    y_score = modelo_lr.predict_proba(X_pca_test)[:, 0]
    mostrar_resultados("Logistic Regression", y_pred, y_score)

with tab2:
    st.markdown("### 📌 Random Forest (64D PCA)")
    y_pred = modelo_rf.predict(X_pca_test)
    y_score = modelo_rf.predict_proba(X_pca_test)[:, 0]
    mostrar_resultados("Random Forest", y_pred, y_score)

with tab3:
    st.markdown("### 📌 CNN (Entrada 16×16)")
    y_pred = modelo_dl.predict(X_test_cnn).argmax(axis=1)
    y_score = modelo_dl.predict(X_test_cnn)[:, 0]
    mostrar_resultados("CNN", y_pred, y_score)
