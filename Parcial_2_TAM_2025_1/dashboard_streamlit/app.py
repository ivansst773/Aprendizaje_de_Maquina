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

# 🧠 Cargar USPS
@st.cache_data
def cargar_datos():
    usps = fetch_openml(name='USPS', version=1)
    X = usps.data
    y = usps.target.astype(int)
    return X, y

X, y = cargar_datos()

st.set_page_config(page_title="USPS Dashboard", layout="wide")
st.title("📊 Parcial 2 TAM 2025 – USPS Dashboard")

# ============================
# 🎯 Proyección con PCA y UMAP
# ============================

st.sidebar.header("Configuración de proyección")
dim = st.sidebar.slider("Dimensiones", min_value=2, max_value=3, value=2)
n_neighbors = st.sidebar.slider("Vecinos (UMAP)", 2, 30, 10)

st.subheader("🔸 Proyección con PCA")
pca = PCA(n_components=dim)
X_pca = pca.fit_transform(X)
st.scatter_chart(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(dim)]).assign(label=y))

st.subheader("🔹 Proyección con UMAP")
umap = UMAP(n_components=dim, n_neighbors=n_neighbors)
X_umap = umap.fit_transform(X)
st.scatter_chart(pd.DataFrame(X_umap, columns=[f"UMAP{i+1}" for i in range(dim)]).assign(label=y))

# ============================
# 🧠 Clasificación Supervisada
# ============================

st.subheader("🔍 Comparación de modelos supervisados")

# Cargar modelos
modelo_lr = joblib.load("modelo_lr.pkl")
modelo_rf = joblib.load("modelo_rf.pkl")
modelo_dl = load_model("modelo_dl.h5")

# Preparar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_cnn = X_test.reshape(-1, 256, 1) if X_test.shape[1] == 256 else X_test.reshape(-1, 16, 16, 1)

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["🔸 Logistic Regression", "🔹 Random Forest", "🔻 CNN"])

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

with tab1:
    st.markdown("### 📌 Logistic Regression")
    y_pred = modelo_lr.predict(X_test)
    y_score = modelo_lr.predict_proba(X_test)[:, 0]
    mostrar_resultados("Logistic Regression", y_pred, y_score)

with tab2:
    st.markdown("### 📌 Random Forest")
    y_pred = modelo_rf.predict(X_test)
    y_score = modelo_rf.predict_proba(X_test)[:, 0]
    mostrar_resultados("Random Forest", y_pred, y_score)

with tab3:
    st.markdown("### 📌 CNN (Keras)")
    y_pred = modelo_dl.predict(X_test_cnn).argmax(axis=1)
    y_score = modelo_dl.predict(X_test_cnn)[:, 0]
    mostrar_resultados("CNN", y_pred, y_score)
