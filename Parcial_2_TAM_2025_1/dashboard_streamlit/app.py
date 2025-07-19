
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="USPS Dashboard", layout="wide")

st.title("📊 Parcial 2 TAM 2025 – USPS Dashboard")

# Cargar datos
@st.cache_data
def cargar_datos():
    usps = fetch_openml(name='USPS', version=1)
    X = usps.data
    y = usps.target.astype(int)
    return X, y

X, y = cargar_datos()

st.sidebar.header("Configuración de proyección")
dim = st.sidebar.slider("Dimensiones", min_value=2, max_value=3, value=2)
n_neighbors = st.sidebar.slider("Vecinos (UMAP)", 2, 30, 10)

# PCA
st.subheader("🎯 Proyección con PCA")
pca = PCA(n_components=dim)
X_pca = pca.fit_transform(X)
st.scatter_chart(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(dim)]).assign(label=y))

# UMAP
st.subheader("🧭 Proyección con UMAP")
umap = UMAP(n_components=dim, n_neighbors=n_neighbors)
X_umap = umap.fit_transform(X)
st.scatter_chart(pd.DataFrame(X_umap, columns=[f"UMAP{i+1}" for i in range(dim)]).assign(label=y))

# Clasificación (si ya tienes modelos entrenados, los puedes cargar con joblib)

st.info("Próximamente: comparador de modelos supervisados.")

