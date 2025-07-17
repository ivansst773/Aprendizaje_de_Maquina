import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

def to_dense(X):
    return X.toarray()


# 📥 Cargar el dataset
url = "https://raw.githubusercontent.com/wblakecannon/ames/master/data/housing.csv"
df = pd.read_csv(url)

# 🎯 Variables
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# 🔍 Separar numéricas y categóricas
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# 🔧 Preprocesamiento con imputación
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# 🔁 División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Random Forest
rf_pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
dump(rf_pipeline, "random_forest_model.pkl")

# 🧪 Lasso
lasso_pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", Lasso(alpha=0.1))
])
lasso_pipeline.fit(X_train, y_train)
dump(lasso_pipeline, "lasso_model.pkl")

# 📈 Gaussian Process
kernel = C(1.0) * RBF()
gpr_pipeline = Pipeline([
    ("pre", preprocessor),
    ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
    ("model", GaussianProcessRegressor(kernel=kernel, alpha=1e-2))
])

gpr_pipeline.fit(X_train, y_train)
dump(gpr_pipeline, "gpr_model.pkl")

print("✅ Modelos entrenados y guardados correctamente.")
