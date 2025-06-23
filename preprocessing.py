import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Separar X (features) y y (target)
X = df.drop(columns=["medv"])
y = df["medv"]

# Crear pipeline: escalado + regresión
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Entrenar
pipeline.fit(X, y)

# Guardar modelo
joblib.dump(pipeline, "model.pkl")
print("✅ Modelo guardado como model.pkl con todas las variables.")

df = pd.read_csv(url)
print(df.columns.tolist())
exit()