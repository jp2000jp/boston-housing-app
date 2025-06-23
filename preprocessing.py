import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Cargar dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

X = df.drop(columns=["medv"])
y = df["medv"]

# Pipelines
def create_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

# Entrenar y guardar cada modelo
models = {
    "lr": create_pipeline(LinearRegression()),
    "rf": create_pipeline(RandomForestRegressor(n_estimators=100, random_state=42)),
    "xgb": create_pipeline(xgb.XGBRegressor(n_estimators=100, random_state=42))
}

for name, pipeline in models.items():
    pipeline.fit(X, y)
    joblib.dump(pipeline, f"model_{name}.pkl")
    print(f"✅ Modelo {name.upper()} guardado como model_{name}.pkl")
