import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("model.pkl")

st.title("🏠 Predicción de Precio de Vivienda (Boston Housing)")
st.write("Completa los datos de la zona para predecir el precio medio (en miles de dólares).")

# Entradas del usuario (usamos los nombres exactos del entrenamiento)
crim = st.number_input("Tasa de criminalidad per cápita (crim)", 0.0, 100.0, 0.1)
zn = st.number_input("Proporción de terrenos residenciales (zn)", 0.0, 100.0, 0.0)
indus = st.slider("Proporción de terrenos industriales (indus)", 0.0, 30.0, 10.0)
chas = st.selectbox("¿Limita con el río Charles? (chas)", [0, 1])
nox = st.slider("Contaminación NOx (nox)", 0.3, 1.0, 0.5)
rm = st.slider("Número medio de habitaciones (rm)", 3.0, 9.0, 6.0)
age = st.slider("Edad promedio de las viviendas (age)", 0.0, 100.0, 50.0)
dis = st.slider("Distancia a zonas de empleo (dis)", 1.0, 13.0, 5.0)
rad = st.slider("Índice de acceso a autopistas (rad)", 1, 24, 4)
tax = st.slider("Tasa de impuestos a la propiedad (tax)", 100, 800, 300)
ptratio = st.slider("Ratio alumno/profesor (ptratio)", 10.0, 22.0, 18.0)
b = st.slider("Población negra (b)", 0.0, 400.0, 350.0)
lstat = st.slider("% bajo estatus socioeconómico (lstat)", 1.0, 40.0, 12.5)

# Crear DataFrame para el modelo
input_data = pd.DataFrame([{
    "crim": crim,
    "zn": zn,
    "indus": indus,
    "chas": chas,
    "nox": nox,
    "rm": rm,
    "age": age,
    "dis": dis,
    "rad": rad,
    "tax": tax,
    "ptratio": ptratio,
    "b": b,
    "lstat": lstat
}])

# Predicción
pred = model.predict(input_data)[0]
st.subheader(f"💰 Precio estimado: ${pred * 1000:,.2f}")
