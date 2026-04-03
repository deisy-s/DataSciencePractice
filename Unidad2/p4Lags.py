# Importar librerías necesarias
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
 
# Cargar los datos
df = pd.read_csv("Documents/VentasP3.csv") 

# Mostrar las primeras filas del df e información general del dataset
print("Primeras filas:") 
print(df.head()) 
 
print("\nInformación del dataset:") 
print(df.info()) 
 
# Convertir la  columna Temporada a valores numericos 
df["Temporada"] = df["Temporada"].map({ 
    "Baja": 0, 
    "Media": 1, 
    "Alta": 2 
}) 

df["Tiempo"] = df["Año"] * 12 + df["Mes"] 
 
# Crear lags features (valores rezagados)
# Desplaza una columna una fila hacia abajo. El valor de la fila actual recibe el valor de la fila anterior 
# Por eso aparecen valores vacíos al inicio 
df["Ventas_lag1"] = df["Ventas"].shift(1) # Ventas del mes pasado
df["Ventas_lag2"] = df["Ventas"].shift(2) # Ventas de hace dos meses
 
# Mostrar el dataset con las nuevas columnas de lags
print("\nDataset con lag features:") 
print(df.head(10)) 
 
# Eliminar nulos
df = df.dropna() 
 
# Graficar nulos
plt.figure(figsize=(10,5)) 

plt.plot(df["Tiempo"], df["Ventas"], marker="o") 

plt.title("Ventas en el tiempo") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.grid() 
plt.show() 
 
# Definir las variables para el train test 
X = df[[ 
    "Tiempo", 
    "Precio", 
    "Publicidad", 
    "Descuento", 
    "Clientes", 
    "Temporada", 
    "Ventas_lag1", 
    "Ventas_lag2" 
]] 
 
y = df["Ventas"] 
 
# Dividir los datos en entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) 
 
# Modelo lineal con lags
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train, y_train) 
pred_lineal = modelo_lineal.predict(X_test) 
 
# Modelo random forest con lags 
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42) 
modelo_rf.fit(X_train, y_train) 
pred_rf = modelo_rf.predict(X_test) 
 
# Evaluar modelos con métricas MSE y R2
def evaluar_modelo(y_real, y_pred, nombre): 
    print(f"\n--- {nombre} ---") 
    print("MSE:", mean_squared_error(y_real, y_pred)) 
    print("R2:", r2_score(y_real, y_pred)) 
 
evaluar_modelo(y_test, pred_lineal, "Regresión Lineal con Lags") 
evaluar_modelo(y_test, pred_rf, "Random Forest con Lags") 
 
# Predicción futura con ambos modelos
futuro = pd.DataFrame({ 
    "Tiempo": [2025*12 + 1], 
    "Precio": [95], 
    "Publicidad": [4500], 
    "Descuento": [25], 
    "Clientes": [600], 
    "Temporada": [2], 
    "Ventas_lag1": [df["Ventas"].iloc[-1]], 
    "Ventas_lag2": [df["Ventas"].iloc[-2]] 
}) 
 
pred_lineal_fut = modelo_lineal.predict(futuro) 
pred_rf_fut = modelo_rf.predict(futuro) 

# Imprimir las predicciones futuras
print("\nPredicción futura:") 
print("Regresión Lineal:", pred_lineal_fut[0]) 
print("Random Forest:", pred_rf_fut[0]) 
 
# Gráfica final con datos reales, predicciones y forecasting futuro
plt.figure(figsize=(12,6)) 
 
plt.plot(df["Tiempo"], df["Ventas"], label="Histórico", color="black") 
plt.plot(X_test["Tiempo"], y_test, label="Real ", marker="o") 
plt.plot(X_test["Tiempo"], pred_lineal, label="Predicción Lineal", linestyle="dashed") 
plt.plot(X_test["Tiempo"], pred_rf, label="Predicción RandomF", linestyle="dashed") 
 
plt.scatter(futuro["Tiempo"], pred_lineal_fut, label="Futuro Lineal", s=120) 
plt.scatter(futuro["Tiempo"], pred_rf_fut, label="Futuro RandomForest", s=120) 
 
plt.title("Ventas reales, predicciones y forecasting futuro") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.legend() 
plt.grid() 
plt.show()