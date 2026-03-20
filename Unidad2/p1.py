# Importar las librerías necesarias
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

# Cargar archivo CSV
df = pd.read_csv("Documents/Ventas.csv") 

# Mostrar las primeras filas del DataFrame y la estadística descriptiva
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe()) 

# Graficar la serie de tiempo de las ventas mensuales
plt.figure(figsize=(10,5)) 
plt.plot(df["Mes"], df["Ventas"], marker="o") 
plt.title("Ventas mensuales retail") 
plt.xlabel("Mes") 
plt.ylabel("Ventas") 
plt.show() 

# VARIABLES INDEPENDIENTES O SERIE DE TIEMPO=X DEPENDIENTE=Y 
X = df[["Mes"]] 
y = df["Ventas"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# MODELO REGRESION LINEAL 
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train, y_train) 
pred_lineal = modelo_lineal.predict(X_test) 

# MODELO REGRESION POLYNOMIAL 
# PolynomialFeatures crea nuevas variables elevando la variable original al cuadrado, al cubo, etc., 
# para que el modelo pueda ajustar curvas en lugar de solo líneas rectas. 
# El grado del polinomio controla qué tan curva puede ser la predicción. Un grado bajo produce líneas simples,  
# mientras que un grado alto permite curvas más complejas, pero si es demasiado alto puede provocar sobreajuste  
# Aprende ruido en lugar de  el patron real de los datos. 
modelo_poli = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()) 
modelo_poli.fit(X_train, y_train) 
pred_poli = modelo_poli.predict(X_test) 

# Medir el desempeño de modelos con métricas
print("\nMODELO LINEAL") 
print("MSE:", mean_squared_error(y_test, pred_lineal)) 
print("R2:", r2_score(y_test, pred_lineal)) 
print("\nMODELO POLINOMIAL") 
print("MSE:", mean_squared_error(y_test, pred_poli)) 
print("R2:", r2_score(y_test, pred_poli)) 

# PREDICCION FUTURA 
mes_futuro = np.array([[61]])
pred_lineal_futuro = modelo_lineal.predict(mes_futuro) 
pred_poli_futuro = modelo_poli.predict(mes_futuro) 

print("\nPrediccion mes 61 (lineal):", pred_lineal_futuro) 
print("Prediccion mes 61 (polinomial):", pred_poli_futuro) 

plt.figure(figsize=(10,6)) 
# datos reales 
plt.plot(df["Mes"], df["Ventas"], marker="o", label="Datos reales") 
# prediccion lineal 
plt.plot(X_test, pred_lineal, marker="x", label="Predicción lineal") 
# prediccion polinomial 
plt.plot(X_test, pred_poli, marker="s", label="Predicción polinomial") 
# prediccion futura 
plt.scatter(61, pred_poli_futuro, color="green", s=150, label="Predicción mes 61  polinomial") 
# prediccion futura 
plt.scatter(61, pred_lineal_futuro, color="orange", s=150, label="Predicción mes 61  lineal") 
plt.xlabel("Mes") 
plt.ylabel("Ventas") 
plt.title("Serie de tiempo con tendencia y estacionalidad") 
plt.legend() 
plt.show()