# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Se grafica primero una serie de tiempo univariada de ventas x mes
# Se aplica un modelo de regresión lineal y polinomial para realizar la predicción de ventas futuras

# Serie de tiempo: Una secuencia de datos ordenados en el tiempo, cada punto es un valor en un tiempo específico

# LIBRERÍAS ------------------------------------------------------------------------------------------------
# Importar las librerías necesarias
import pandas as pd # Para manipulación de datos
import numpy as np # Para operaciones numéricas
import matplotlib.pyplot as plt # Para graficación
from sklearn.linear_model import LinearRegression # Para modelo de regresión lineal
from sklearn.preprocessing import PolynomialFeatures # Para crear características polinomiales
from sklearn.pipeline import make_pipeline # Para crear un pipeline que combine características polinomiales y regresión lineal
from sklearn.model_selection import train_test_split # Para dividir el dataset en entrenamiento y prueba
from sklearn.metrics import mean_squared_error, r2_score # Para evaluar el desempeño del modelo

# CARGA Y ANÁLISIS DE DATOS --------------------------------------------------------------------------------
# Cargar archivo CSV
df = pd.read_csv("Documents/Ventas.csv") 

# Mostrar las primeras filas del DataFrame y la estadística descriptiva
print(df.head()) 
print("\nEstadistica descriptiva") 

# df.describe muestra 8 datos:
# count: La cantidad de valores no nulos en cada columna
# mean: El promedio de cada columna
# std: La desviación estándas o variabilidad de los datos 
# min: El valor menor de cada columna
# 25%: Percentil
# 50%: Percentil
# 75%: Percentil
# max: El valor máximo de cada columna
print(df.describe()) 

# GRAFICACIÓN SERIE DE TIEMPO UNIVARIADA -------------------------------------------------------------------
# Graficar la serie de tiempo (univariada) de las ventas mensuales
plt.figure(figsize=(10,5)) 
plt.plot(df["Mes"], df["Ventas"], marker="o") 
plt.title("Ventas mensuales retail") 
plt.xlabel("Mes") 
plt.ylabel("Ventas") 
plt.show() 

# CREACIÓN DE MODELOS DE REGRESIÓN LINEAL Y POLINOMIAL ------------------------------------------------------
# Variable independiente o serie de tiempo = X 
# Variable dependiente = y
X = df[["Mes"]] 
y = df["Ventas"] 
# Dividir los datos en entrenamiento y prueba (75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Modelo regresión lineal
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train, y_train)  # Entrenar el modelo con los datos de entrenamiento
pred_lineal = modelo_lineal.predict(X_test) # Realizar predicciones con el modelo entrenado

# Modelo regresión polinomial (grado 3) 

# PolynomialFeatures crea nuevas variables elevando la variable original al cuadrado, al cubo, etc., 
# para que el modelo pueda ajustar curvas en lugar de solo líneas rectas

# El grado del polinomio (degree=3) controla qué tan curva puede ser la predicción. Un grado bajo produce líneas 
# simples, mientras que un grado alto permite curvas más complejas, pero si es demasiado alto puede provocar sobreajuste
  
# Aprende ruido en lugar de  el patron real de los datos
modelo_poli = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()) # Crear un pipeline que primero transforma las características a polinomiales y luego aplica regresión lineal
modelo_poli.fit(X_train, y_train) # Entrenar el modelo con los datos de entrenamiento
pred_poli = modelo_poli.predict(X_test) # Realizar predicciones con el modelo entrenado

# CÁLCULO DE MÉTRICAS DE DESEMPEÑO --------------------------------------------------------------------------
# Medir el desempeño de modelos con métricas

# MSE (Mean Squared Error) mide el error promedio al cuadrado entre las predicciones y los valores reales
# Un MSE más bajo indica un mejor ajuste del modelo a los datos (entre menor, mejor)

# R2 (R-squared) mide la proporción de la varianza en la variable dependiente que es explicada por el modelo
# Un R2 más cercano a 1 indica un mejor ajuste del modelo a los datos (entre más cercas a 1, mejor)
print("\nMODELO LINEAL") 
print("MSE:", mean_squared_error(y_test, pred_lineal)) 
print("R2:", r2_score(y_test, pred_lineal)) 
print("\nMODELO POLINOMIAL") 
print("MSE:", mean_squared_error(y_test, pred_poli)) 
print("R2:", r2_score(y_test, pred_poli)) 

# PREDICCIÓN FUTURA -----------------------------------------------------------------------------------------
# Predicción futura
mes_futuro = np.array([[61]]) # Predecir las ventas para el mes 61 (enero) utilizando ambos modelos
pred_lineal_futuro = modelo_lineal.predict(mes_futuro)  # Realizar la predicción con el modelo lineal
pred_poli_futuro = modelo_poli.predict(mes_futuro) # Realizar la predicción con el modelo polinomial

# Mostrar las predicciones
print("\nPrediccion mes 61 (lineal):", pred_lineal_futuro) 
print("Prediccion mes 61 (polinomial):", pred_poli_futuro) 

# GRAFICACIÓN -----------------------------------------------------------------------------------------------
# Graficar las predicciones futuras junto con los datos reales y las predicciones de ambos modelos
plt.figure(figsize=(10,6)) # Configurar el tamaño de la figura

# Datos reales 
plt.plot(df["Mes"], df["Ventas"], marker="o", label="Datos reales") 

# Predicción lineal 
plt.plot(X_test, pred_lineal, marker="x", label="Predicción lineal") 
# Predicción polinomial 
plt.plot(X_test, pred_poli, marker="s", label="Predicción polinomial") 

# Predicción futura (polinomial)
# La primer variable (61) representa el mes para el cual se hace la predicción
# La segunda variable (pred_poli_futuro) representa el valor de ventas predicho para ese mes utilizando el modelo polinomial
# El color y el tamaño del punto (s=150) se utilizan para destacar la predicción futura en la gráfica
plt.scatter(61, pred_poli_futuro, color="green", s=150, label="Predicción mes 61  polinomial") 
# Predicción futura (lineal)
plt.scatter(61, pred_lineal_futuro, color="orange", s=150, label="Predicción mes 61  lineal") 

plt.xlabel("Mes") 
plt.ylabel("Ventas") 
plt.title("Serie de tiempo con tendencia y estacionalidad") 
plt.legend() 
plt.show()