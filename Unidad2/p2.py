# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Se grafica primero una serie de tiempo univariada de ventas x mes
# Se aplica un modelo de regresión lineal y polinomial para realizar la predicción futura

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
df = pd.read_csv("Documents/Lluvias.csv") 

# Este dataset no contiene un id, por lo cual se crea una columna nueva con esto que va de 1 a 120, 
# para poder graficar la serie de tiempo y aplicar los modelos
df["id"] = df.index + 1 # Agregar un id por medio del índice del DataFrame, sumando 1 para que comience en 1 en vez de 0

# Mostrar las primeras filas del DataFrame y la estadística descriptiva
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe()) 

# GRAFICACIÓN SERIE DE TIEMPO UNIVARIADA -------------------------------------------------------------------
# Graficar la serie de tiempo de las lluvias
plt.figure(figsize=(10,5)) 
plt.plot(df["id"], df["Lluvia_mm"], marker="o") 
plt.title("Cantidad de lluvia por mes") 
plt.xlabel("Mes") 
plt.ylabel("Lluvia") 
plt.show()

# FILTRADO DE DATOS ----------------------------------------------------------------------------------------
# Filtrar solamente las lluvias de agosto
df_filter = df[df['Mes_nombre'] == 'Agosto'].copy()
print(df_filter.head()) # Mostrar el DataFrame filtrado para confirmar si se realizó de la manera correcta

# CREACIÓN DE MODELOS DE REGRESIÓN LINEAL Y POLINOMIAL -----------------------------------------------------
# Establecer variables
X = df_filter[["id"]] # Variable independiente
y = df_filter["Lluvia_mm"] # Variable dependiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) # Dividir los datos en datos de entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)

# Crear modelo de regresión lineal
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train.values, y_train.values) # Entrenar el modelo con los datos de entrenamiento
pred_lineal = modelo_lineal.predict(X_test.values) # Hacer predicciones con el modelo entrenado usando los datos de prueba

# Crear modelo de regresión polinomial (grado 3) 
modelo_poli = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()) # Crear un modelo que, se transforman las variables con PolynomialFeatures y luego se aplica LinearRegression
modelo_poli.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_poli = modelo_poli.predict(X_test.values) # Hacer predicciones con el modelo entrenado
# Nota: El .values es a causa de un error con sklearn debido a valores no numéricos

# Medir el desempeño de modelos con métricas
# MSE -> entre menor, mejor
# R2 -> entre más cercas a 1, mejor
print("\nMODELO LINEAL") 
print("MSE:", mean_squared_error(y_test, pred_lineal)) 
print("R2:", r2_score(y_test, pred_lineal)) 
print("\nMODELO POLINOMIAL") 
print("MSE:", mean_squared_error(y_test, pred_poli)) 
print("R2:", r2_score(y_test, pred_poli)) 

# PREDICCIÓN FUTURA ----------------------------------------------------------------------------------------
# Realizar predicción futura
mes_futuro = np.array([[122]]) 
pred_lineal_futuro = modelo_lineal.predict(mes_futuro) 
pred_poli_futuro = modelo_poli.predict(mes_futuro) 

print("\nPrediccion agosto 2025 (lineal):", pred_lineal_futuro) 
print("Prediccion agosto 2025 (polinomial):", pred_poli_futuro) 

# GRAFICACIÓN ----------------------------------------------------------------------------------------------
# Graficar los datos reales, las predicciones de ambos modelos y la predicción futura
plt.figure(figsize=(10,6)) # Establecer el tamaño de la figura

plt.plot(df_filter["id"], df_filter["Lluvia_mm"], marker="o", label="Datos reales") # Graficar los datos reales

plt.plot(X_test, pred_lineal, marker="x", label="Predicción lineal") # Graficar las predicciones del modelo lineal
plt.plot(X_test, pred_poli, marker="s", label="Predicción polinomial") # Graficar las predicciones del modelo polinomial

plt.scatter(122, pred_poli_futuro, color="green", s=150, label="Predicción Agosto 2025 polinomial") # Graficar la predicción futura del modelo polinomial
plt.scatter(122, pred_lineal_futuro, color="orange", s=150, label="Predicción Agosto 2025 lineal") # Graficar la predicción futura del modelo lineal

plt.xlabel("Mes (Agosto)") 
plt.ylabel("Lluvia (mm)") 
plt.title("Serie de tiempo con tendencia y estacionalidad") 
plt.legend() 
plt.show()