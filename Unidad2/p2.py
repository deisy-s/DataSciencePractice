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
df = pd.read_csv("Documents/Lluvias.csv") 

df["id"] = df.index + 1 # Agregar un id que va de 0 a 120

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

# Graficar la serie de tiempo de las lluvias
plt.figure(figsize=(10,5)) 
plt.plot(df["id"], df["Lluvia_mm"], marker="o") 
plt.title("Cantidad de lluvia por mes") 
plt.xlabel("Mes") 
plt.ylabel("Lluvia") 
plt.show()

# Filtrar solamente las lluvias de agosto
df_filter = df[df['Mes_nombre'] == 'Agosto'].copy()
print(df_filter.head()) # Mostrar el DataFrame filtrado para confirmar si se realizó de la manera correcta

# Establecer variables
X = df_filter[["id"]] # Variable independiente
y = df_filter["Lluvia_mm"] # Variable dependiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) # Dividir los datos en datos de entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)

# Crear modelo de regresión lineal
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train.values, y_train.values) # Entrenar el modelo con los datos de entrenamiento
pred_lineal = modelo_lineal.predict(X_test.values) # Hacer predicciones con el modelo entrenado usando los datos de prueba

# Crear modelo de regresión polinomial
modelo_poli = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()) # Crear un modelo que, se transforman las variables con PolynomialFeatures y luego se aplica LinearRegression
modelo_poli.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_poli = modelo_poli.predict(X_test.values) # Hacer predicciones con el modelo entrenado
# Nota: El .values es a causa de un error con sklearn debido a valores no numéricos

# Medir el desempeño de modelos con métricas
print("\nMODELO LINEAL") 
print("MSE:", mean_squared_error(y_test, pred_lineal)) 
print("R2:", r2_score(y_test, pred_lineal)) 
print("\nMODELO POLINOMIAL") 
print("MSE:", mean_squared_error(y_test, pred_poli)) 
print("R2:", r2_score(y_test, pred_poli)) 

# Realizar predicción futura
mes_futuro = np.array([[122]]) 
pred_lineal_futuro = modelo_lineal.predict(mes_futuro) 
pred_poli_futuro = modelo_poli.predict(mes_futuro) 

print("\nPrediccion agosto 2025 (lineal):", pred_lineal_futuro) 
print("Prediccion agosto 2025 (polinomial):", pred_poli_futuro) 

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