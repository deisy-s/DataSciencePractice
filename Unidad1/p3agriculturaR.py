# Practica 3
# Clasificación con scikit learn

import pandas as pd # Maneja la tabla de datos
import matplotlib.pyplot as plt # Crear la gráfica
from sklearn.linear_model import LinearRegression # Modelo de entrenamiento de regresión lineal

# Cargar el CSV
df = pd.read_csv("Documents/agricultura.csv")

# Guardamos los datos de interes en una variable X y Y
vars = ['Riego_mm', 'Humedad_pct', 'Temperatura_C'] # Juntar las 3 variables en un arreglo
X = df[vars] # Variable independiente
y = df['Rendimiento_t_ha'] # Variable dependiente

# Instanciamos y ajustamos el modelo de regresión lineal
slr = LinearRegression() # Standar Linear Regression (slr)
slr.fit(X,y) # Entrenar el modelo

# Predicciones del modelo
y_pred = slr.predict(X)

# Crear la gráfica
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, color='blue') # Predicciones

# Dibujar una línea de referencia (diagonal de 45 grados)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)

plt.xlabel("Rendimiento")
plt.ylabel("Rendimiento estimado")
plt.title("Rendimiento de cosecha")
plt.show()