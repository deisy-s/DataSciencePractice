# Regresión lineal para diabetes
import pandas as pd # Maneja la tabla de datos
import matplotlib.pyplot as plt # Crear la gráfica
from sklearn.linear_model import LinearRegression # Modelo de regresión lineal

# Cargar el csv
df = pd.read_csv("Documents/diabetes.csv")

# Guardamos los datos de interes en una variable X y Y
vars = ['Insulin', 'BloodPressure', 'BMI'] # Juntar las variables necesarias en un arreglo
X = df[vars] # Variable independiente
y = df['Glucose'] # Variable dependiente

# Instanciamos y ajustamos el modelo de regresión lineal
# Crea el objeto del modelo vacío
slr = LinearRegression() # Standar Linear Regression (slr)
slr.fit(X,y) # Entrenar el modelo
# Nota: C debe ser una matriz (DataFrame) y Y un vector (Serie)

# Predicciones del modelo de acuerdo a lo aprendido
y_pred = slr.predict(X)

# Crear la gráfica
plt.figure(figsize=(6,6))
# Mapear la realidad (y) vs las predicciones (y_pred)
plt.scatter(y, y_pred, color='lightgreen') # Mapear predicciones

# Dibujar una línea de referencia (diagonal de 45 grados)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', lw=2)

plt.xlabel("Glucosa real")
plt.ylabel("Glucosa estimada")
plt.title("Glucosa real vs estimada")
plt.show()