# Importar las librerías necesarias
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

# Cargar archivo CSV 
df = pd.read_csv("Documents/VentasP3.csv") 

df["id"] = df.index + 1 # Agregar un id unico

df["Tiempo"] = df["Año"].astype(str) + "-" + df["Mes"].astype(str) # Crear una nueva columna 'Tiempo' combinando 'Año' y 'Mes_nombre'

# Convierte texto a variables numéricas
temp = LabelEncoder() # Crear un codificador de etiquetas para la columna 'Temporada'

# fit_transform(): fit aprende cuántas categorías hay y transform cambia cata categoría a un número
df['Temporada'] = temp.fit_transform(df['Temporada']) # Transformar la columna 'Temporada' a valores numéricos

# Mostrar las primeras filas del DataFrame y la estadística descriptiva
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe()) 
print(df.dtypes) # Mostrar el tipo de dato de cada columna

# Graficar la serie de tiempo de las ventas
plt.figure(figsize=(10,5)) 
plt.plot(df["id"], df["Ventas"], marker="o", label="Ventas") #Al utilizar Tiempo, se muestra un error porque no es numerico
plt.title("Serie de tiempo multivariada: Variables mensuales") 
plt.xlabel("Mes") 
plt.ylabel("Valor") 
plt.legend() 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 

# # Establecer variables
X = df[["id", "Precio", "Publicidad", "Descuento", "Clientes", "Temporada"]] # Variable independiente
y = df["Ventas"] # Variable dependiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) # Dividir los datos en datos de entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)

# Crear modelo de regresión lineal
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train.values, y_train.values) # Entrenar el modelo con los datos de entrenamiento
pred_lineal = modelo_lineal.predict(X_test.values) # Hacer predicciones con el modelo entrenado usando los datos de prueba

# Crear modelo de regresión polinomial
modelo_poli = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()) # Crear un modelo que, se transforman las variables con PolynomialFeatures y luego se aplica LinearRegression
modelo_poli.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_poli = modelo_poli.predict(X_test.values) # Hacer predicciones con el modelo entrenado
# Nota: El .values es a causa de un error con sklearn debido a valores no numéricos

# Crear modelo de clasificación con Random Forest
modelo_random = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)
modelo_random.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_random = modelo_random.predict(X_test.values)

# Medir el desempeño de modelos con métricas
print("\nMODELO LINEAL") 
print("MSE:", mean_squared_error(y_test, pred_lineal)) 
print("R2:", r2_score(y_test, pred_lineal)) 
print("\nMODELO POLINOMIAL") 
print("MSE:", mean_squared_error(y_test, pred_poli)) 
print("R2:", r2_score(y_test, pred_poli)) 
print("\nMODELO RANDOM FOREST")
print("MSE:", mean_squared_error(y_test, pred_random))
print("R2:", r2_score(y_test, pred_random))

# Realizar predicción futura
mes_futuro = np.array([[38, 95, 4500, 25, 500, 0]]) # LabelEncoder convierte la Temporada alta en 0
pred_lineal_futuro = modelo_lineal.predict(mes_futuro) 
pred_poli_futuro = modelo_poli.predict(mes_futuro) 
pred_random_futuro = modelo_random.predict(mes_futuro)

print("\nPrediccion Enero 2023 (lineal):", pred_lineal_futuro) 
print("Prediccion Enero 2023 (polinomial):", pred_poli_futuro) # Esta es sumamente erronea
print("Prediccion Enero 2023 (Random Forest):", pred_random_futuro) 

# Graficar los datos reales, las predicciones de ambos modelos y la predicción futura
plt.figure(figsize=(12,6))
# Extraer solo la columna del 'id' de X_test
meses_test = X_test.iloc[:, 0] # Se mostraba un error porque X_test es un df con varias columnas

plt.plot(meses_test, y_test, marker='o', label='Real', color='black', linewidth=2)
plt.plot(meses_test, pred_lineal, marker='x', label='Predicción Lineal', linestyle='--')
plt.plot(meses_test, pred_poli, marker='s', label='Predicción Polinomial', linestyle='--')
plt.scatter(38, pred_poli_futuro, color="green", s=150, edgecolors='black', label="Enero 2023 Polinomial", zorder=5)
plt.scatter(38, pred_lineal_futuro, color="orange", s=150, edgecolors='black', label="Enero 2023 Lineal", zorder=5)

plt.title("Ajuste de Modelos: Real vs Predicción")
plt.xlabel('Mes')
plt.ylabel('Ventas')
plt.tick_params(axis='x', rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Graficar Random Forest
print(classification_report(y_test, pred_random))

plt.figure(figsize=(12, 6)) # Establecer el tamaño de la figura para la gráfica
# Dibuja el arbol
plot_tree(modelo_random.estimators_[0], feature_names=X.columns, filled=True, fontsize=10)
plt.show()