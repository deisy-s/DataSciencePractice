# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Se grafica primero una serie de tiempo de ventas x tiempo
# Se aplica un modelo de regresión lineal, polinomial y de clasificación para realizar la predicción de 
# ventas futuras, los cuales utilizan varias variables independientes

# LIBRERÍAS ------------------------------------------------------------------------------------------------
# Importar las librerías necesarias
import pandas as pd # Para manipulación de datos
import numpy as np # Para operaciones numéricas
import matplotlib.pyplot as plt # Para graficación
from sklearn.linear_model import LinearRegression # Para modelo de regresión lineal
from sklearn.preprocessing import PolynomialFeatures # Para crear características polinomiales
from sklearn.preprocessing import LabelEncoder # Para convertir variables categóricas a numéricas
from sklearn.ensemble import RandomForestRegressor # Para modelo de regresión Random Forest
from sklearn.pipeline import make_pipeline # Para crear un pipeline que combine características polinomiales y regresión lineal
from sklearn.model_selection import train_test_split # Para dividir el dataset en entrenamiento y prueba
from sklearn.metrics import mean_squared_error, r2_score # Para evaluar el desempeño de los modelos de regresión
from sklearn.metrics import classification_report # Para evaluar el desempeño del modelo de clasificación
from sklearn.tree import plot_tree # Para graficar el árbol de decisión del modelo Random Forest

# CARGA Y ANÁLISIS DE DATOS --------------------------------------------------------------------------------
# Cargar archivo CSV 
df = pd.read_csv("Documents/VentasP3.csv") 

# El dataset no tiene un id único, por lo cual se crea una columna nueva con esto
# para poder graficar la serie de tiempo y aplicar los modelos
df["id"] = df.index + 1 # Agregar un id unico

df["Tiempo"] = df["Año"].astype(str) + "-" + df["Mes"].astype(str).str.zfill(2) # Crear una nueva columna 'Tiempo' combinando 'Año' y 'Mes_nombre'

# Convierte texto a variables numéricas
temp = LabelEncoder() # Crear un codificador de etiquetas para la columna 'Temporada'

# fit_transform(): fit aprende cuántas categorías hay y transform cambia cada categoría a un número
df['Temporada'] = temp.fit_transform(df['Temporada']) # Transformar la columna 'Temporada' a valores numéricos

# Mostrar las primeras filas del DataFrame y la estadística descriptiva
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe()) 
print(df.dtypes) # Mostrar el tipo de dato de cada columna

# GRAFICACIÓN SERIE DE TIEMPO ------------------------------------------------------------------------------
# Graficar la serie de tiempo de las ventas
plt.figure(figsize=(12,5)) 

# Para utilizar la columna 'Tiempo' como eje x, es necesario convertirla a un formato de fecha o a un número
plt.plot(df["Tiempo"], df["Ventas"], marker="o", label="Ventas")

plt.title("Serie de tiempo: Variables mensuales") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.xticks(rotation=45) # Rotar las etiquetas del eje x para mejorar la legibilidad
plt.legend() 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 

# CREACIÓN DE MODELOS DE REGRESIÓN LINEAL, POLINOMIAL Y CLASIFICACIÓN --------------------------------------
# Establecer variables
X = df[["id", "Precio", "Publicidad", "Descuento", "Clientes", "Temporada"]] # Variable independiente
y = df["Ventas"] # Variable dependiente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) # Dividir los datos en datos de entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)

# Crear modelo de regresión lineal
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train.values, y_train.values) # Entrenar el modelo con los datos de entrenamiento
pred_lineal = modelo_lineal.predict(X_test.values) # Hacer predicciones con el modelo entrenado usando los datos de prueba

# Crear modelo de regresión polinomial (grado 2)
modelo_poli = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()) # Crear un modelo que, se transforman las variables con PolynomialFeatures y luego se aplica LinearRegression
modelo_poli.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_poli = modelo_poli.predict(X_test.values) # Hacer predicciones con el modelo entrenado
# Nota: El .values es a causa de un error con sklearn debido a valores no numéricos

# Crear modelo de regresión con Random Forest
modelo_random = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=40)
modelo_random.fit(X_train.values, y_train.values) # Entrenar el modelo
pred_random = modelo_random.predict(X_test.values) # Hacer predicciones con el modelo entrenado

# Medir el desempeño de modelos con métricas
# MSE -> entre menor, mejor
# R2 -> entre más cercas a 1, mejor
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
# Se predice para el mes 38 (enero, 2023) utilizando los modelos, con las siguientes características:
# Precio: 95
# Publicidad: 4500
# Descuento: 25
# Clientes: 600
# Temporada: Alta
mes_futuro = np.array([[38, 95, 4500, 25, 500, 0]]) # LabelEncoder convierte la Temporada alta en 0

pred_lineal_futuro = modelo_lineal.predict(mes_futuro) 
pred_poli_futuro = modelo_poli.predict(mes_futuro) 
pred_random_futuro = modelo_random.predict(mes_futuro)

# Mostrar las predicciones futuras
print("\nPrediccion Enero 2023 (lineal):", pred_lineal_futuro) 
print("Prediccion Enero 2023 (polinomial):", pred_poli_futuro) # Esta es sumamente erronea
print("Prediccion Enero 2023 (Random Forest):", pred_random_futuro) 

# GRAFICACIÓN ----------------------------------------------------------------------------------------------
# Graficar los datos reales, las predicciones de los modelos y la predicción futura
plt.figure(figsize=(12,6))

# Extraer solo la columna del 'id' de X_test
meses_test = X_test.iloc[:, 0] # Se mostraba un error porque X_test es un df con varias columnas

# Datos reales
plt.plot(meses_test, y_test, marker='o', label='Real', color='black', linewidth=2)
# Predicción lineal
plt.plot(meses_test, pred_lineal, marker='x', label='Predicción Lineal', linestyle='--')
# Predicción polinomial
plt.plot(meses_test, pred_poli, marker='s', label='Predicción Polinomial', linestyle='--')
# Predicción random forest
plt.plot(meses_test, pred_random, marker='d', label='Predicción Random Forest', linestyle='--')
# Predicción futura polinomial
plt.scatter(38, pred_poli_futuro, color="green", s=150, edgecolors='black', label="Enero 2023 Polinomial", zorder=5)
# Predicción futura lineal
plt.scatter(38, pred_lineal_futuro, color="orange", s=150, edgecolors='black', label="Enero 2023 Lineal", zorder=5)
# Predicción futura random forest
plt.scatter(38, pred_random_futuro, color="blue", s=150, edgecolors='black', label="Enero 2023 Random Forest", zorder=5)

plt.title("Ajuste de Modelos: Real vs Predicción")
plt.xlabel('Mes')
plt.ylabel('Ventas')
plt.tick_params(axis='x', rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Graficar árbol de Random Forest (para visualizar la variable más importante)
print(classification_report(y_test, pred_random))

plt.figure(figsize=(12, 6)) # Establecer el tamaño de la figura para la gráfica
# Dibuja el arbol
plot_tree(modelo_random.estimators_[0], feature_names=X.columns, filled=True, fontsize=10)
plt.show()