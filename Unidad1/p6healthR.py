import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------
# Carga y limpieza inicial de datos

df = pd.read_csv('Documents/Salud.csv') # Cargar los datos del csv en un DataFrame de pandas

df.dropna(inplace=True) # Eliminar vacios
# inplace establece que los cambios se guardan directamente en df sin tener que declarar df=...

# ----------------------------------------------------------------
# Limpieza de texto

df['Fumador'] = df['Fumador'].str.lower() # str.lower() Convertir toda la columna a minúsculas

# str.replace() Buscar un patrón por regex y reemplazarlo con otro
# regex = True establece que es una expresión de búsqueda, no texto literal
df['Edad'] = df['Edad'].str.replace(r'[a-zA-Zñ ]', '', regex=True) 
df['Glucosa'] = df['Glucosa'].str.replace(r'[a-zA-Z]', '', regex=True)
df['IMC'] = df['IMC'].str.replace(r',', '.', regex=True)
df['IMC'] = df['IMC'].str.replace(r'\.\d', '', regex=True)

# ----------------------------------------------------------------
# Fase de codificación

# Convierte texto a variables numéricas
actividad = LabelEncoder() # Crear un codificador de etiquetas para la columna 'Actividad'
fumador = LabelEncoder()

# fit_transform(): fit aprende cuántas categorías hay y transform cambia cata categoría a un número
df['Actividad_Fisica'] = actividad.fit_transform(df['Actividad_Fisica']) # Transformar la columna 'Actividad' a valores numéricos
df['Fumador'] = fumador.fit_transform(df['Fumador'])

# ----------------------------------------------------------------
# Fase de casting (conversión de tipos)

# pd.to_numeric() convierte el string en el df a un tipo numérico
# errors='coerce' convierte los valores no numéricos a NaN
# downcast='integer' intenta convertir a un tipo de dato entero más pequeño para ahorrar memoria
df['IMC'] = pd.to_numeric(df['IMC'], errors='coerce') # Garantizar que la columna 'IMC' sea numérica, convirtiendo los valores no numéricos a NaN
df['Presion_Arterial'] = pd.to_numeric(df['Presion_Arterial'], errors='coerce') # Asegurar que la columna 'Presion_Arterial' sea numérica, convirtiendo los valores no numéricos a NaN
df['Glucosa'] = pd.to_numeric(df['Glucosa'], downcast='integer', errors='coerce') # Convertir a entero, convirtiendo los valores no numéricos a NaN
df['Edad'] = pd.to_numeric(df['Edad'], downcast='integer', errors='coerce')

df.dropna(inplace=True) # Eliminar filas con valores no numéricos después de la conversión

df.to_clipboard() # Copia el DataFrame limpio al portapapeles para pegarlo 

# ----------------------------------------------------------------
# Fase de modelado de regresión lineal

# Crea la matriz X sin la variable a predecir
# axis = 1 indica que se eliminará la columna 'Glucosa' (si fuera axis=0 se eliminaría una fila)
X = df.drop(['Glucosa'], axis=1) # Variable independiente
y = df['Glucosa'] # Variable dependiente
slr = LinearRegression() # Crear el objeto del modelo
# Entrenar el modelo
# Estudia la relación entre X y y
slr.fit(X,y)

y_pred = slr.predict(X) # Adivina los valores de glucosa basándose en lo aprendido por el modelo

# ----------------------------------------------------------------
# Fase de visualización de resultados

plt.figure(figsize=(12,6)) # Establecer el tamaño de la figura para la gráfica
plt.scatter(y, y_pred, color='lightblue') # Mapear predicciones, nube de puntos
# Línea de referencia, diagonal de 45 grados, para comparar la realidad con las predicciones
# Si se encuentra en la línea, el modelo predice perfectamente, si está por debajo o por encima, el modelo se equivoca
plt.plot([y.min(), y.max()], [y.min(), y.max()], color = "black", lw=2) 

# Etiquetas y título de la gráfica
plt.xlabel("Glucosa real")
plt.ylabel("Glucosa estimada")
plt.title("Glucosa real vs estimada")
plt.show()