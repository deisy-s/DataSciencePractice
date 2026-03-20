import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

# ----------------------------------------------------------------
# Carga y limpieza inicial de datos

df = pd.read_csv('Documents/Salud.csv') # Cargar los datos del csv en un DataFrame de pandas

df.dropna(inplace=True) # Eliminar vacios
# inplace establece que los cambios se guardan directamente en df sin tener que declarar df=...

# ----------------------------------------------------------------
# Limpieza de texto

df['Fumador'] = df['Fumador'].str.lower()# str.lower() Convertir toda la columna a minúsculas

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

df['Glucosa'] = df['Glucosa'].apply(lambda x: 1 if x > 126 else 0)

df.dropna(inplace=True) # Eliminar filas con valores no numéricos después de la conversión

df.to_clipboard() # Copia el DataFrame limpio al portapapeles para pegarlo 

# ----------------------------------------------------------------
# Fase de classificación con Random Forest

# Crea la matriz X sin la variable a predecir
# axis = 1 indica que se eliminará la columna 'Fumador' (si fuera axis=0 se eliminaría una fila)
X = df.drop('Fumador', axis=1) # Variable independiente
y = df['Fumador'] # Variable dependiente

# Dividir datos en entrenamiento y testeo
# test_size=0.3 significa que el 30% de los datos se usarán para testeo y el 70% para entrenamiento
# random_state=40 asegura que la división de los datos sea reproducible, es decir, que cada vez que 
# se ejecute el código con el mismo random_state, se obtendrá la misma división de los datos.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)

# Crear modelo y entrenarlo
# n_estimators=100 significa que el modelo de Random Forest se construirá utilizando 100 árboles de decisión individuales. 
# Cada árbol se entrenará con una muestra aleatoria de los datos de entrenamiento, y el resultado final del modelo se basará 
# en la agregación de las predicciones de todos los árboles.
# Si se sube, el modelo suele ser más robusto y preciso, pero también puede ser más lento de entrenar y predecir. Si se baja, 
# el modelo puede ser más rápido, pero también puede ser menos preciso.
modelo = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)
modelo.fit(X_train, y_train) # Entrenar el modelo

# Predicciones y Evaluación
pred = modelo.predict(X_test)

# ----------------------------------------------------------------
# Fase de visualización de resultados

# classification_report genera un informe de clasificación que incluye métricas como precisión, recall, f1-score y soporte 
# para cada clase, lo que ayuda a evaluar el rendimiento del modelo en la tarea de clasificación.
print(classification_report(y_test, pred))

plt.figure(figsize=(12, 6)) # Establecer el tamaño de la figura para la gráfica
# Dibuja el arbol
# plot_tree() es una función de scikit-learn que se utiliza para visualizar la estructura de un árbol de decisión.
# En este caso, se está visualizando el primer árbol del modelo de Random Forest (modelo.estimators_[0]), utilizando 
# los nombres de las características (feature_names=X.columns) y llenando los nodos con colores (filled=True) para 
# facilitar la interpretación. El parámetro fontsize=10 establece el tamaño de la fuente para los textos en el gráfico.
plot_tree(modelo.estimators_[0], feature_names=X.columns, filled=True, fontsize=10)
plt.show()