import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score 

# 1. Exploración del dataset
# Carga de datos
df = pd.read_csv("Documents/dataset_agricola.csv")

# Mostrar las primeras filas
print(df.head()) 

# Mostrar el tipo de dato de cada columna
print("\nTipos de datos")
print(df.dtypes)

le = LabelEncoder()
df["Plagas"] = le.fit_transform(df["Plagas"])

# Detectar valores nulos y eliminarlos
df.dropna(inplace=True)

# 2. Análisis estadístico
# Mostrar estadísticas descriptivas de las variables numéricas
print("\nEstadistica descriptiva") 
print(df.describe())

# 3. Visualización de datos
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Serie de tiempo multivariada: Rendimiento
plt.figure(figsize=(8, 4.8)) # Configurar el tamaño de la figura

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["Fecha"], df["Humedad"], marker="o", color="hotpink", label="Humedad") 
plt.plot(df["Fecha"], df["Temperatura"], marker="s", color="indigo", label="Temperatura") 
plt.plot(df["Fecha"], df["Lluvia"], marker="^", color="mediumorchid", label="Lluvia") 
plt.plot(df["Fecha"], df["Rendimiento"], marker="*", color="red", label="Rendimiento") 

plt.title("Serie de tiempo: Rendimiento") # Agregar título al gráfico

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1,6,12)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center")

plt.tick_params(axis='x', rotation=45)

plt.ylabel("Valor") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

plt.grid(True, alpha=0.3) # Agregar una cuadrícula al fondo del gráfico con transparencia

plt.tight_layout()
#plt.show() 

# Histograma del rendimiento
plt.figure(figsize=(12, 6)) # Configurar el tamaño de la figura
plt.hist(df["Rendimiento"], bins=10, color="lightcoral", edgecolor="black")
plt.title("Histograma del rendimiento") # Agregar título al gráfico
plt.xlabel("Rendimiento") # Agregar etiqueta al eje x
plt.ylabel("Frecuencia") # Agregar etiqueta al eje y
#plt.show() 

# 4. Correlación entre variables
print("\nCorrelación:") 
print(df[["Humedad", "Temperatura", "Lluvia", "Rendimiento", "Plagas"]].corr()) 

# 5. Variables inteligentes
# Condiciones
conditions = [
    (df['Humedad'] >= 50) & (df['Temperatura'] <= 25),
    ((df['Humedad'] < 50) & (df['Humedad'] >= 40)) | ((df['Temperatura'] > 25) & (df['Temperatura'] <= 30)),
    (df['Humedad'] < 40) & (df['Temperatura'] > 30)
]

# Resultado de cada condición
values = [0, 1, 2]

# Crear la columna con las condiciones y los valores correspondientes
df['Riesgo'] = np.select(conditions, values, default=1)
print(df.head()) 

# 6. Alertas automáticas
# Condiciones
conditions = [
    (df['Humedad'] < 40),
    (df['Temperatura'] < 20),
    (df['Temperatura'] < 20) & (df['Humedad'] > 40),
    (df['Humedad'] > 60)
]

# Resultado de cada condición
values = [
    "Alerta: Humedad baja",
    "Alerta: Temperatura baja, riesgo de heladas",
    "Alerta: Temperatura baja y humedad alta, riesgo de heladas",
    "Alerta: Demasiada humedad, riesgo de enfermedades"
]

# Crear la columna con las condiciones y los valores correspondientes
df['Alerta'] = np.select(conditions, values, default="Sin alertas")
print(df.head()) 

# 7. Modelo de clasificación
# Variables
X = df[["Temperatura", "Humedad", "Lluvia", "Plagas"]] 
y = df["Riesgo"]

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) # Dividir los datos en datos de entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)

# DecisionTreeClassifier
# Crear modelo y entrenarlo
decision_model = DecisionTreeClassifier(max_depth=3, random_state=42)
decision_model.fit(X_train, y_train)

# Evaluar
y_pred = decision_model.predict(X_test)
print(classification_report(y_test,y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy, "\n")

# Visualizar arbol
plt.figure(figsize=(15,8))
plot_tree(decision_model, feature_names=X.columns, class_names=[str(c) for c in decision_model.classes_], filled=True)
#plt.show()

# RandomForestClassifier
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)
modelo_rf.fit(X_train, y_train)

# Evaluar
rf_pred = modelo_rf.predict(X_test)
print(classification_report(y_test, rf_pred))

accuracy = accuracy_score(y_test, rf_pred)
print("Precisión:", accuracy)

# Visualizar arbol
plt.figure(figsize=(12, 6))
plot_tree(modelo_rf.estimators_[0], feature_names=X.columns, class_names=[str(c) for c in modelo_rf.classes_], filled=True, fontsize=10)
#plt.show()

# RandomForestRegressor
df["Rendimiento_lag1"] = df["Rendimiento"].shift(1) # Rendimiento del mes pasado
df["Rendimiento_lag2"] = df["Rendimiento"].shift(2)
df = df.dropna()
X = df[["Temperatura", "Humedad", "Lluvia", "Plagas", "Rendimiento_lag1", "Rendimiento_lag2"]] 
y = df["Rendimiento"]

X_trainr, X_testr, y_trainr, y_testr = train_test_split(X, y, test_size=0.3, random_state=42) 
modelo_r = RandomForestRegressor(n_estimators=100, random_state=42) 
modelo_r.fit(X_trainr, y_trainr) 
pred_r = modelo_r.predict(X_testr) 

X_testr = X_testr.copy()
X_testr["Prediccion"] = pred_r
X_testr["Real"] = y_testr
#X_testr.to_csv("predicciones.csv", sep=',', encoding='utf-8', index=False, header=True)
 
# Evaluar modelos con métricas MSE y R2
def evaluar_modelo(y_real, y_pred, nombre): 
    print(f"\n--- {nombre} ---") 
    print("MSE:", mean_squared_error(y_real, y_pred)) 
    print("R2:", r2_score(y_real, y_pred)) 

futuro = pd.DataFrame({ 
    "Temperatura": [28],
    "Humedad": [45], 
    "Lluvia": [10],
    "Plagas": [0],
    "Rendimiento_lag1": [df["Rendimiento"].iloc[-1]], 
    "Rendimiento_lag2": [df["Rendimiento"].iloc[-2]] 
}) 
pred_rf_fut = modelo_r.predict(futuro) 

print("\nPredicción futura:") 
print("Random Forest:", pred_rf_fut[0]) 

# Asignar la predicción al DataFrame 'futuro'
futuro["Prediccion_Rendimiento"] = pred_rf_fut

# Guardar en un archivo CSV
futuro.to_csv("prediccion_futura.csv", index=False, encoding='utf-8', sep=',')

# 9. Exportar el dataset y pasar a Power BI
df.to_csv("df_a.csv", sep=',', encoding='utf-8', index=False, header=True)
