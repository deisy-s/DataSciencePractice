# Decision Tree para Diabetes
import pandas as pd # Maneja la tabla de datos
#from sklearn.metrics import accuracy_score # Evaluar el modelo
from sklearn.model_selection import train_test_split # Dividir datos en entrenamiento y testeo
from sklearn.tree import DecisionTreeClassifier # Modelo de árbol de decisión
from sklearn.ensemble import RandomForestClassifier # Modelo de Random Forest
from sklearn.metrics import classification_report # Evaluar el modelo
from sklearn.tree import plot_tree # Visualizar el árbol
import matplotlib.pyplot as plt # Crear la gráfica

# Cargar datos
df = pd.read_csv("Documents/diabetes.csv")

# Separar la variable dependiente
X = df.drop("Outcome", axis=1) # Drop la variable dependiente del DataFrame para obtener las variables independientes
y = df["Outcome"] # Variable dependiente, que es la que queremos predecir (si el paciente es diabético o no)

# Dividir datos en entrenamiento y testeo
# test_size=0.3 significa que el 30% de los datos se usarán para testeo y el 70% para entrenamiento
# random_state=40 asegura que la división de los datos sea reproducible, es decir, que cada vez que se ejecute el código con el mismo random_state, se obtendrá la misma división de los datos.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)

# # Crear modelo y entrenarlo (árbol de decisión)
# modelo = DecisionTreeClassifier(max_depth=3) # max_depth=3 limita la profundidad del árbol a 3 niveles, lo que ayuda a prevenir el sobreajuste al evitar que el árbol se vuelva demasiado complejo.
# # Si el max_depth es muy bajo, es un modelo demasiado sencillo y no captura la complejidad de los datos, lo que se llama subajuste. Si el max_depth es muy alto, el modelo puede capturar demasiado ruido en los datos de entrenamiento, lo que se llama sobreajuste.
# modelo.fit(X_train, y_train) # Entrenar el modelo

# # Evaluar
# y_pred = modelo.predict(X_test) # Predicciones
# print(classification_report(y_test,y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred)) # accuracy_score calcula la proporción de predicciones correctas sobre el total de predicciones realizadas, proporcionando una medida de la precisión general del modelo.

# # Gráfica
# # Visualizar arbol
# plt.figure(figsize=(12,6))
# plot_tree(modelo, feature_names=X.columns, class_names=["No Diabético", "Diabético"],filled=True)
# plt.show()

# Crear modelo y entrenarlo (Random Forest)
# n_estimators=100 significa que el modelo de Random Forest se construirá utilizando 100 árboles de decisión individuales. Cada árbol se entrenará con una muestra aleatoria de los datos de entrenamiento, y el resultado final del modelo se basará en la agregación de las predicciones de todos los árboles.
# Si se sube, el modelo suele ser más robusto y preciso, pero también puede ser más lento de entrenar y predecir. Si se baja, el modelo puede ser más rápido, pero también puede ser menos preciso.
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)
modelo_rf.fit(X_train, y_train) # Entrenar el modelo

# Predicciones y Evaluación
rf_pred = modelo_rf.predict(X_test) # Predicciones
print(classification_report(y_test, rf_pred)) # classification_report genera un informe de clasificación que incluye métricas como precisión, recall, f1-score y soporte para cada clase, lo que ayuda a evaluar el rendimiento del modelo en la tarea de clasificación.

# Visualizar la gráfica
plt.figure(figsize=(12, 6))
# Dibuja el arbol
plot_tree(modelo_rf.estimators_[0], feature_names=X.columns, class_names=["No Diabético", "Diabético"], filled=True, fontsize=10)
plt.show()