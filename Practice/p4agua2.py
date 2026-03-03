import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

# 1. Cargar datos
df = pd.read_csv("Documents/CalidadAgua.csv")

# 2. Preprocesamiento
le_Fuente = LabelEncoder()
le_Temporada = LabelEncoder()
le_Apta = LabelEncoder()

df["Fuente"] = le_Fuente.fit_transform(df["Fuente"])
df["Temporada"] = le_Temporada.fit_transform(df["Temporada"])
df["Apta"] = le_Apta.fit_transform(df["Apta"])

# 3. Separar características (X) y objetivo (y)
X = df.drop("Apta", axis=1)
y = df["Apta"]

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# 5. Crear y entrenar el modelo
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=40)
modelo_rf.fit(X_train, y_train)

# 6. Predicciones y Evaluación
rf_pred = modelo_rf.predict(X_test)
print(classification_report(y_test, rf_pred))

# 7. Visualizar un árbol del bosque
plt.figure(figsize=(12, 6))
plot_tree(modelo_rf.estimators_[0], 
          feature_names=X.columns, 
          class_names=le_Apta.classes_.astype(str), 
          filled=True, 
          fontsize=10)
plt.show()