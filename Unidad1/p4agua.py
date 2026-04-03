import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
df = pd.read_csv("Documents/CalidadAgua.csv")

# Codificar variables
le_fuente = LabelEncoder()
le_temp = LabelEncoder()
le_apta = LabelEncoder()

# Transformar las vars en datos enteros que funcionen para el modelo
df["Fuente"] = le_fuente.fit_transform(df["Fuente"])
df["Temporada"] = le_temp.fit_transform(df["Temporada"])
df["Apta"] = le_apta.fit_transform(df["Apta"])

# Separar vars
X = df.drop("Apta", axis=1)
y = df["Apta"]

# Dividir datos en entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)

# Crear modelo y entrenarlo
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualizar arbol
plt.figure(figsize=(12,6))
plot_tree(modelo, feature_names=X.columns, class_names=le_apta.classes_,filled=True)
plt.show()
