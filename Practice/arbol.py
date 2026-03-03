import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("Documents/agricultura.csv")

# Codificar variables
le_suelo = LabelEncoder()
le_fert = LabelEncoder()
le_cultivo = LabelEncoder()

# Transformar las vars en datos enteros que funcionen para el modelo
df["Tipo_Suelo"] = le_suelo.fit_transform(df["Tipo_Suelo"])
df["Fertilizante"] = le_fert.fit_transform(df["Fertilizante"])
df["Cultivo"] = le_cultivo.fit_transform(df["Cultivo"])

# Separar vars
X = df.drop("Cultivo", axis=1)
y = df["Cultivo"]

# Dividir datos en entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=40)

# Crear modelo y entrenarlo
modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print(classification_report(y_test,y_pred))

# Visualizar arbol
plt.figure(figsize=(12,6))
plot_tree(modelo, feature_names=X.columns, class_names=le_cultivo.classes_,filled=True)
plt.show()