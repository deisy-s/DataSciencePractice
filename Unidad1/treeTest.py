import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report 
from sklearn.tree import plot_tree 

df = pd.read_csv('DatasetVentas.csv')

# limpieza
df.dropna(inplace=True)

le_campana = LabelEncoder()
df['Campana_Marketing'] = le_campana.fit_transform(df['Campana_Marketing'])
le_ciudad = LabelEncoder()
df['Ciudad'] = le_ciudad.fit_transform(df['Ciudad'])
le_producto = LabelEncoder()
df['Producto'] = le_producto.fit_transform(df['Producto'])

df['Ventas'] = pd.to_numeric(df['Ventas'], errors='coerce')

# Evaluar si ventas >= 1000
def classify_sale(sale):
    return 1 if sale >= 1000 else 0

df['Result'] = df['Ventas'].apply(classify_sale) # Variable nueva que identifica si ventas es >= a 1000

df.to_clipboard()

# modelo
X = df.drop(['Result', 'Ventas', 'Campana_Marketing'], axis=1)
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

model = RandomForestClassifier(n_estimators=100, random_state=40, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# visualizar resultados
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12,6))

plot_tree(model.estimators_[0], feature_names=X.columns, filled=True, fontsize=10)
plt.show()