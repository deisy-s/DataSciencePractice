import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('DatasetVentas.csv')

# limpieza
df.dropna(inplace=True)

df['Campana_Marketing'] = df['Campana_Marketing'].str.lower()

le_campana = LabelEncoder()
df['Campana_Marketing'] = le_campana.fit_transform(df['Campana_Marketing'])
le_ciudad = LabelEncoder()
df['Ciudad'] = le_ciudad.fit_transform(df['Ciudad'])
le_producto = LabelEncoder()
df['Producto'] = le_producto.fit_transform(df['Producto'])

df['Ventas'] = pd.to_numeric(df['Ventas'], errors='coerce')
df['Campana_Marketing'] = pd.to_numeric(df['Campana_Marketing'], errors='coerce')
df['Ciudad'] = pd.to_numeric(df['Ciudad'], errors='coerce')
df['Producto'] = pd.to_numeric(df['Producto'], errors='coerce')

df.to_clipboard()

# modelo
X = df[['Producto', 'Precio_Unitario', 'Cantidad_Vendida', 'Descuento_pct','Ciudad']]
y = df['Ventas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# graficar
plt.figure(figsize=(12,6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Estimados')
plt.title("Ventas")
plt.show()