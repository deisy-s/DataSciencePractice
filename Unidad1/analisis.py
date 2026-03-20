import pandas as pd
df = pd.read_csv('Documents/Practica1.csv') # Cargar los datos del csv en un DataFrame de pandas
print(df)

# Métodos de agregación
print('Promedio:', df['Calificacion'].mean()) # Calcular el promedio de la columna 'Calificacion'
print('Máxima:', df['Calificacion'].max()) # Calcular el valor máximo de la columna 'Calificacion'
print('Mínima:', df['Calificacion'].min()) # Calcular el valor mínimo de la columna 'Calificacion'

import matplotlib.pyplot as plt

# Crear un gráfico de barras con los nombres en el eje x y las calificaciones en el eje y
# Si se invierten las variables, no funciona de manera correcta el gráfico, ya que intenta
# poner números en el eje x y nombres en el eje y, lo cual no es posible.
plt.bar(df['Nombre'], df['Calificacion']) 
plt.title('Calificaciones de alumnos')
plt.xlabel('Alumnno')
plt.ylabel('Calificacion')
plt.show()