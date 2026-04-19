# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Las series univariadas contienen 1 sola variable que se analiza a lo largo del tiempo
# Se muestra cómo graficar 1 sola variable a lo largo del tiempo para analizar su comportamiento

# LIBRERÍAS ------------------------------------------------------------------------------------------------
# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt 

# Cargar archivo CSV 
df = pd.read_csv("Documents/SerueUnivariada.csv") 

# GRAFICACIÓN ----------------------------------------------------------------------------------------------
# Graficar una sola variable a lo largo del tiempo 
plt.figure(figsize=(8, 4.5)) # Configurar el tamaño de la figura

plt.plot(df["Mes"], df["Ventas"], marker="o") # Graficar 1 sola variable con un marcador

plt.title("Serie de tiempo univariada: Ventas mensuales") # Agregar título

plt.xlabel("Mes") # Agregar etiqueta al eje x
plt.ylabel("Ventas") # Agregar etiqueta al eje y

plt.grid(True, alpha=0.3) # Mostrar una cuadrícula de fondo con transparencia

plt.tight_layout() 
plt.show() 