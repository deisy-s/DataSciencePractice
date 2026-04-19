# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Las series multivariadas contienen múltiples variables que pueden influir entre sí
# Se muestra cómo graficar varias variables a lo largo del tiempo para analizar su 
# comportamiento conjunto

# LIBRERÍAS ------------------------------------------------------------------------------------------------
# Importar las librerías necesarias
import pandas as pd # Para manipulación de datos
import matplotlib.pyplot as plt # Para graficación

# Cargar archivo CSV 
df = pd.read_csv("Documents/SerieMultivariada.csv") 

# GRAFICACIÓN ----------------------------------------------------------------------------------------------
# Graficar varias variables a lo largo del tiempo 
plt.figure(figsize=(8, 4.8)) # Configurar el tamaño de la figura

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["Mes"], df["Ventas"], marker="o", label="Ventas") 
plt.plot(df["Mes"], df["Temperatura"], marker="s", label="Temperatura") 
plt.plot(df["Mes"], df["Lluvia_mm"], marker="^", label="Lluvia (mm)") 

plt.title("Serie de tiempo multivariada: Variables mensuales") # Agregar título al gráfico

plt.xlabel("Mes") # Agregar etiqueta al eje x
plt.ylabel("Valor") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

plt.grid(True, alpha=0.3) # Agregar una cuadrícula al fondo del gráfico con transparencia

plt.tight_layout()
plt.show() 