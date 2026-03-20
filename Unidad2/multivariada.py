# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivo CSV 
df = pd.read_csv("Documents/SerieMultivariada.csv") 

# Graficar varias variables a lo largo del tiempo 
plt.figure(figsize=(8, 4.8)) 
plt.plot(df["Mes"], df["Ventas"], marker="o", label="Ventas") 
plt.plot(df["Mes"], df["Temperatura"], marker="s", label="Temperatura") 
plt.plot(df["Mes"], df["Lluvia_mm"], marker="^", label="Lluvia (mm)") 
plt.title("Serie de tiempo multivariada: Variables mensuales") 
plt.xlabel("Mes") 
plt.ylabel("Valor") 
plt.legend() 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 