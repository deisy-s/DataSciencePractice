# Importar las librerías necesarias
import pandas as pd 
import matplotlib.pyplot as plt 

# Cargar archivo CSV 
df = pd.read_csv("Documents/SerueUnivariada.csv") 

# Graficar una sola variable a lo largo del tiempo 
plt.figure(figsize=(8, 4.5)) 
plt.plot(df["Mes"], df["Ventas"], marker="o") 
plt.title("Serie de tiempo univariada: Ventas mensuales") 
plt.xlabel("Mes") 
plt.ylabel("Ventas") 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 