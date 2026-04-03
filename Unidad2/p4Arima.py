# Importar librerías necesarias
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pandas.plotting import autocorrelation_plot 
from statsmodels.tsa.arima.model import ARIMA
 
# Cargar los datos
df = pd.read_csv("Documents/VentasP3.csv") 

df["Tiempo"] = df["Año"] * 12 + df["Mes"] 

# Mostrar las primeras filas del df e información general del dataset 
print(df.head()) 

# Graficar la serie de tiempo de ventas
plt.figure(figsize=(10,5)) 

plt.plot(df["Tiempo"], df["Ventas"], marker="o") 

plt.title("Serie de tiempo de ventas") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.grid() 
plt.show() 
 
# Crear lags features (valores rezagados)
df["Ventas_lag1"] = df["Ventas"].shift(1) # Ventas del mes pasado
df["Ventas_lag2"] = df["Ventas"].shift(2) # Ventas de hace dos meses

# Eliminar nulos
df = df.dropna() 
 
# Imprimir correlación entre ventas y lags
# Valores cercanos a 1 alta relación 
# Valores  cercanos a 0 baja relacion  
print("\nCorrelación:") 
print(df[["Ventas", "Ventas_lag1", "Ventas_lag2"]].corr()) 

# Graficar autocorrelación
plt.figure(figsize=(8,5)) 

autocorrelation_plot(df["Ventas"]) 

plt.title("Autocorrelación de ventas") 
plt.show() 
 
# Modelo ARIMA

# Usar solo la serie 
serie = df["Ventas"] 
 
modelo_arima = ARIMA(serie, order=(2,1,2)) # Crear modelo ARIMA 
 
modelo_arima_fit = modelo_arima.fit() # Entrenar el modelo ARIMA con la serie de ventas
 
print(modelo_arima_fit.summary()) # Imprimir resumen del modelo ARIMA entrenado
 
# Forecasting con ARIMA 
predicciones = modelo_arima_fit.forecast(steps=3) 

print("\nPredicciones futuras:") 
print(predicciones)

# Graficar resultados reales y predicciones futuras
plt.figure(figsize=(10,6)) 

plt.plot(df["Tiempo"], df["Ventas"], label="Histórico")  # Datos reales 
tiempo_futuro = [df["Tiempo"].iloc[-1] + i for i in range(1,4)] # Datos futuros

plt.plot(tiempo_futuro, predicciones, marker="o", linestyle="dashed", label="Forecast ARIMA") 

plt.title("Forecasting con ARIMA") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.legend() 
plt.grid() 
plt.show()