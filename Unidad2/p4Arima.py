# EXPLICACIÓN ----------------------------------------------------------------------------------------------
# Se grafica primero una serie de tiempo de ventas x tiempo
# Se aplica un modelo ARIMA para realizar la predicción de ventas futuras

# ARIMA: Es un modelo estadístico para series de tiempo
# AR (AutoRegressive/AutoRegresivo): El valor actual depende de sus valores pasados (lags)
# I (Integrated/Integrado): Se elimina tendencia
# MA (Moving Average/Media móvil): El valor actual depende de los errores pasados del modelo

# ARIMA(order=(p,d,q))
# p: número de términos autoregresivos (lags)
# d: número de diferencias necesarias para hacer la serie estacionaria (eliminar tendencia)
# q: número de términos de media móvil (errores pasados)

# ARIMA hace uso de:
# Series de tiempo: base del análisis, es una secuencia de datos ordenados en el tiempo, cada punto es un valor en un tiempo específico
# Forecasting: Es el proceso de hacer predicciones sobre eventos futuros en base de datos históricos y análisis de tendencias actuales
# Lags: Se utilizan valores pasados de la variable objetivo como características para predecir el valor actual o futuro
# Correlación: mide la relación entre dos variables, valores cercanos a 1 indican alta relación, valores cercanos a 0 indican baja relación
# Autocorrelación: detecta patrones temporales en la serie de tiempo, muestra cómo se relaciona la serie consigo misma en diferentes rezagos (lags)

# LIBRERÍAS ------------------------------------------------------------------------------------------------
# Importar librerías necesarias
import pandas as pd # Para manipulación de datos
import matplotlib.pyplot as plt # Para graficación
from pandas.plotting import autocorrelation_plot # Para graficar la autocorrelación de la serie de tiempo
from statsmodels.tsa.arima.model import ARIMA # Para crear y entrenar el modelo ARIMA
 
# CARGA Y ANÁLISIS DE DATOS --------------------------------------------------------------------------------
# Cargar los datos
df = pd.read_csv("Documents/VentasP3.csv") 

# Crear una variable nueva, Tiempo, que combina el año y mes
df["Tiempo"] = df["Año"] * 12 + df["Mes"] 

# Mostrar las primeras filas del df e información general del dataset 
print(df.head()) 

# GRAFICACIÓN SERIE DE TIEMPO UNIVARIADA -------------------------------------------------------------------
# Graficar la serie de tiempo de ventas
plt.figure(figsize=(10,5)) 

plt.plot(df["Tiempo"], df["Ventas"], marker="o") 

plt.title("Serie de tiempo de ventas") 
plt.xlabel("Tiempo") 
plt.ylabel("Ventas") 
plt.grid() 
plt.show() 
 
# CREAR LAGS FEATURES --------------------------------------------------------------------------------------
# Crear lags features (valores rezagados)
df["Ventas_lag1"] = df["Ventas"].shift(1) # Ventas del mes pasado
df["Ventas_lag2"] = df["Ventas"].shift(2) # Ventas de hace dos meses

# Eliminar nulos
df = df.dropna() 
 
# Imprimir correlación entre ventas y lags
# Valores cercanos a 1 alta relación 
# Valores  cercanos a 0 baja relacion  
# La correlación mide la relación entre dos variables
print("\nCorrelación:") 
print(df[["Ventas", "Ventas_lag1", "Ventas_lag2"]].corr()) 

# Graficar autocorrelación
plt.figure(figsize=(8,5)) 

# La autocorrelación detecta patrones temporales en la serie de tiempo
autocorrelation_plot(df["Ventas"]) 

plt.title("Autocorrelación de ventas") 
plt.show() 
 
# MODELO ARIMA ---------------------------------------------------------------------------------------------

# Usar solo la serie 
serie = df["Ventas"] 

modelo_arima = ARIMA(serie, order=(2,1,2)) # Crear modelo ARIMA 
 
modelo_arima_fit = modelo_arima.fit() # Entrenar el modelo ARIMA con la serie de ventas
 
print(modelo_arima_fit.summary()) # Imprimir resumen del modelo ARIMA entrenado
 
# Forecasting con ARIMA 
# steps=3 indica que se quieren predecir los próximos 3 meses después del último dato disponible
predicciones = modelo_arima_fit.forecast(steps=3) 

# Imprimir las predicciones futuras
print("\nPredicciones futuras:") 
print(predicciones)

# GRAFICACIÓN ----------------------------------------------------------------------------------------------
# Graficar resultados reales y predicciones futuras por ARIMA
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