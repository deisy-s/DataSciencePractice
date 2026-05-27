import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder

# Carga de datos
df = pd.read_csv("Documents/seattle-weather.csv")
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe())

# Codificar la variable categórica 'weather'
le = LabelEncoder()
df["weather"] = le.fit_transform(df["weather"])
df["date"] = pd.to_datetime(df["date"])
print(df.head()) 
print("\nEstadistica descriptiva") 
print(df.describe())

# -------------------------------------------------------------------------------------
# Serie de tiempo
# -------------------------------------------------------------------------------------

# Precipitación
plt.figure(figsize=(12,6))

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["date"], df["precipitation"], marker="o", label="Precipitación") 

plt.title("Serie de tiempo: Clima en Seattle") # Agregar título al gráfico

plt.xlabel("Fecha") # Agregar etiqueta al eje x
plt.ylabel("Precipitación") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

# Configurar los ticks del eje x para mostrar el inicio de cada año y cada tercer mes
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1,4,7,10)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center")
plt.tick_params(axis='x', rotation=45)

plt.grid(True, alpha=0.3)

plt.show() 

# Temperatura máxima
plt.figure(figsize=(12,6))

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["date"], df["temp_max"], marker="*", label="Temperatura Máxima") 

plt.title("Serie de tiempo: Clima en Seattle") # Agregar título al gráfico

plt.xlabel("Fecha") # Agregar etiqueta al eje x
plt.ylabel("Temperatura Máxima") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1,4,7,10)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center")
plt.tick_params(axis='x', rotation=45)

plt.grid(True, alpha=0.3)

plt.show() 

# Temperatura mínima
plt.figure(figsize=(12,6))

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["date"], df["temp_min"], marker="*", label="Temperatura Mínima") 

plt.title("Serie de tiempo: Clima en Seattle") # Agregar título al gráfico

plt.xlabel("Fecha") # Agregar etiqueta al eje x
plt.ylabel("Temperatura Mínima") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1,4,7,10)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center")
plt.tick_params(axis='x', rotation=45)

plt.grid(True, alpha=0.3)

plt.show() 

# Viento
plt.figure(figsize=(12,6))

# Graficar cada variable con un marcador diferente y una etiqueta para la leyenda
plt.plot(df["date"], df["wind"], marker="o", label="Viento") 

plt.title("Serie de tiempo: Clima en Seattle") # Agregar título al gráfico

plt.xlabel("Fecha") # Agregar etiqueta al eje x
plt.ylabel("Viento") # Agregar etiqueta al eje y

plt.legend() # Agregar una leyenda para identificar cada variable

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_minor_locator(mdates.MonthLocator((1,4,7,10)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center")
plt.tick_params(axis='x', rotation=45)

plt.grid(True, alpha=0.3)

plt.show() 

# -------------------------------------------------------------------------------------
# Correlación
# -------------------------------------------------------------------------------------
# Crear lags features (valores rezagados)
df["Weather_lag1"] = df["weather"].shift(1) 
df["Weather_lag2"] = df["weather"].shift(2) 
df["Weather_lag3"] = df["weather"].shift(3) 

print("\nCorrelación:") 
print(df[["weather", "Weather_lag1", "Weather_lag2", "Weather_lag3"]].corr()) 

 