import pandas as pd
import numpy as np

# 1. Cargar los archivos originales
df_steel = pd.read_csv("Documents/Steel_industry_data.csv")
df_weather_orig = pd.read_csv("Documents/WeatherDataset.csv")

# Asegurar que las fechas del acero estén en datetime
df_steel["date"] = pd.to_datetime(df_steel["date"], dayfirst=True)

# 2. Extraer la línea de tiempo exacta del acero en 2018
# El dataset de acero tiene registros cada 15 minutos, necesitamos el clima para cada uno
df_weather_2018 = pd.DataFrame({"Date/Time": df_steel["date"]})

# 3. Mapear los patrones estacionales del dataset original (2012) al nuevo (2018)
# Convertimos el dataset original a datetime para extraer sus meses, días y horas
df_weather_orig["Date/Time"] = pd.to_datetime(df_weather_orig["Date/Time"])

# Creamos una columna clave "Mes_Dia_Hora" en el archivo original para poder copiar sus valores
df_weather_orig["Key"] = df_weather_orig["Date/Time"].dt.strftime("%m-%d %H:00")
# Eliminamos duplicados horarios por si acaso la API repitió datos
df_weather_orig_clean = df_weather_orig.drop_duplicates(subset=["Key"])

# 4. Crear la misma clave en el nuevo esqueleto de 2018 (redondeando a la hora más cercana)
df_weather_2018["Key"] = df_weather_2018["Date/Time"].dt.strftime("%m-%d %H:00")

# 5. Fusionar los datos climáticos originales mapeándolos al año 2018
columnas_clima = [
    "Temp_C", "Dew Point Temp_C", "Rel Hum_%", 
    "Wind Speed_km/h", "Visibility_km", "Press_kPa", "Weather"
]

df_weather_2018 = pd.merge(
    df_weather_2018, 
    df_weather_orig_clean[["Key"] + columnas_clima], 
    on="Key", 
    how="left"
)

# 6. Limpieza final: Rellenar pequeños huecos (por desfases de minutos) y remover la clave temporal
df_weather_2018[columnas_clima] = df_weather_2018[columnas_clima].ffill().bfill()
df_weather_2018.drop(columns=["Key"], inplace=True)

# Formatear la fecha de salida idéntica al formato original string
df_weather_2018["Date/Time"] = df_weather_2018["Date/Time"].dt.strftime("%m/%d/%Y %H:%M")

# 7. Guardar el nuevo dataset listo para usar
df_weather_2018.to_csv("Documents/WeatherDataset_2018.csv", index=False)

print("¡Dataset generado con éxito!")
print(f"Total de registros climáticos creados para 2018: {len(df_weather_2018)}")
print(df_weather_2018.head())