import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from prophet import Prophet

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
df_consumo = pd.read_csv("Documents/consumo_energetico.csv")
df_op = pd.read_csv("Documents/operacion_energia.csv")

# ============================================================
# 2. REVISION Y LIMPIEZA (CONSUMO ENERGETICO)
# ============================================================
print("Consumo Energetico")
print(df_consumo.head())

print("\nTipos de datos:")
print(df_consumo.dtypes)

print("\nValores nulos:")
print(df_consumo.isnull().sum())

# Eliminar filas con valores nulos
df_consumo.dropna(inplace=True)

# Convertir columna Fecha a datetime
df_consumo["Fecha"] = pd.to_datetime(df_consumo["Fecha"])

# ============================================================
# 3. REVISION Y LIMPIEZA (OPERACION ENERGETICA)
# ============================================================
print("\nOperacion Energetica")
print(df_op.head())

print("\nTipos de datos:")
print(df_op.dtypes)

print("\nValores nulos:")
print(df_op.isnull().sum())

# Eliminar filas con valores nulos
df_op.dropna(inplace=True)

# Convertir columna Fecha a datetime
df_op["Fecha"] = pd.to_datetime(df_op["Fecha"])

# ============================================================
# 4. CONSUMO DE API
# ============================================================
url = "https://api.open-meteo.com/v1/forecast?latitude=25.45&longitude=-108.08&daily=temperature_2m_max,precipitation_sum,relative_humidity_2m_mean&timezone=auto"

try:
    # Realizar peticion GET a la API
    response = requests.get(url)

    # Verificar que la respuesta fue exitosa (200)
    response.raise_for_status()

    # Extraer informacion en formato JSON
    datos_json = response.json()["daily"]

    # Convertir JSON a DataFrame
    df_clima = pd.DataFrame(datos_json)

    # Renombrar columnas
    df_clima = df_clima.rename(columns={
        "time": "Fecha",
        "temperature_2m_max": "Temp_Max",
        "precipitation_sum": "Precipitacion",
        "relative_humidity_2m_mean": "Humedad_Relativa"
    })

    # Convertir columna Fecha a datetime
    df_clima["Fecha"] = pd.to_datetime(df_clima["Fecha"])

    # Eliminar valores nulos
    df_clima.dropna(inplace=True)

    print("\nDatos Climaticos (API)")
    print(df_clima.head())

    print("\nTipos de datos:")
    print(df_clima.dtypes)

    # ============================================================
    # 5. INTEGRACION DE DATASETS 
    # ============================================================

    # Unir csvs por fecha
    merge = pd.merge(df_consumo, df_op, on="Fecha", how="inner")

    # Unir el resultado con la api por fecha
    merge = pd.merge(merge, df_clima, on="Fecha", how="left")

    print("\nDataset completo")
    print(merge.head())
    print(f"\nRegistros totales: {len(merge)}")
    print(f"\nColumnas: {list(merge.columns)}")

    # Verificar que no haya nulos tras el merge
    print("\nValores nulos en dataset maestro:")
    print(merge.isnull().sum())

    # Exportar dataset maestro para Power BI
    merge.to_csv("Documents/dataset_maestro.csv", index=False)
    print("\nDataset maestro guardado en Documents/dataset_maestro.csv")

    # ============================================================
    # 6. EXPLORACION DE DATOS
    # ============================================================

    col_consumo = "Consumo_kWh"

    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # Serie del tiempo comportamiento de consumo
    axs[0].plot(merge["Fecha"], merge[col_consumo], color="steelblue", linewidth=1)
    axs[0].set_title("Comportamiento del Consumo Energetico")
    axs[0].set_xlabel("Fecha")
    axs[0].set_ylabel("Consumo (kWh)")

    # Relación entre temperatura y consumo
    axs[1].scatter(merge["Temperatura"], merge[col_consumo], color="tomato", alpha=0.5)
    axs[1].set_title("Relación entre Temperatura y Consumo")
    axs[1].set_xlabel("Temperatura Maxima (C)")
    axs[1].set_ylabel("Consumo (kWh)")

    # Días críticos (consumo mas alto)
    limite = merge[col_consumo].mean() + merge[col_consumo].std()
    dias_criticos = merge[merge[col_consumo] >= limite]

    axs[2].plot(merge["Fecha"], merge[col_consumo], color="steelblue", linewidth=1, label="Consumo")
    axs[2].axhline(limite, color="red", linestyle="--", label=f"Limite critico: {limite:.2f}")
    axs[2].scatter(dias_criticos["Fecha"], dias_criticos[col_consumo], color="red", zorder=5, label="Dias criticos")
    axs[2].set_title("Dias Criticos de Consumo")
    axs[2].set_xlabel("Fecha")
    axs[2].set_ylabel("Consumo (kWh)")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("Models/exploracion_energetica.png") # Guardar la foto
    plt.show()

    print(f"\nDias criticos detectados: {len(dias_criticos)}")
    print(dias_criticos[["Fecha", col_consumo]].to_string(index=False))

    # ============================================================
    # 7. PROPHET
    # ============================================================
    # Preparar dataset con las columnas ds y y
    df_prophet = merge[["Fecha", col_consumo]].rename(columns={
        "Fecha": "ds",
        col_consumo: "y"
    })

    # Crear y entrenar el modelo
    modelo = Prophet()
    modelo.fit(df_prophet)

    # Generar predicciones a 12 meses futuros
    futuro = modelo.make_future_dataframe(periods=12, freq="MS")
    forecast = modelo.predict(futuro)

    print("\nPredicciones Prophet")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12).to_string(index=False))

    # Gráfica de la predicción
    fig_prophet = modelo.plot(forecast)
    fig_prophet.suptitle("Forecasting de Consumo Energetico", fontsize=14)
    plt.savefig("models/forecasting_prophet.png") # Guardar la foto
    plt.show()

    # Gráfica de componentes: tendencia y estacionalidad
    fig_comp = modelo.plot_components(forecast)
    plt.savefig("models/componentes_prophet.png") # Guardar la foto
    plt.show()

    # Exportar predicciones en csv para Power BI
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("dataset_forecast.csv", index=False)
    print("\nPredicciones guardadas en dataset_forecast.csv")

except requests.exceptions.RequestException as e:
    print(f"Error al conectar con la API: {e}")