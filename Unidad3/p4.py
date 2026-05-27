import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from prophet import Prophet

# ============================================================
# CARGA DE DATOS
# ============================================================
df_austin_weather = pd.read_csv("Documents/austin_weather.csv")
df_steel = pd.read_csv("Documents/Steel_industry_data.csv")
df_weather = pd.read_csv("Documents/WeatherDataset_2018.csv")

# ============================================================
# REVISION Y LIMPIEZA (Clima de austin)
# ============================================================
print("Austin Weather")
print(df_austin_weather.head())

print("\nTipos de datos:")
print(df_austin_weather.dtypes)

print("\nValores nulos:")
print(df_austin_weather.isnull().sum())

# Eliminar filas con valores nulos
df_austin_weather.dropna(inplace=True)

# Convertir columna Fecha a datetime
df_austin_weather["Date"] = pd.to_datetime(df_austin_weather["Date"], errors="coerce").dt.normalize()

# ============================================================
# REVISION Y LIMPIEZA (clima general)
# ============================================================
print("\nClima General")
print(df_weather.head())

print("\nTipos de datos:")
print(df_weather.dtypes)

print("\nValores nulos:")
print(df_weather.isnull().sum())

# Eliminar filas con valores nulos
df_weather.dropna(inplace=True)

# Convertir columna Fecha a datetime
df_weather["Date/Time"] = pd.to_datetime(df_weather["Date/Time"], errors="coerce").dt.normalize()
df_weather.rename(columns={"Date/Time": "Date"}, inplace=True)

# ============================================================
# REVISION Y LIMPIEZA (steel industry)
# ============================================================
print("\nAcero")
print(df_steel.head())

print("\nTipos de datos:")
print(df_steel.dtypes)

print("\nValores nulos:")
print(df_steel.isnull().sum())

# Eliminar filas con valores nulos
df_steel.dropna(inplace=True)

# Convertir columna Fecha a datetime
df_steel["date"] = pd.to_datetime(df_steel["date"], dayfirst=True, errors="coerce").dt.normalize()
df_steel.rename(columns={"date": "Date"}, inplace=True)

# este dataset tiene varios registros por cada día, cada 15 segundos
# se agrupan por día sumando el consumo y sus variables para entrenar prophet con el consumo diario total
df_steel_diario = df_steel.groupby("Date", as_index=False).agg({
    "Usage_kWh": "sum",
    "Lagging_Current_Reactive.Power_kVarh": "sum",
    "Leading_Current_Reactive_Power_kVarh": "sum",
    "CO2(tCO2)": "sum"
})

try:
    # ============================================================
    # INTEGRACION DE DATASETS 
    # ============================================================

    # Unir csvs por fecha
    merge = pd.merge(df_weather, df_steel_diario, on="Date", how="inner")
    merge = pd.merge(merge, df_austin_weather, on="Date", how="outer")
    # outer me hace el join sin importar coincidencia de fecha, la mayoria de los registros quedan nulos

    # Realizar alertas
    print("\nCalculando Umbrales y Alertas Matemáticas")

    # Alerta 1: Consumo > límite crítico (es la media + desviación estándar)
    limite_consumo_critico = merge["Usage_kWh"].mean() + (1.5 * merge["Usage_kWh"].std())
    merge["Alerta_Consumo_Critico"] = np.where(merge["Usage_kWh"] >= limite_consumo_critico, 1, 0)

    # Alerta 2: Temperaturas extremas (< 3°C || > 35°C)
    merge["Alerta_Temp_Extrema"] = np.where((merge["Temp_C"] <= 3) | (merge["Temp_C"] >= 35), 1, 0)

    # se define la capacidad max de la fabrica
    capacidad_maxima_planta = merge["Usage_kWh"].max()

    # dataset completo
    print("\nDataset completo")
    print(merge.head())
    print(f"\nRegistros totales: {len(merge)}")
    print(f"\nColumnas: {list(merge.columns)}")

    # Verificar que no haya nulos tras el merge
    print("\nValores nulos en dataset maestro:")
    print(merge.isnull().sum())

    # Exportar dataset para Power BI
    merge.to_csv("dataset_industry_weather.csv", index=False)
    print("\nDataset guardado en dataset_industry_weather.csv")

    # ============================================================
    # EXPLORACION DE DATOS
    # ============================================================

    # modificar este tipo de dato porque pasa como string
    merge["Temp_C"] = pd.to_numeric(merge["Temp_C"], errors='coerce')

    # establecer variable para no repetir
    col_consumo = "Usage_kWh"

    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    # Serie del tiempo comportamiento de consumo
    axs[0].plot(merge["Date"], merge[col_consumo], color="steelblue", linewidth=1)
    axs[0].set_title("Comportamiento del Consumo Energetico")
    axs[0].set_xlabel("Fecha")
    axs[0].set_ylabel("Consumo (kWh)")

    # Relación entre temperatura y consumo
    axs[1].scatter(merge["Temp_C"], merge[col_consumo], color="tomato", alpha=0.5)
    axs[1].set_title("Relación entre Temperatura y Consumo")
    axs[1].set_xlabel("Temperatura (C)")
    axs[1].set_ylabel("Consumo (kWh)")

    # Días críticos (consumo mas alto)
    limite = merge[col_consumo].mean() + merge[col_consumo].std()
    dias_criticos = merge[merge[col_consumo] >= limite]

    axs[2].plot(merge["Date"], merge[col_consumo], color="steelblue", linewidth=1, label="Consumo")
    axs[2].axhline(limite, color="red", linestyle="--", label=f"Limite critico: {limite:.2f}")
    axs[2].scatter(dias_criticos["Date"], dias_criticos[col_consumo], color="red", zorder=5, label="Dias criticos")
    axs[2].set_title("Dias Criticos de Consumo")
    axs[2].set_xlabel("Fecha")
    axs[2].set_ylabel("Consumo (kWh)")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("U3P4/consumo.png") # Guardar la foto
    plt.show()

    # ============================================================
    # PROPHET
    # ============================================================
    print("\nEntrenando Prophet")
    
    # preparar columnas ds y y de Prophet
    df_prophet = merge[["Date", col_consumo]].rename(columns={
        "Date": "ds",
        col_consumo: "y"
    })

    # Eliminar las filas vacías de los años 2013-2017 (predecir para 2018)
    df_prophet = df_prophet.dropna(subset=["y"])

    # Ordenar cronológicamente las fechas (prophet se confunde)
    df_prophet = df_prophet.sort_values(by="ds").reset_index(drop=True)

    # Crear y configurar el modelo
    # no se puede evaluar estacionalidad porque va por dia y semana, son demasiados registros
    modelo = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    # entrenar Prophe
    modelo.fit(df_prophet)

    # Generar predicciones a futuro (por días)
    futuro = modelo.make_future_dataframe(periods=90, freq="D")
    forecast = modelo.predict(futuro)

    print("\nPredicciones Prophet:")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12).to_string(index=False))

    # Alertar
    # Crecimiento Energético: Diferencia de la tendencia día con día
    forecast["Crecimiento_Energetico"] = forecast["trend"].diff().fillna(0)

    # Riesgo de sobrecarga futura
    forecast["Riesgo_Sobrecarga"] = np.where(forecast["yhat_upper"] > (capacidad_maxima_planta * 0.95), 1, 0)

    # Gráfica de la predicción
    fig_prophet = modelo.plot(forecast)
    fig_prophet.suptitle("Forecasting de Consumo Energetico Diario", fontsize=14)
    plt.savefig("U3P4/forecasting_prophet.png") 
    plt.show()

    # Gráfica de componentes: tendencia y estacionalidad semanal
    fig_comp = modelo.plot_components(forecast)
    plt.savefig("U3P4/componentes_prophet.png") 
    plt.show()

    # Exportar predicciones lógicas en csv para Power BI
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "Crecimiento_Energetico", "Riesgo_Sobrecarga"]].to_csv("U3P4/dataset_forecast.csv", index=False)
    print("\nPredicciones exitosas guardadas en U3P4/dataset_forecast.csv")

    # Gráfica de predicciones futuras
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(merge["Date"], merge[col_consumo], color="steelblue", label="Consumo Real")
    axs[0].axhline(limite_consumo_critico, color="red", linestyle="--", label="Límite Crítico")
    axs[0].set_title("Histórico Energético y Umbral de Alertas")
    axs[0].legend()

    modelo.plot(forecast, ax=axs[1])
    axs[1].set_title("Pronóstico")
    plt.tight_layout()
    plt.savefig("U3P4/monitoreo_completo.png")
    plt.show()

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")