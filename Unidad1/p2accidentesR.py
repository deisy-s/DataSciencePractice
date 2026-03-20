import pandas as pd # Maneja la tabla de datos
import statsmodels.formula.api as smf # Modelo estadístico
import matplotlib.pyplot as plt # Crear la gráfica

# Cargar el csv
df = pd.read_csv("Documents/accidentes.csv")

# Filtrar solamente accidentes mortales
df_filter = df[df['accidentes'] == 'accidentes mortales'].copy()

# Agregar la columna de accidentes totales de fin de semana, accidentes totales entre semana, 
# y accidentes de noche (solamente accidentes mortales)
df_filter['accidentes_fin'] = df_filter['sabado'] + df_filter['domingo']
df_filter['accidentes_entre_semana'] = df_filter['lunes'] + df_filter['martes'] + df_filter['miercoles'] + df_filter['jueves'] + df_filter['viernes'] + df_filter['sabado'] + df_filter['domingo']
df_filter['noche'] = df_filter['luz_noche']

# Modelo estilo R
modelo = smf.ols(
    formula="accidentes_fin ~ accidentes_entre_semana + noche", 
    data=df_filter
).fit()

# Reporte estadístico
print(modelo.summary())

# Gráfica
# Obtener predicciones 
df_filter["Estimacion"] = modelo.predict(df_filter)
plt.figure(figsize=(6,6))
plt.scatter(df_filter["accidentes_fin"], df_filter["Estimacion"])
# Línea diagonal de referencia
plt.plot(
    [df_filter["accidentes_fin"].min(), df_filter["accidentes_fin"].max()],
    [df_filter["accidentes_fin"].min(), df_filter["accidentes_fin"].max()],
    color='red'
)

plt.xlabel("Accidentes reales")
plt.ylabel("Accidentes estimadas")
plt.title("Accidentes Fines de Semana de Noche")
plt.show()