import pandas as pd # Maneja la tabla de datos
import statsmodels.formula.api as smf # Modelo estadístico
import matplotlib.pyplot as plt # Crear la gráfica

# Cargar el csv
df = pd.read_csv("Documents/agricultura.csv")

# Modelo estilo R
# Crea un modelo de mínimos cuadrados ordinarios (OLS) con la fórmula dada y los datos del DataFrame
# El parámetro de fórmula es vital, ya que define la relación entre la variable dependiente 
# (Rendimiento_t_ha) y las variables independientes (Riego_mm, Temperatura_C, Humedad_pct).
modelo = smf.ols(
    formula="Rendimiento_t_ha ~ Riego_mm + Temperatura_C + Humedad_pct", 
    data=df
).fit() # fit() calculos los coeficientes de la ecuación lineal que mejor se ajusta a los datos, minimizando la suma de los errores al cuadrado.

# Reporte estadístico
# sumarry genera el reporte estadístico del modelo, que incluye información sobre los coeficientes de las variables independientes,
# su significancia estadística, el valor de R-cuadrado, entre otros indicadores que ayudan a evaluar la calidad del modelo y la relación entre las variables.
print(modelo.summary())

# Gráfica
# Obtener predicciones 
df["Estimacion"] = modelo.predict(df)
plt.figure(figsize=(6,6))
# Compara la realidad vs estimación
plt.scatter(df["Rendimiento_t_ha"], df["Estimacion"])
# Línea diagonal de referencia
# Esta es una referencia de perfección, entre más cercas los puntos a ella, mejor es el modelo
plt.plot(
    [df["Rendimiento_t_ha"].min(), df["Rendimiento_t_ha"].max()],
    [df["Rendimiento_t_ha"].min(), df["Rendimiento_t_ha"].max()],
    color='red'
)

plt.xlabel("Rendimiento Real (t/ha)")
plt.ylabel("Rendimiento (t/ha)")
plt.title("Regresión Lineal Lineal-Real vs Estimación")
plt.show()