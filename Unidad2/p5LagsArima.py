# Importar librerías necesarias
import pandas as pd 
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 
from pandas.plotting import autocorrelation_plot 
from statsmodels.tsa.arima.model import ARIMA
 
# Cargar los datos
df = pd.read_csv("Documents/Energia.csv") 

# Mostrar las primeras filas del df e información general del dataset
print("Primeras filas:") 
print(df.head()) 
 
print("\nInformación del dataset:") 
print(df.info())

df["Tiempo"] = df["Año"] * 12 + df["Mes"] 

# Graficar la serie de tiempo univariada
plt.figure(figsize=(8, 4.5)) 
plt.plot(df["Tiempo"], df["Demanda_MW"], marker="o") 
plt.title("Serie de tiempo univariada: Demanda por tiempo") 
plt.xlabel("Tiempo") 
plt.ylabel("Demanda") 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 

# Graficar la serie de tiempo multivariada
plt.figure(figsize=(8, 4.8)) 
plt.plot(df["Tiempo"], df["Temperatura"], marker="o", label="Temperatura") 
plt.plot(df["Tiempo"], df["Demanda_MW"], marker="s", label="Demanda") 
plt.title("Serie de tiempo multivariada: Variables por tiempo") 
plt.xlabel("Tiempo") 
plt.ylabel("Valor") 
plt.legend() 
plt.grid(True, alpha=0.3) 
plt.tight_layout() 
plt.show() 

# Crear lags features (valores rezagados)
# Desplaza una columna una fila hacia abajo. El valor de la fila actual recibe el valor de la fila anterior 
# Por eso aparecen valores vacíos al inicio 
df["Demanda_lag1"] = df["Demanda_MW"].shift(1) # Demanda del mes pasado
df["Demanda_lag2"] = df["Demanda_MW"].shift(2) # Demanda de hace dos meses
 
# Mostrar el dataset con las nuevas columnas de lags
print("\nDataset con lag features:") 
print(df.head(10)) 
 
# Eliminar nulos
df = df.dropna() 

# Imprimir correlación
# Valores cercanos a 1 alta relación 
# Valores  cercanos a 0 baja relacion  
print("\nCorrelación:") 
print(df[["Demanda_MW", "Demanda_lag1", "Demanda_lag2"]].corr()) 

# Graficar autocorrelación
plt.figure(figsize=(8,5)) 

autocorrelation_plot(df["Demanda_MW"]) 

plt.title("Autocorrelación de demanda") 
plt.show() 
 
# Graficar nulos
plt.figure(figsize=(10,5)) 

plt.plot(df["Tiempo"], df["Demanda_MW"], marker="o") 

plt.title("Demanda en el tiempo") 
plt.xlabel("Tiempo") 
plt.ylabel("Demanda") 
plt.grid() 
plt.show() 
 
# Definir las variables para el train test 
X = df[[ 
    "Tiempo", 
    "Temperatura", 
    "Demanda_lag1", 
    "Demanda_lag2" 
]] 
 
y = df["Demanda_MW"] 
 
# Dividir los datos en entrenamiento y prueba, sin mezclar el orden temporal (shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False) 
 
# Modelo lineal con lags
modelo_lineal = LinearRegression() 
modelo_lineal.fit(X_train, y_train) 
pred_lineal = modelo_lineal.predict(X_test) 
 
# Modelo random forest con lags 
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42) 
modelo_rf.fit(X_train, y_train) 
pred_rf = modelo_rf.predict(X_test) 
 
# Evaluar modelos con métricas MSE y R2
def evaluar_modelo(y_real, y_pred, nombre): 
    print(f"\n--- {nombre} ---") 
    print("MSE:", mean_squared_error(y_real, y_pred)) 
    print("R2:", r2_score(y_real, y_pred)) 
 
evaluar_modelo(y_test, pred_lineal, "Regresión Lineal con Lags") 
evaluar_modelo(y_test, pred_rf, "Random Forest con Lags") 
 
# Predicción futura con ambos modelos
futuro = pd.DataFrame({ 
    "Tiempo": [2025*12 + 1], 
    "Temperatura": [25], 
    "Demanda_lag1": [df["Demanda_MW"].iloc[-1]], 
    "Demanda_lag2": [df["Demanda_MW"].iloc[-2]] 
}) 
 
pred_lineal_fut = modelo_lineal.predict(futuro) 
pred_rf_fut = modelo_rf.predict(futuro) 

# Imprimir las predicciones futuras
print("\nPredicción futura:") 
print("Regresión Lineal:", pred_lineal_fut[0]) 
print("Random Forest:", pred_rf_fut[0]) 
 
# Gráfica final con datos reales, predicciones y forecasting futuro
plt.figure(figsize=(12,6)) 
 
plt.plot(df["Tiempo"], df["Demanda_MW"], label="Histórico", color="black") 
plt.plot(X_test["Tiempo"], y_test, label="Real ", marker="o") 
plt.plot(X_test["Tiempo"], pred_lineal, label="Predicción Lineal", linestyle="dashed") 
plt.plot(X_test["Tiempo"], pred_rf, label="Predicción RandomF", linestyle="dashed") 
 
plt.scatter(futuro["Tiempo"], pred_lineal_fut, label="Futuro Lineal", s=120) 
plt.scatter(futuro["Tiempo"], pred_rf_fut, label="Futuro RandomForest", s=120) 
 
plt.title("Demanda reales, predicciones y forecasting futuro") 
plt.xlabel("Tiempo") 
plt.ylabel("Demanda") 
plt.legend() 
plt.grid() 
plt.show()

# ------------------------------------------------------------------
# ARIMA
# ------------------------------------------------------------------

# Usar solo la serie 
serie = df["Demanda_MW"] 
 
modelo_arima = ARIMA(serie, order=(2,1,2)) # Crear modelo ARIMA 
 
modelo_arima_fit = modelo_arima.fit() # Entrenar el modelo ARIMA con la serie de demanda
 
print(modelo_arima_fit.summary()) # Imprimir resumen del modelo ARIMA entrenado
 
# Forecasting con ARIMA 
predicciones = modelo_arima_fit.forecast(steps=3) 

print("\nPredicciones futuras:") 
print(predicciones)

# Graficar resultados reales y predicciones futuras
plt.figure(figsize=(10,6)) 

plt.plot(df["Tiempo"], df["Demanda_MW"], label="Histórico")  # Datos reales 
tiempo_futuro = [df["Tiempo"].iloc[-1] + i for i in range(1,4)] # Datos futuros

plt.plot(tiempo_futuro, predicciones, marker="o", linestyle="dashed", label="Forecast ARIMA") 

plt.title("Forecasting con ARIMA") 
plt.xlabel("Tiempo") 
plt.ylabel("Demanda") 
plt.legend() 
plt.grid() 
plt.show()