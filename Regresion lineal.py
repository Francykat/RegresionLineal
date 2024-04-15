import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Leer el archivo CSV
data = pd.read_csv('datos.csv')

# Definir las variables X e Y
X = data['Variable_Independiente'].values.reshape(-1, 1)
Y = data['Variable_Dependiente'].values

# Crear el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, Y)

# Obtener los coeficientes de la ecuación de la recta
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]

# Imprimir la ecuación de la recta
print(f"La ecuación de la recta es: Y = {intercepto} + {pendiente}*X")

# Graficar los datos y la línea de regresión
plt.scatter(X, Y, color='blue')
plt.plot(X, modelo.predict(X), color='red')
plt.title('Regresión Lineal')
plt.xlabel('Variable Independiente')
plt.ylabel('Variable Dependiente')
plt.show()

# Predecir los valores de Y usando el modelo
Y_pred = modelo.predict(X)

# Calcular los errores
mse = mean_squared_error(Y, Y_pred)
rmse = mean_squared_error(Y, Y_pred, squared=False)
mae = mean_absolute_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print(f"Error cuadrático medio (MSE): {mse}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse}")
print(f"Error absoluto medio (MAE): {mae}")
print(f"Coeficiente de determinación R^2: {r2}")