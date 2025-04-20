import pandas as pd

# =============================== Cargar CSV y Calcular Promedio ===============================
# Cambia la ruta a la ubicación de tu archivo CSV
PATH = r'C:\Users\juanj\Desktop\Deteccion-Ceramicas\Resultados\resultados_prediccion_lbp_nn_TensorFlow.csv'
OUTPUT_PATH = 'resultados_prediccion_lbp_nn_TensorFlow.csv'

# Cargar el CSV. Cambia 'tu_archivo_filtrado.csv' por el nombre real de tu archivo.
df = pd.read_csv(PATH)

# Asegurarse de que la primera columna tenga el nombre vacío
nombres = list(df.columns)
nombres[0] = ""
df.columns = nombres

# Calcular el promedio para cada columna (excepto la primera) y redondearlo a 2 decimales;
# para la primera columna se asigna "PROMEDIO"
fila_promedio = ["PROMEDIO"]
for i in range(1, len(df.columns)):
    promedio = df.iloc[:, i].mean()
    fila_promedio.append(round(promedio, 2))

# Crear un DataFrame con la fila de promedios
df_promedio = pd.DataFrame([fila_promedio], columns=df.columns)

# Insertar la fila de promedios al comienzo del DataFrame original
df_resultado = pd.concat([df_promedio, df], ignore_index=True)

# Guardar el DataFrame resultante en un nuevo CSV
df_resultado.to_csv(OUTPUT_PATH, index=False)
