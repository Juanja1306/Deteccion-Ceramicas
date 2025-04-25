import os
import joblib
import numpy as np
from PIL import Image
import lbp_hist as lbp

# =============================== Configuraciones ===============================
# Ruta a la carpeta raíz con subcarpetas (cada una es una etiqueta)
# Ruta del modelo guardado
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_RFC_FINAL.pkl"
IMAGE_PATH = r"C:\Users\juanj\Pictures\florencia2.jpg"

# =============================== Cargar el Modelo ===============================
if os.path.exists(MODEL_PATH):
    size = os.path.getsize(MODEL_PATH)
    print(f"Tamaño del archivo de modelo: {size} bytes")
    if size == 0:
        raise IOError("El archivo de modelo está vacío")
    # ← Aquí cargas realmente el modelo en `clf`
    clf = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

classes = sorted(os.listdir(ROOT_DIR))
# print("Clases encontradas:", classes, "\n")
# print(f"Total de clases: {len(classes)}\n")

# =============================== Función de Predicción para una Imagen ===============================
def predict_image_top5(image_path):
    """
    Realiza la predicción de la clase para una imagen y retorna una lista con los 5 resultados
    más probables ordenados de mayor a menor probabilidad.
    
    Retorna:
      - Una lista de tuplas (etiqueta, probabilidad)
        donde el primer elemento es la predicción con mayor probabilidad y los siguientes son los 4 que le siguen.
    """
    features = lbp.compute_lbp(image_path)
    if features is None:
        print("No se pudo extraer las características de la imagen.")
        return None
    # Convertir el vector de características a formato 2D para la predicción
    features = features.reshape(1, -1)
    # Obtener el array de probabilidades para cada clase
    prob_array = clf.predict_proba(features)[0]
    # Ordenar los índices de mayor a menor probabilidad y obtener los 5 primeros
    top5_indices = np.argsort(prob_array)[::-1][:5]
    # Mapear cada índice a la etiqueta y su probabilidad
    top5 = [(classes[idx], prob_array[idx]) for idx in top5_indices]
    return top5

# ================================= Predecir ===============================================
def predict():
    if not os.path.exists(IMAGE_PATH):
        print("La ruta especificada no existe.")
    else:
        results = predict_image_top5(IMAGE_PATH)
        if results is not None:
            print("\nTop-5 Predicciones:")
            # Imprime la predicción principal y luego las 4 siguientes
            print(f"Mejor coincidencia: {results[0][0]} con una confianza de {results[0][1]*100:.2f}%")
            print("Otras coincidencias:")
            for label, conf in results[1:]:
                print(f"  {label}: {conf*100:.2f}%")
        else:
            print("No se pudo realizar la predicción.")

# ================================== Main ===============================================
if __name__ == "__main__":
    predict()