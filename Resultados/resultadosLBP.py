import os
import random
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore

# =============================== Configuraciones ===============================
# Ruta a la carpeta raíz con subcarpetas (cada una es una etiqueta)
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
# Ruta del modelo (RandomForest entrenado con LBP) guardado
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp.pkl"
# Tamaño deseado para las imágenes (debe coincidir con el usado en entrenamiento)
IMAGE_SIZE = (512, 512)
# Porcentaje de imágenes a evaluar por subcarpeta
SAMPLE_PERCENT = 0.2

# Parámetros para LBP
RADIUS = 3             # Radio del LBP
N_POINTS = 8 * RADIUS  # Número de puntos alrededor del pixel central
METHOD = 'uniform'     # Método de cálculo del LBP

# =============================== Función para extraer LBP ===============================
def compute_lbp(image_path):
    """
    Calcula el descriptor LBP para una imagen dada:
      - Abre la imagen en escala de grises
      - La redimensiona al tamaño definido
      - Calcula el patrón binario local (LBP) y su histograma normalizado
    """
    try:
        image = Image.open(image_path).convert("L")  # Convertir a escala de grises
        image = image.resize(IMAGE_SIZE)             # Redimensionar
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Se define un número de bins para el histograma (el +3 se usa para abarcar todos los valores)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# =============================== Cargar el Modelo ===============================
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# =============================== Obtener Clases ===============================
# Se asume que cada subcarpeta de ROOT_DIR corresponde a una etiqueta
classes = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
print("Clases detectadas:", classes)

# =============================== Evaluación por Carpeta ===============================
# Diccionario para almacenar resultados por etiqueta
results = {}

# Extensiones válidas para imágenes
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Recorrer cada subcarpeta (etiqueta)
for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_imagenes_en_carpeta = len(all_images)
    if total_imagenes_en_carpeta == 0:
        continue  # Saltar carpetas sin imágenes

    # Calcular el tamaño de la muestra (al menos 1 imagen)
    sample_size = max(1, int(total_imagenes_en_carpeta * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total_images = 0
    correct_predictions = 0
    confidences = []

    # Procesar cada imagen de la muestra
    for img_name in sampled_images:
        img_path = os.path.join(label_dir, img_name)
        features = compute_lbp(img_path)
        if features is None:
            continue  # Saltar si hubo error al procesar la imagen
        # Convertir la característica en una matriz 2D para la predicción
        features = features.reshape(1, -1)
        # Realizar la predicción (predicción y probabilidad)
        pred_idx = clf.predict(features)[0]
        prob_array = clf.predict_proba(features)[0]
        confidence = np.max(prob_array)
        # Mapeo del índice predicho a la etiqueta (suponemos el mismo orden que en 'classes')
        pred_label = classes[pred_idx] if pred_idx < len(classes) else "Desconocido"
        total_images += 1
        if pred_label == label:
            correct_predictions += 1
        confidences.append(confidence)

    if total_images == 0:
        continue

    # Calcular las métricas para la etiqueta actual
    accuracy = (correct_predictions / total_images) * 100
    avg_confidence = (sum(confidences) / len(confidences)) * 100
    results[label] = {
        "Total Imagenes Evaluadas": total_images,
        "Aciertos": correct_predictions,
        "Accuracy (%)": accuracy,
        "Confianza Promedio (%)": avg_confidence
    }
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"  Total imágenes evaluadas: {total_images}")
    print(f"  Aciertos: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Confianza Promedio: {avg_confidence:.2f}%")

# =============================== Resumen de Resultados ===============================
# Convertir los resultados a un DataFrame para un fácil análisis y ordenarlos por accuracy (de menor a mayor)
df = pd.DataFrame.from_dict(results, orient="index")
df = df.sort_values(by="Accuracy (%)", ascending=True)

print("\nTabla de resultados por etiqueta (ordenadas por Accuracy):")
print(df.to_string())

# Identificar la subcarpeta con el peor desempeño (primer registro del DataFrame ordenado)
worst_label = df.index[0]
worst_metrics = df.loc[worst_label]
print("\nLa subcarpeta con peor desempeño es:")
print(f"  Subcarpeta: {worst_label}")
print(worst_metrics.to_string())

# =============================== Exportar Resultados ===============================
csv_output_path = "resultados_prediccion_lbp_rf.csv"
df.to_csv(csv_output_path)
print(f"\nTabla de resultados guardada en: {csv_output_path}")
print("Fin del script.")