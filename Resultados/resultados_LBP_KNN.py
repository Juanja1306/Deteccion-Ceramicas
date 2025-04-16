import os
import random
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore
import warnings

# ========================= Configuraciones Iniciales =========================
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Rutas y parámetros (ajusta las rutas según tu estructura)
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"  # Carpeta raíz donde cada subcarpeta es una etiqueta
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_KNN.pkl"  # Modelo KNN previamente entrenado
IMAGE_SIZE = (512, 512)          # Debe coincidir con el usado en entrenamiento
SAMPLE_PERCENT = 0.2             # Porcentaje de imágenes a evaluar por subcarpeta

# Parámetros para LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

# ========================= Función para calcular LBP =========================
def compute_lbp(image_path):
    """
    Calcula el descriptor LBP para una imagen:
      - Abre la imagen en escala de grises.
      - La redimensiona a IMAGE_SIZE.
      - Calcula el patrón binario local y normaliza su histograma.
    """
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Se calcula el histograma extendido (se agrega +3 para cubrir todos los valores)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# ========================= Cargar el Modelo KNN =========================
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
    print("Modelo KNN cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# ========================= Obtener Etiquetas =========================
# Se asume que cada subcarpeta en ROOT_DIR corresponde a una clase
classes = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
print("Clases detectadas:", classes)

# ========================= Evaluación por Carpeta =========================
results = {}
# Definir las extensiones válidas para imágenes
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_imagenes = len(all_images)
    if total_imagenes == 0:
        continue  # Saltar carpetas sin imágenes

    # Calcular el tamaño de muestra (al menos 1 imagen)
    sample_size = max(1, int(total_imagenes * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total_images = 0
    correct_predictions = 0
    confidences = []

    # Procesar cada imagen de la muestra
    for img_name in sampled_images:
        img_path = os.path.join(label_dir, img_name)
        features = compute_lbp(img_path)
        if features is None:
            continue  # Saltar si ocurre un error en la lectura o procesamiento
        features = features.reshape(1, -1)
        # Realizar la predicción y obtener la probabilidad asociada
        pred_idx = clf.predict(features)[0]
        prob_array = clf.predict_proba(features)[0]
        confidence = np.max(prob_array)
        # Mapear el índice predicho a la etiqueta (suponiendo que el orden de clases es el mismo)
        pred_label = classes[pred_idx] if pred_idx < len(classes) else "Desconocido"
        total_images += 1
        if pred_label == label:
            correct_predictions += 1
        confidences.append(confidence)

    if total_images == 0:
        continue

    # Calcular métricas para la subcarpeta actual
    accuracy = (correct_predictions / total_images) * 100
    avg_conf = (sum(confidences) / len(confidences)) * 100
    results[label] = {
         "Total Imagenes Evaluadas": total_images,
         "Aciertos": correct_predictions,
         "Accuracy (%)": accuracy,
         "Confianza Promedio (%)": avg_conf
    }
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"  Total imágenes evaluadas: {total_images}")
    print(f"  Aciertos: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Confianza Promedio: {avg_conf:.2f}%")

# ========================= Resumen y Exportación de Resultados =========================
# Convertir los resultados a un DataFrame para un análisis más fácil, ordenado por accuracy
df = pd.DataFrame.from_dict(results, orient="index")
df = df.sort_values(by="Accuracy (%)", ascending=True)

print("\nTabla de resultados por etiqueta (ordenadas por Accuracy):")
print(df.to_string())

# Identificar la subcarpeta con peor desempeño (primer registro del DataFrame ordenado)
worst_label = df.index[0]
worst_metrics = df.loc[worst_label]
print("\nLa subcarpeta con peor desempeño es:")
print(f"  Subcarpeta: {worst_label}")
print(worst_metrics.to_string())

# Guardar el DataFrame en un archivo CSV
csv_output_path = "resultados_prediccion_lbp_KNN.csv"
df.to_csv(csv_output_path)
print(f"\nTabla de resultados guardada en: {csv_output_path}")
print("Fin del script.")
