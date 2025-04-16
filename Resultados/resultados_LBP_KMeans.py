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

# Rutas y parámetros (ajusta según tu estructura)
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"  # Carpeta raíz donde cada subcarpeta representa una etiqueta
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_KMeans.pkl"  # Ruta del modelo entrenado basado en KMeans
IMAGE_SIZE = (512, 512)          # Debe coincidir con el tamaño usado en entrenamiento
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
      - Calcula el histograma normalizado del descriptor LBP.
    """
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# ========================= Cargar el Modelo KMeans =========================
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
    print("Modelo KMeans cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# ========================= Obtener las Etiquetas =========================
# Se asume que cada subcarpeta en ROOT_DIR corresponde a una clase (etiqueta)
classes = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
print("Clases detectadas:", classes)

# ========================= Evaluación por Subcarpeta =========================
results = {}
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_images = len(all_images)
    if total_images == 0:
        continue  # Saltar subcarpetas sin imágenes

    # Seleccionar una muestra aleatoria de las imágenes (al menos 1)
    sample_size = max(1, int(total_images * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total = 0
    correct = 0

    for img_name in sampled_images:
        img_path = os.path.join(label_dir, img_name)
        features = compute_lbp(img_path)
        if features is None:
            continue  # Saltar si hay error en el procesamiento
        # La predicción del clasificador KMeans es un número (la etiqueta asignada al clúster)
        features = features.reshape(1, -1)
        pred_idx = clf.predict(features)[0]

        # Se mapea el índice predicho al nombre de la etiqueta
        predicted_label = classes[pred_idx] if pred_idx < len(classes) else "Desconocido"
        total += 1
        if predicted_label == label:
            correct += 1

    if total == 0:
        continue

    accuracy = (correct / total) * 100
    results[label] = {
        "Total Imagenes Evaluadas": total,
        "Aciertos": correct,
        "Accuracy (%)": accuracy
    }
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"  Total imágenes evaluadas: {total}")
    print(f"  Aciertos: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")

# ========================= Resumen y Exportación de Resultados =========================
# Convertir los resultados a un DataFrame para un análisis ordenado
df = pd.DataFrame.from_dict(results, orient="index")
df = df.sort_values(by="Accuracy (%)", ascending=True)

print("\nTabla de resultados por etiqueta (ordenadas por Accuracy):")
print(df.to_string())

# Identificar la subcarpeta con peor desempeño (el primer registro del DataFrame ordenado)
worst_label = df.index[0]
print("\nLa subcarpeta con peor desempeño es:")
print(f"  Subcarpeta: {worst_label}")
print(df.loc[worst_label].to_string())

# Exportar el DataFrame a un archivo CSV
csv_output_path = "resultados_prediccion_lbp_KMeans.csv"
df.to_csv(csv_output_path)
print(f"\nTabla de resultados guardada en: {csv_output_path}")
print("Fin del script.")
