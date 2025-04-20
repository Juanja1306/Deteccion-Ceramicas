import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore
import warnings

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# ========================= CONFIGURACIONES =========================
# Ruta a la carpeta raíz con subcarpetas (cada una es una etiqueta)
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
# Ruta del modelo entrenado (Keras) guardado
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_NN_Tensorflow.h5"
# Tamaño de la imagen (debe coincidir con el usado en entrenamiento)
IMAGE_SIZE = (512, 512)
# Porcentaje de imágenes a evaluar por subcarpeta
SAMPLE_PERCENT = 0.2

# Parámetros para LBP (deben coincidir con los usados en entrenamiento)
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

# ========================= FUNCIÓN PARA CALCULAR LBP =========================
def compute_lbp(image_path):
    """
    Abre la imagen, la convierte a escala de grises, la redimensiona
    y calcula el histograma normalizado del descriptor LBP.
    """
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # El histograma se calcula con (N_POINTS+2) bins
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalización
    return hist

# ========================= CARGA DEL MODELO =========================
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Modelo cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# ========================= OBTENCIÓN DE LAS ETIQUETAS =========================
# Se asume que cada subcarpeta en ROOT_DIR es una etiqueta. Se ordenan alfabéticamente.
classes = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
print("Clases detectadas:", classes)
num_classes = len(classes)

# El input_dim (cantidad de características) es igual a N_POINTS + 2, ya que el histograma usa esos bins.
input_dim = N_POINTS + 2

# ========================= EVALUACIÓN POR SUBCARPETA =========================
results = {}
# Se definen las extensiones válidas para imágenes
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_images = len(all_images)
    if total_images == 0:
        continue  # Si la subcarpeta no tiene imágenes, se omite

    # Seleccionar aleatoriamente una muestra de imágenes (al menos 1)
    sample_size = max(1, int(total_images * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total = 0
    correct = 0
    confidences = []  # Para almacenar la confianza (porcentaje) de cada predicción

    for img_name in sampled_images:
        img_path = os.path.join(label_dir, img_name)
        hist = compute_lbp(img_path)
        if hist is None:
            continue  # Omitir imágenes que den error
        # Convertir el histograma a un arreglo de forma (1, input_dim)
        input_array = np.expand_dims(hist, axis=0)
        # Realizar la predicción: model.predict devuelve un vector de probabilidades
        probabilities = model.predict(input_array, verbose=0)
        # Se obtiene la probabilidad máxima y el índice correspondiente
        confidence = np.max(probabilities)
        predicted_idx = np.argmax(probabilities)
        # Mapear el índice a la etiqueta (se asume mismo orden que en 'classes')
        pred_label = classes[predicted_idx] if predicted_idx < len(classes) else "Desconocido"
        total += 1
        if pred_label == label:
            correct += 1
        confidences.append(confidence * 100)  # Convertir a porcentaje

    if total == 0:
        continue

    accuracy = (correct / total) * 100
    avg_conf = np.mean(confidences)
    results[label] = {
        "Total Imagenes Evaluadas": total,
        "Aciertos": correct,
        "Accuracy (%)": accuracy,
        "Confianza Promedio (%)": avg_conf
    }
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"  Total imágenes evaluadas: {total}")
    print(f"  Aciertos: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Confianza Promedio: {avg_conf:.2f}%")

# ========================= RESUMEN Y EXPORTACIÓN DE RESULTADOS =========================
# Crear un DataFrame a partir de los resultados para facilitar el análisis
df = pd.DataFrame.from_dict(results, orient="index")
# Ordenar la tabla por Accuracy de menor a mayor
df = df.sort_values(by="Accuracy (%)", ascending=True)
print("\nTabla de resultados por etiqueta (ordenadas por Accuracy):")
print(df.to_string())

# Identificar la subcarpeta con peor desempeño (primer registro del DataFrame ordenado)
worst_label = df.index[0]
print("\nLa subcarpeta con peor desempeño es:")
print(f"  Subcarpeta: {worst_label}")
print(df.loc[worst_label].to_string())

# Exportar el DataFrame a un archivo CSV
csv_output_path = "resultados_prediccion_lbp_nn_TensorFlow.csv"
df.to_csv(csv_output_path)
print(f"\nTabla de resultados guardada en: {csv_output_path}")
print("Fin del script.")
