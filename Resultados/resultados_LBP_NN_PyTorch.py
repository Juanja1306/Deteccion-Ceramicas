import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# ========================= Configuraciones Iniciales =========================
# Directorio con las imágenes organizadas en subcarpetas (cada una es una etiqueta)
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"
# Ruta del modelo entrenado (PyTorch)
MODEL_PATH = "mejor_modelo_lbp.pt"
# Tamaño de la imagen (debe coincidir con el usado en entrenamiento)
IMAGE_SIZE = (512, 512)
# Porcentaje de imágenes a evaluar por subcarpeta
SAMPLE_PERCENT = 0.2

# Parámetros para LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

# Configuración del dispositivo (GPU si está disponible; sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo a usar:", device)

# ========================= Función para calcular LBP =========================
def compute_lbp(image_path):
    """
    Abre la imagen, la convierte a escala de grises, la redimensiona y calcula el histograma normalizado del descriptor LBP.
    """
    try:
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Se calcula el histograma. Para obtener el número de bins se usa np.arange(0, N_POINTS+3) lo que produce N_POINTS+2 contenedores.
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ========================= Definición del Modelo =========================
# Se define la misma arquitectura utilizada durante el entrenamiento.
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)      # Primera capa oculta con 128 neuronas
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)             # Segunda capa oculta con 64 neuronas
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)     # Capa de salida con número de neuronas igual al número de clases

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ========================= Obtención de las Etiquetas =========================
# Se asume que cada subcarpeta en ROOT_DIR es una etiqueta. Se ordenan alfabéticamente para mantener la coherencia.
classes = sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
print("Clases detectadas:", classes)
num_classes = len(classes)

# ========================= Determinar input_dim =========================
# La función compute_lbp crea un histograma con (N_POINTS + 2) valores.
input_dim = N_POINTS + 2

# ========================= Cargar el Modelo =========================
model = NeuralNet(input_dim, num_classes).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Modelo cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# ========================= Evaluación por Subcarpeta =========================
results = {}
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_images = len(all_images)
    if total_images == 0:
        continue

    # Seleccionar aleatoriamente una muestra de imágenes (mínimo 1 imagen)
    sample_size = max(1, int(total_images * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total = 0
    correct = 0
    confidences = []

    for img_name in sampled_images:
        img_path = os.path.join(label_dir, img_name)
        hist = compute_lbp(img_path)
        if hist is None:
            continue
        # Convertir el histograma a tensor y prepararlo para la inferencia
        input_tensor = torch.tensor(hist, dtype=torch.float32).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Agregar dimensión de batch

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Convertir las salidas a probabilidades
            confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100  # Expresarlo en porcentaje

        # Mapear el índice predicho a la etiqueta (suponiendo el mismo orden que en 'classes')
        pred_label = classes[predicted_class] if predicted_class < len(classes) else "Desconocido"
        total += 1
        if pred_label == label:
            correct += 1
        confidences.append(confidence)

    if total == 0:
        continue
    accuracy = (correct / total) * 100
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
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

# ========================= Resumen y Exportación de Resultados =========================
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
csv_output_path = "resultados_prediccion_lbp_nn.csv"
df.to_csv(csv_output_path)
print(f"\nTabla de resultados guardada en: {csv_output_path}")
print("Fin del script.")
