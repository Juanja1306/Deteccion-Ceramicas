import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Configuraciones
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"  # Directorio raíz con subcarpetas (labels)
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\checkpoint_fold1_latest.pth"  # Ruta del modelo/checkpoint guardado
IMAGE_SIZE = (512, 512)  # Mismo tamaño que en entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_PERCENT = 0.2  # Porcentaje de imágenes a evaluar por subcarpeta

# Cargar el checkpoint para obtener el número de clases
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    checkpoint_state = checkpoint['model_state_dict']
else:
    checkpoint_state = checkpoint

# Extraer el número de clases a partir de la forma de la capa final (fc)
saved_num_classes = checkpoint_state['fc.weight'].shape[0]
print("Número de clases en el checkpoint:", saved_num_classes)

# Obtener las clases definidas en ROOT_DIR
classes = sorted(os.listdir(ROOT_DIR))
if len(classes) != saved_num_classes:
    print("Advertencia: el número de subcarpetas en ROOT_DIR ({}) no coincide con el número de clases en el checkpoint ({}).".format(len(classes), saved_num_classes))
    # Aquí podrías cargar la lista original de clases desde un archivo si la tuvieras.

# Para la reconstrucción del modelo se usará el número de clases del checkpoint
NUM_CLASSES = saved_num_classes

# Definir las mismas transformaciones que en entrenamiento
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convertir a RGB
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Reconstruir el modelo (ResNet50) con la misma arquitectura
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Cargar los pesos desde el checkpoint
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()  # Establecer el modelo en modo evaluación

def predict_image_top1(image_path):
    """
    Realiza la predicción top1 para una imagen y retorna la etiqueta predicha y la confianza.
    """
    try:
        image = Image.open(image_path).convert("L")
    except Exception as e:
        print(f"Error abriendo la imagen {image_path}: {e}")
        return None, None
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    # Obtener el valor y el índice con mayor probabilidad
    top_prob, top_idx = torch.max(probabilities, dim=1)
    top_prob = top_prob.item()
    top_idx = top_idx.item()
    predicted_label = classes[top_idx] if top_idx < len(classes) else "Desconocido"
    return predicted_label, top_prob

# Diccionario para almacenar los resultados por label
results = {}

# Definir extensiones válidas para imagenes
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# Recorrer cada subcarpeta (label)
for label in classes:
    label_dir = os.path.join(ROOT_DIR, label)
    if not os.path.isdir(label_dir):
        continue  # Saltar si no es directorio

    # Obtener lista de imágenes válidas en la subcarpeta
    all_images = [img for img in os.listdir(label_dir) if img.lower().endswith(valid_extensions)]
    total_images_en_carpeta = len(all_images)
    if total_images_en_carpeta == 0:
        continue

    # Calcular el 20% (al menos 1 imagen)
    sample_size = max(1, int(total_images_en_carpeta * SAMPLE_PERCENT))
    sampled_images = random.sample(all_images, sample_size)

    total_images = 0
    correct_predictions = 0
    confidences = []
    
    # Iterar sobre la muestra de imágenes
    for image_name in sampled_images:
        image_path = os.path.join(label_dir, image_name)
        total_images += 1
        pred_label, conf = predict_image_top1(image_path)
        if pred_label is None:
            continue
        confidences.append(conf)
        if pred_label == label:
            correct_predictions += 1
    
    # Calcular métricas para el label
    accuracy = (correct_predictions / total_images) * 100
    avg_confidence = (sum(confidences) / len(confidences)) * 100  # Convertir a porcentaje
    
    results[label] = {
        "Total Imagenes Evaluadas": total_images,
        "Aciertos": correct_predictions,
        "Accuracy (%)": accuracy,
        "Confianza Promedio (%)": avg_confidence
    }
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"Total imágenes evaluadas: {total_images}")

# Convertir resultados a DataFrame y ordenar por accuracy (de menor a mayor)
df = pd.DataFrame.from_dict(results, orient="index")
df = df.sort_values(by="Accuracy (%)", ascending=True)

print("\nTabla de labels con peores resultados (ordenados por Accuracy):")
print(df.to_string())

# Imprimir la subcarpeta con peor desempeño (el primer registro del DataFrame ordenado)
worst_label = df.index[0]
worst_metrics = df.loc[worst_label]
print("\nLa subcarpeta con peor desempeño es:")
print(f"Subcarpeta: {worst_label}")
print(worst_metrics.to_string())

# Opcional: guardar la tabla en un archivo CSV
df.to_csv("tabla_peores_resultados_20porciento.csv")
