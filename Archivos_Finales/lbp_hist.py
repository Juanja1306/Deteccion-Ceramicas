import os
import json
import warnings
import numpy as np
from json import JSONDecodeError
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Parámetros para LBP
RADIUS = 3             # Radio del LBP
N_POINTS = 8 * RADIUS  # Número de puntos alrededor del pixel central
METHOD = 'uniform'     # Método de cálculo del LBP
IMAGE_SIZE = (512, 512)  # Tamaño deseado para las imágenes

# Función para calcular LBP de una imagen
def compute_lbp(image_path):
    """
    Calcula y retorna el histograma normalizado del descriptor LBP de una imagen.
    """
    try:
        # Abrir la imagen en escala de grises y redimensionar
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None
    # Convertir la imagen a matriz numpy y calcular LBP
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Calcular histograma del LBP con número de bins adecuado
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# Función para cargar el dataset completo
def load_dataset(root_dir, json_file="datasetLBP.json"):
    
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
        except (JSONDecodeError, KeyError):
            print("⚠️ JSON inválido, regenerando desde las imágenes…")
            
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    features = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                features.append(compute_lbp(img_path))
                labels.append(class_to_idx[cls])
    
    features = np.array(features)
    labels = np.array(labels)       
                
    # Guardar los datos preprocesados en un archivo JSON
    data = {
        "features": features.tolist(),  # Convertir np.array a lista para poder serializar en JSON
        "labels": labels.tolist()
    }
    with open(json_file, "w") as f:
        json.dump(data, f)
    print("Datos preprocesados guardados en:", json_file)
    
    return features, labels