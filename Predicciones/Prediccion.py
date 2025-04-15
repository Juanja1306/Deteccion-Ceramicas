import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuraciones
ROOT_DIR = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"  # Directorio con las subcarpetas (labels)
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\checkpoint_fold1_latest.pth"  # Ruta del modelo o checkpoint guardado
IMAGE_SIZE = (512, 512)  # Mismo tamaño que en entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Obtener las clases (labels) a partir de las subcarpetas
classes = sorted(os.listdir(ROOT_DIR))
NUM_CLASSES = len(classes)
# print(f"Clases encontradas: {classes}")
# print(f"Número de clases: {NUM_CLASSES}")

# Definir las mismas transformaciones que se aplicaron en entrenamiento
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convertir a RGB para el modelo preentrenado
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Crear el modelo con la misma arquitectura (ResNet50 en este caso)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Cargar el checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
# Si el .pth contiene más información (por ejemplo, 'model_state_dict'), accedemos correctamente
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()  # Poner el modelo en modo evaluación

def predict_image(image_path):
    """
    Función para predecir la clase de una imagen.
    """
    # Abrir imagen en modo escala de grises
    image = Image.open(image_path).convert("L")
    # Aplicar transformaciones
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
    
    # Mapear el índice predicho a la etiqueta (nombre de la subcarpeta)
    predicted_class = classes[predicted.item()]
    return predicted_class


import torch.nn.functional as F

def predict_image_top3(image_path):
    """
    Función para predecir las 3 clases más probables para una imagen
    y retornar su nombre y porcentaje de confianza.
    """
    # Abrir la imagen en modo escala de grises y aplicar transformaciones
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Realizar la predicción y obtener probabilidades
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Obtener los 3 índices con mayor probabilidad
    top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
    
    # Convertir a listas para un manejo más sencillo
    top3_prob = top3_prob.squeeze().tolist()
    top3_idx = top3_idx.squeeze().tolist()
    
    # Mapear cada índice a su etiqueta y calcular el porcentaje
    top3_predictions = [(classes[idx], prob * 100) for idx, prob in zip(top3_idx, top3_prob)]
    return top3_predictions

# Ejemplo de uso para una imagen individual
image_path = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido\ALTAR\LARA_60X60_GRIS_MANTA_NOV22_L7_rot0_blur1.jpg"  # Cambia esta ruta por la de tu imagen
try:
    predictions = predict_image_top3(image_path)
    print(f"Predicciones para la imagen {image_path}:")
    for cls, confidence in predictions:
        print(f"  Clase: {cls}, Confianza: {confidence:.2f}%")
except Exception as e:
    print(f"Error procesando {image_path}: {e}")



# # Ejemplo de predicción para una única imagen
# image_path = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido\AMELIE\AMELIE_BLUE_rot0_crop_rand1.jpg" 
# try:
#     prediction = predict_image(image_path)
#     print(f"Imagen: {image_path} -> Predicción: {prediction}")
# except Exception as e:
#     print(f"Error procesando {image_path}: {e}")

