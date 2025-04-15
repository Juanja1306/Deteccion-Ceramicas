import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
from torchvision.models import ResNet50_Weights

# Deshabilitar el límite de tamaño de imagen en Pillow
Image.MAX_IMAGE_PIXELS = None

# Configuración inicial
ROOT_DIR = r"D:\DATA FINAL\Ruido"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = (1024, 1024)  # Tamaño deseado para las imágenes
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5  # Número de folds para validación cruzada
MODEL_SAVE_PATH = "mejor_modelo.pth"  # Ruta para guardar el mejor modelo
SAVE_INTERVAL = 100  # Guardar checkpoint cada 100 batches
NUM_WORKERS = 16  # Número de workers para cargar datos en paralelo

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convertir a RGB para compatibilidad con modelos preentrenados
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset personalizado
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Abrir imagen en escala de grises
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Crear dataset completo
full_dataset = CustomImageDataset(root_dir=ROOT_DIR, transform=transform)

# Dividir el dataset usando KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Lista para almacenar accuracies de cada fold
accuracies = []
best_accuracy = 0.0  # Para rastrear el mejor accuracy

# Validación cruzada con KFold
for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")

    # Subconjuntos de entrenamiento y prueba con DataLoader que usan NUM_WORKERS
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=NUM_WORKERS)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=test_subsampler, num_workers=NUM_WORKERS)

    # Calcular el número de batches
    train_size = len(train_idx)  # Número de imágenes en el conjunto de entrenamiento
    test_size = len(test_idx)    # Número de imágenes en el conjunto de prueba

    num_train_batches = (train_size + BATCH_SIZE - 1) // BATCH_SIZE  # Equivalente a math.ceil(train_size / BATCH_SIZE)
    num_test_batches = (test_size + BATCH_SIZE - 1) // BATCH_SIZE    # Equivalente a math.ceil(test_size / BATCH_SIZE)

    print(f"Fold {fold + 1}:")
    print(f"  Tamaño del conjunto de entrenamiento: {train_size}, Número de batches: {num_train_batches}")
    print(f"  Tamaño del conjunto de prueba: {test_size}, Número de batches: {num_test_batches}")