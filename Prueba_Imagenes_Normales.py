import os
import time
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torchvision import models, transforms  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
from torchvision.models import ResNet50_Weights  # type: ignore

Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
BATCH_SIZE =32
NUM_EPOCHS = 15
LEARNING_RATE = 0.002
IMAGE_SIZE = (512, 512)  # Tamaño deseado para las imágenes
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5  # Número de folds para validación cruzada
MODEL_SAVE_PATH = "mejor_modelo.pth"  # Ruta para guardar el mejor modelo
SAVE_INTERVAL = 100  # Guardar checkpoint cada 100 batches
NUM_WORKERS = 12  # Número de workers para cargar datos en paralelo

torch.set_num_threads(16)

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
    fold_start_time = time.time()  # Inicia el timer del fold

    # Subconjuntos de entrenamiento y prueba con DataLoader que usan NUM_WORKERS
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=NUM_WORKERS)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=test_subsampler, num_workers=NUM_WORKERS)
    
    # Inicializar el modelo preentrenado (ResNet50 como ejemplo)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Ruta del checkpoint para este fold
    checkpoint_path = f"checkpoint_fold{fold+1}_latest.pth"
    start_epoch = 0  # Época a partir de la cual se reanuda
    start_batch = 0  # Batch desde el cual continuar en la época de reanudación

    # Si existe un checkpoint, cargar estado
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint.get('batch_index', 0)
        print(f"Reanudando fold {fold+1} desde epoch {start_epoch+1} y batch {start_batch+1}")

    # Entrenamiento del modelo
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()  # Inicia timer de la época

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            # Si se reanuda y estamos en la misma epoch, saltar batches ya procesados
            if epoch == start_epoch and batch_index < start_batch:
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Guardar checkpoint periódicamente
            if (batch_index + 1) % SAVE_INTERVAL == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_index': batch_index + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
                print(f"Checkpoint guardado: {checkpoint_path} (Epoch {epoch+1}, Batch {batch_index+1})")

        # Guardar checkpoint al final de la época
        torch.save({
            'epoch': epoch + 1,  # Se incrementa para indicar que la epoch terminó
            'batch_index': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader)
        }, checkpoint_path)
        print(f"Checkpoint final de la epoch guardado: {checkpoint_path}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        epoch_end_time = time.time()  # Fin del timer de la época
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Tiempo: {epoch_duration:.2f} segundos")
        
        # Reiniciar start_batch para las epochs siguientes
        start_batch = 0

    # Evaluación del modelo en el conjunto de prueba
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.2f}%")
    accuracies.append(test_accuracy)

    # Guardar el mejor modelo global si se mejora la accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%")

    # (Opcional) Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    fold_end_time = time.time()  # Fin del timer del fold
    fold_duration = fold_end_time - fold_start_time
    print(f"Tiempo total para Fold {fold + 1}: {fold_duration:.2f} segundos")

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy:.2f}%")