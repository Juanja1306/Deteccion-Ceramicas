import os
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torchvision import models, transforms #type: ignore
from torch.utils.data import DataLoader, Dataset #type: ignore
from PIL import Image
from sklearn.model_selection import KFold
import numpy as np

# Configuración inicial
ROOT_DIR = "ruta/a/tu/carpeta/raiz"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (1024, 1024)  # Tamaño deseado para las imágenes
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5  # Número de folds para validación cruzada
MODEL_SAVE_PATH = "mejor_modelo.pth"  # Ruta para guardar el mejor modelo

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convertir a RGB para compatibilidad con modelos preentrenados
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalización para escala de grises
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
    print(f"Fold {fold + 1}/{NUM_FOLDS}")

    # Subconjuntos de entrenamiento y prueba
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

    # Cargar modelo preentrenado (ResNet50 como ejemplo)
    model = models.resnet50(pretrained=True)

    # Ajustar la primera capa convolucional para aceptar imágenes en escala de grises
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Reemplazar la capa fully connected para adaptarse al número de clases
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Mover el modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrenamiento del modelo
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

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
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.2f}%\n")
    accuracies.append(test_accuracy)

    # Guardar el mejor modelo
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%\n")

# Calcular el accuracy final promedio
final_accuracy = np.mean(accuracies)
print(f"Final Average Test Accuracy: {final_accuracy:.2f}%")