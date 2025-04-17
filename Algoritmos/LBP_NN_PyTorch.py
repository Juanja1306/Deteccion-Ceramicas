import os
import time
import json
import numpy as np
from sklearn.model_selection import KFold
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import json

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# ============================ CONFIGURACIÓN ============================
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a las imágenes, organizadas en subcarpetas por clase.
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5                           # Número de folds para validación cruzada
IMAGE_SIZE = (512, 512)                 # Tamaño deseado para las imágenes
MODEL_SAVE_PATH = "mejor_modelo_lbp.pt"   # Archivo para guardar el mejor modelo
RANDOM_STATE = 42
EPOCHS = 50                           # Número de épocas de entrenamiento
BATCH_SIZE = 32                       # Tamaño del batch de entrenamiento

# Parámetros para la red neuronal
HIDDEN_LAYER_SIZES = [128, 64]          # Neuronas en cada capa oculta
LEARNING_RATE = 0.001

# Parámetros para LBP
RADIUS = 3                            # Radio para el LBP
N_POINTS = 8 * RADIUS                 # Número de puntos alrededor del pixel central
METHOD = 'uniform'                    # Método para calcular el LBP

# Configuración de reproducibilidad y dispositivo de cómputo
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo a usar:", device)


# ============================ FUNCIONES DE PREPROCESAMIENTO ============================
def compute_lbp(image_path):
    """
    Carga una imagen, la convierte a escala de grises, la redimensiona y calcula su histograma LBP.
    """
    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE)
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Crear histograma y normalizar
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def load_dataset(root_dir, json_file="dataset.json"):
    """
    Recorre las subcarpetas (etiquetas) y extrae el histograma LBP de cada imagen.
    Primero verifica si existe un archivo JSON con los datos preprocesados. 
    Si existe, carga y retorna dichos datos; en caso contrario, procesa las imágenes y guarda los resultados.
    """
    if os.path.exists(json_file):
        print("Cargando datos desde el archivo JSON...")
        with open(json_file, "r") as f:
            data = json.load(f)
        features = np.array(data["features"])
        labels = np.array(data["labels"])
        return features, labels

    # Si no existe el JSON, se procede a procesar los datos
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    features = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            print(f"Procesando clase: {cls}")
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


# ============================ DEFINICIÓN DEL MODELO CON PYTORCH ============================
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_LAYER_SIZES[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZES[0], HIDDEN_LAYER_SIZES[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZES[1], num_classes)
        # No se aplica softmax en la salida porque CrossEntropyLoss lo incluye internamente.

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# ============================ CARGA DE DATOS ============================
print("Cargando dataset...")
X, y = load_dataset(ROOT_DIR)
input_dim = X.shape[1]

# Convertir los arrays de numpy a tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ============================ VALIDACIÓN CRUZADA CON KFOLD ============================
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
accuracies = []
best_accuracy = 0.0

# Sistema de checkpoints: se buscará un archivo de checkpoint para reanudar en caso de existir.
start_fold = 0
for fold_num in range(NUM_FOLDS, 0, -1):
    checkpoint_path = f"checkpoint_fold{fold_num}_latest.pt"
    if os.path.exists(checkpoint_path):
        start_fold = fold_num - 1  # Convertir a índice 0-based
        print(f"Se encontró un checkpoint en el fold {fold_num}. Reanudando desde este fold.")
        break

fold_index = 0
for train_idx, test_idx in kf.split(X_tensor):
    if fold_index < start_fold:
        print(f"Saltando fold {fold_index + 1} ya completado o sin checkpoint pendiente...")
        fold_index += 1
        continue

    print(f"\nIniciando Fold {fold_index + 1}/{NUM_FOLDS}")
    fold_start_time = time.time()
    
    # División de los datos de entrenamiento y prueba para el fold actual.
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    # Crear DataLoader para el conjunto de entrenamiento.
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Ruta del checkpoint para este fold.
    checkpoint_path = f"checkpoint_fold{fold_index + 1}_latest.pt"
    
    # Crear el modelo y moverlo al dispositivo (GPU o CPU).
    model = NeuralNet(input_dim, NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Si existe un checkpoint, se carga el modelo y el optimizador.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Reanudando fold {fold_index + 1} desde el checkpoint guardado.")
    else:
        # Entrenar el modelo durante el número de épocas establecido.
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()           # Reiniciar los gradientes
                outputs = model(batch_x)          # Forward pass
                loss = criterion(outputs, batch_y)  # Calcular la pérdida
                loss.backward()                   # Backward pass
                optimizer.step()                  # Actualizar los pesos
                
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Fold {fold_index + 1} Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")
        
        # Guardar el checkpoint para este fold
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path} para fold {fold_index + 1}.")

    # ============================ EVALUACIÓN DEL MODELO ============================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        X_test_device = X_test.to(device)
        y_test_device = y_test.to(device)
        outputs = model(X_test_device)
        _, predicted = torch.max(outputs, 1)
        total += y_test_device.size(0)
        correct += (predicted == y_test_device).sum().item()
    accuracy = correct / total
    accuracies.append(accuracy)
    print(f"Fold {fold_index + 1} Test Accuracy: {accuracy * 100:.2f}%")

    # Guardar el mejor modelo global en base a la precisión del test
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy * 100:.2f}%")

    fold_end_time = time.time()
    print(f"Tiempo total para Fold {fold_index + 1}: {fold_end_time - fold_start_time:.2f} segundos")

    # (Opcional) Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    fold_index += 1

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy * 100:.2f}%")
print(f"Mejor modelo guardado en: {MODEL_SAVE_PATH} con accuracy: {best_accuracy * 100:.2f}%")
