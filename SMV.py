import os
import time
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
IMAGE_SIZE = (224, 224)  # Tamaño usado para el extractor (modelo preentrenado)
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases
NUM_FOLDS = 5  # Número de folds para validación cruzada
MODEL_SAVE_PATH = "mejor_modelo_svm.pkl"  # Ruta para guardar el mejor modelo final
NUM_EPOCHS = 15  # Número de épocas para el entrenamiento incremental
BATCH_SIZE = 32  # Tamaño de mini-batch para el entrenamiento incremental
SAVE_INTERVAL = 100  # Guardar checkpoint cada SAVE_INTERVAL batches

# Directorio para guardar los checkpoints
CHECKPOINT_DIR = "SVM_Checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convertir a 3 canales para compatibilidad
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
dataset = CustomImageDataset(root_dir=ROOT_DIR, transform=transform)

# Utilizar un modelo preentrenado (ResNet50) como extractor de características.
# Se elimina la capa final de clasificación para obtener vectores de características.
feature_extractor = models.resnet50(pretrained=True)
feature_extractor.fc = torch.nn.Identity()
feature_extractor.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

# Función para extraer características de las imágenes usando el extractor preentrenado
def extract_features(loader):
    features = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = feature_extractor(images)
            output = output.cpu().numpy()  # Convertir a numpy array
            features.append(output)
            labels_list.extend(labels.numpy())
    features = np.concatenate(features, axis=0)
    return features, np.array(labels_list)

# Extraer características de todo el dataset
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
X, y = extract_features(data_loader)
print("Forma de las características:", X.shape)

# Validación cruzada con KFold para entrenar y evaluar el clasificador incremental (SVM lineal)
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
accuracies = []
best_accuracy = 0.0
best_model = None

# Iterar sobre cada fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Definir la ruta del checkpoint para el fold actual
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold{fold+1}_latest.pkl")
    
    # Verificar si existe un checkpoint para reanudar el entrenamiento
    if os.path.exists(checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        clf = checkpoint['model']
        start_epoch = checkpoint['epoch']
        global_batch = checkpoint['global_batch']
        print(f"Reanudando Fold {fold+1} desde epoch {start_epoch+1} y global_batch {global_batch}")
    else:
        clf = SGDClassifier(loss='hinge', max_iter=1, tol=None, warm_start=True)
        start_epoch = 0
        global_batch = 0
    
    num_batches = int(np.ceil(len(X_train) / BATCH_SIZE))
    start_time_fold = time.time()
    
    # Entrenamiento incremental: iterar en epochs y mini-batches
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Barajar los datos en cada epoch
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Si reanudamos en medio de una epoch, calcular el batch inicial
        start_batch = global_batch % num_batches if epoch == start_epoch else 0
        
        for batch_index in range(start_batch, num_batches):
            start = batch_index * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            # La primera llamada a partial_fit requiere pasar el parámetro 'classes'
            if epoch == 0 and batch_index == start_batch:
                clf.partial_fit(X_batch, y_batch, classes=np.unique(y))
            else:
                clf.partial_fit(X_batch, y_batch)
            
            global_batch += 1
            # Guardar checkpoint cada SAVE_INTERVAL batches
            if global_batch % SAVE_INTERVAL == 0:
                checkpoint_data = {
                    'model': clf,
                    'epoch': epoch,
                    'global_batch': global_batch
                }
                joblib.dump(checkpoint_data, checkpoint_path)
                print(f"Checkpoint guardado: {checkpoint_path} (Epoch {epoch+1}, Batch {batch_index+1})")
        
        # Al final de la epoch, guardar checkpoint
        checkpoint_data = {
            'model': clf,
            'epoch': epoch + 1,
            'global_batch': global_batch
        }
        joblib.dump(checkpoint_data, checkpoint_path)
        print(f"Checkpoint final de epoch guardado: {checkpoint_path}")
    
    elapsed_time = time.time() - start_time_fold
    print(f"Tiempo total de entrenamiento en Fold {fold+1}: {elapsed_time:.2f} segundos")
    
    # Evaluación del modelo en el conjunto de prueba
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Fold {fold+1} - Accuracy en test: {acc:.2f}%")
    accuracies.append(acc)
    
    # Guardar el mejor modelo basado en la accuracy
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = clf
        joblib.dump(best_model, MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%")

final_accuracy = np.mean(accuracies)
print(f"\nAccuracy promedio final: {final_accuracy:.2f}%")
