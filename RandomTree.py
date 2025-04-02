import os
import time
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Para guardar/cargar checkpoints

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
NUM_EPOCHS = 15
IMAGE_SIZE = (512, 512)  # Nuevo tamaño: 512x512
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases
NUM_FOLDS = 5  # Número de folds para validación cruzada
MODEL_SAVE_PATH = "mejor_modelo.pkl"  # Ruta para guardar el mejor modelo
SAVE_INTERVAL = 100  # Se mantiene en la estructura (no se utiliza en batch)
NUM_WORKERS = 12  # No se utiliza en este ejemplo

torch.set_num_threads(16)

# Transformaciones para las imágenes (en escala de grises, 1 canal)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),  # Resultado shape: (1, 512, 512)
    transforms.Normalize(mean=[0.5], std=[0.5])
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

# Función para convertir un subconjunto del dataset a arrays de NumPy
def dataset_to_numpy(dataset, indices):
    X_list = []
    y_list = []
    for idx in indices:
        image, label = dataset[idx]
        # Convertir el tensor (1, 512, 512) a vector
        X_list.append(image.numpy().flatten())
        y_list.append(label)
    return np.array(X_list), np.array(y_list)

# Crear dataset completo
full_dataset = CustomImageDataset(root_dir=ROOT_DIR, transform=transform)

# Dividir el dataset usando KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

accuracies = []
best_accuracy = 0.0  # Para rastrear el mejor accuracy

trees_per_epoch = 10  # Número de árboles a agregar por epoch

# Definir la carpeta para los checkpoints
checkpoint_folder = "Tree_Checkpoints"
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

# Validación cruzada con KFold
for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")
    fold_start_time = time.time()

    # Convertir subconjuntos de entrenamiento y prueba a arrays de NumPy
    X_train, y_train = dataset_to_numpy(full_dataset, train_idx)
    X_test, y_test = dataset_to_numpy(full_dataset, test_idx)
    
    # Inicializar el Random Forest con warm_start=True para entrenamiento incremental
    clf = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=42)
    
    # Ruta del checkpoint para este fold, dentro de la carpeta Tree_Checkpoints
    checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_fold{fold+1}_latest.pkl")
    start_epoch = 0  # Época a partir de la cual se reanuda

    # Si existe un checkpoint, cargar estado
    if os.path.exists(checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        clf = checkpoint['model']
        start_epoch = checkpoint['epoch']
        print(f"Reanudando fold {fold+1} desde epoch {start_epoch+1}")
    
    # Entrenamiento del modelo (simulación de epochs incrementales)
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Incrementar el número de árboles
        clf.n_estimators += trees_per_epoch
        
        # Entrenar el modelo (se agregan nuevos árboles manteniendo los anteriores)
        clf.fit(X_train, y_train)
        
        # Calcular precisión en entrenamiento (opcional)
        train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred) * 100.0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Accuracy: {train_accuracy:.2f}%")
        
        # Guardar checkpoint al final de la epoch
        joblib.dump({
            'epoch': epoch + 1,
            'model': clf
        }, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path} (Epoch {epoch+1})")
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Tiempo de epoch: {epoch_duration:.2f} segundos")
    
    # Evaluación del modelo en el conjunto de prueba
    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred) * 100.0
    print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.2f}%")
    accuracies.append(test_accuracy)
    
    # Guardar el mejor modelo global si se mejora la accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%")
    
    # Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    fold_duration = time.time() - fold_start_time
    print(f"Tiempo total para Fold {fold + 1}: {fold_duration:.2f} segundos")

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy:.2f}%")
