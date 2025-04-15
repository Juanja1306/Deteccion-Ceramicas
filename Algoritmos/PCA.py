import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torchvision import transforms #type: ignore
import joblib  # Para guardar y cargar checkpoints

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
IMAGE_SIZE = (512, 512)   # Tamaño deseado: 512x512
NUM_FOLDS = 5             # Número de folds para validación cruzada
NUM_COMPONENTS = 50       # Número de componentes principales a extraer (ajustable)
CHECKPOINT_DIR = "./PCA_Checkpoints"  # Directorio para guardar los checkpoints

# Crear el directorio si no existe
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Transformación para las imágenes
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Redimensiona la imagen a 512x512
    transforms.ToTensor(),          # Convierte la imagen a tensor; para escala de grises, el tensor tendrá forma [1, 512, 512]
])

# Función para cargar el dataset y convertir cada imagen en un vector aplanado
def load_dataset(root_dir, transform):
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    images = []
    labels = []
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                try:
                    # Abrir la imagen en escala de grises y aplicar transformaciones
                    img = Image.open(img_path).convert("L")
                    img = transform(img)
                    img = img.numpy()
                    img = img.flatten()  # Aplanar la imagen a un vector
                    images.append(img)
                    labels.append(class_to_idx[cls])
                except Exception as e:
                    print(f"Error al cargar {img_path}: {e}")
                    
    return np.array(images), np.array(labels)

# Cargar el dataset completo
X, y = load_dataset(ROOT_DIR, transform)
print("Dataset cargado:", X.shape, y.shape)

# Configurar validación cruzada con KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
accuracies = []

# Determinar el último fold completado
last_completed_fold = -1
for fold in range(NUM_FOLDS):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold{fold+1}.pkl")
    if os.path.exists(checkpoint_path):
        last_completed_fold = fold

print(f"Último fold completado: {last_completed_fold + 1}/{NUM_FOLDS}")

# Continuar desde el siguiente fold
start_fold = last_completed_fold + 1
if start_fold >= NUM_FOLDS:
    print("Todos los folds ya han sido completados. No hay más folds por procesar.")
else:
    print(f"Reanudando desde el fold {start_fold + 1}/{NUM_FOLDS}")

for fold in range(start_fold, NUM_FOLDS):
    print(f"\nFold {fold + 1}/{NUM_FOLDS}")
    
    # Dividir en conjuntos de entrenamiento y prueba
    train_idx, test_idx = list(kf.split(X))[fold]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_fold{fold+1}.pkl")
    
    # Si existe el checkpoint, se carga el modelo y PCA guardados
    if os.path.exists(checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        pca = checkpoint["pca"]
        clf = checkpoint["clf"]
        # Transformar los datos de entrenamiento y prueba usando el PCA cargado
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(f"Checkpoint cargado desde: {checkpoint_path}")
    else:
        # Ajustar PCA sobre el conjunto de entrenamiento y transformar ambos conjuntos
        pca = PCA(n_components=NUM_COMPONENTS)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Entrenar el clasificador (Regresión Logística)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_pca, y_train)
        
        # Guardar el PCA y el clasificador como checkpoint
        joblib.dump({"pca": pca, "clf": clf}, checkpoint_path)
        print(f"Checkpoint guardado en: {checkpoint_path}")
    
    # Evaluación en el conjunto de prueba
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en fold {fold + 1}: {acc * 100:.2f}%")
    accuracies.append(acc)

final_accuracy = np.mean(accuracies)
print(f"\nAccuracy promedio final: {final_accuracy * 100:.2f}%")