import os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image
import joblib
import warnings
import json

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5  # Número de folds para validación cruzada
IMAGE_SIZE = (512, 512)  # Tamaño deseado para las imágenes
MODEL_SAVE_PATH = "mejor_modelo_lbp_KNN.pkl"  # Ruta para guardar el mejor modelo
RANDOM_STATE = 42
N_NEIGHBORS = 5  # Número de vecinos para KNN (puedes ajustar este parámetro)

# Parámetros para LBP
RADIUS = 3             # Radio del LBP
N_POINTS = 8 * RADIUS  # Número de puntos alrededor del pixel central
METHOD = 'uniform'     # Método de cálculo del LBP

# Función para calcular LBP de una imagen
def compute_lbp(image_path):
    image = Image.open(image_path).convert("L")  # Convertir a escala de grises
    image = image.resize(IMAGE_SIZE)             # Redimensionar
    gray_image = np.array(image)                 # Convertir a matriz numpy
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# Función para cargar el dataset completo
def load_dataset(root_dir, json_file="datasetLBP.json"):
    """
    Recorre las subcarpetas (etiquetas) y extrae el histograma LBP de cada imagen.
    Primero verifica si existe un archivo JSON con los datos preprocesados. 
    Si existe, carga y retorna dichos datos; en caso contrario, procesa las imágenes y guarda los resultados.
    """
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
        except (json.JSONDecodeError, KeyError):
            print("⚠️ JSON inválido, regenerando desde las imágenes…")

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


# Cargar datos
print("Cargando dataset...")
X, y = load_dataset(ROOT_DIR)

# *********************** BUSCAR CHECKPOINT DEL ÚLTIMO FOLD *************************
start_fold = 0  # Por defecto se comienza desde el primer fold (índice 0)
# Se revisa desde el último fold hacia el primero para ver si existe un checkpoint guardado
for fold_num in range(NUM_FOLDS, 0, -1):
    checkpoint_path = f"checkpoint_fold{fold_num}_latest.pkl"
    if os.path.exists(checkpoint_path):
        start_fold = fold_num - 1  # Convertir a índice 0-based
        print(f"Se encontró un checkpoint en el fold {fold_num}. Reanudando desde este fold.")
        break
# **********************************************************************************

# Validación cruzada con KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
accuracies = []
best_accuracy = 0.0  # Para rastrear el mejor accuracy

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Si el fold actual es menor que el fold reanudable, se salta
    if fold < start_fold:
        print(f"Saltando fold {fold + 1} ya completado o sin checkpoint pendiente...")
        continue

    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")
    fold_start_time = time.time()

    # Dividir datos de entrenamiento y prueba
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Ruta del checkpoint para este fold
    checkpoint_path = f"checkpoint_fold{fold + 1}_latest.pkl"

    # Verificar si existe un checkpoint para este fold
    if os.path.exists(checkpoint_path):
        checkpoint = joblib.load(checkpoint_path)
        clf = checkpoint['clf']
        print(f"Reanudando fold {fold + 1} desde el checkpoint guardado.")
    else:
        # Crear el clasificador KNN
        clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        clf.fit(X_train, y_train)
        
        # Guardar el checkpoint para este fold
        joblib.dump({'clf': clf}, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path} para fold {fold + 1}.")

    # Evaluar el modelo en el fold actual
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy * 100:.2f}%")

    # Guardar el mejor modelo global si se mejora la accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy * 100:.2f}%")

    fold_end_time = time.time()
    print(f"Tiempo total para Fold {fold + 1}: {fold_end_time - fold_start_time:.2f} segundos")

    # (Opcional) Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy * 100:.2f}%")
print(f"Mejor modelo guardado en: {MODEL_SAVE_PATH} con accuracy: {best_accuracy * 100:.2f}%")
