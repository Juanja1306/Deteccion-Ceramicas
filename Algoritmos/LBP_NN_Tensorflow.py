import os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image
import joblib
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import json

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Verificar disponibilidad de GPU (TensorFlow usará la GPU automáticamente si está instalada)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU(s) disponibles:", gpus)
else:
    print("No se encontró GPU. Se usará CPU.")

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5                         # Número de folds para validación cruzada
IMAGE_SIZE = (512, 512)               # Tamaño deseado para las imágenes
MODEL_SAVE_PATH = "mejor_modelo_lbp_NN.h5" # Ruta para guardar el mejor modelo
RANDOM_STATE = 42
EPOCHS = 50                           # Número de épocas para entrenar la red
BATCH_SIZE = 32                       # Tamaño del batch

# Parámetros para la red neuronal
HIDDEN_LAYER_SIZES = [128, 64]          # Capas ocultas (128 y 64 neuronas)
LEARNING_RATE = 0.001

# Parámetros para LBP
RADIUS = 3                            # Radio del LBP
N_POINTS = 8 * RADIUS                 # Número de puntos alrededor del pixel central
METHOD = 'uniform'                    # Método de cálculo del LBP

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


# Función para construir el modelo de red neuronal
def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(HIDDEN_LAYER_SIZES[0], activation='relu'))
    model.add(Dense(HIDDEN_LAYER_SIZES[1], activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=LEARNING_RATE)
    # Usamos sparse categorical crossentropy, ya que las etiquetas son enteros
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Fijar la semilla para reproducibilidad
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Cargar datos
print("Cargando dataset...")
X, y = load_dataset(ROOT_DIR)
input_dim = X.shape[1]

# Validación cruzada con KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
accuracies = []
best_accuracy = 0.0

# Sistema de checkpoints: se guardarán los modelos en formato .h5
start_fold = 0
# Se revisa desde el último fold hacia el primero para detectar un checkpoint guardado previamente
for fold_num in range(NUM_FOLDS, 0, -1):
    checkpoint_path = f"checkpoint_fold{fold_num}_latest.h5"
    if os.path.exists(checkpoint_path):
        start_fold = fold_num - 1  # Convertir a índice 0-based
        print(f"Se encontró un checkpoint en el fold {fold_num}. Reanudando desde este fold.")
        break

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    if fold < start_fold:
        print(f"Saltando fold {fold + 1} ya completado o sin checkpoint pendiente...")
        continue

    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")
    fold_start_time = time.time()

    # División de los datos
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Ruta del checkpoint para este fold
    checkpoint_path = f"checkpoint_fold{fold + 1}_latest.h5"
    
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
        print(f"Reanudando fold {fold + 1} desde el checkpoint guardado.")
    else:
        model = build_model(input_dim, NUM_CLASSES)
        model.fit(X_train, y_train, 
                  epochs=EPOCHS, 
                  batch_size=BATCH_SIZE, 
                  validation_split=0.1,
                  verbose=1)
        model.save(checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path} para fold {fold + 1}.")

    # Evaluación en el fold actual
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy * 100:.2f}%")

    # Guardar el mejor modelo global si se mejora la accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.save(MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy * 100:.2f}%")

    fold_end_time = time.time()
    print(f"Tiempo total para Fold {fold + 1}: {fold_end_time - fold_start_time:.2f} segundos")

    # (Opcional) Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy * 100:.2f}%")
print(f"Mejor modelo guardado en: {MODEL_SAVE_PATH} con accuracy: {best_accuracy * 100:.2f}%")
