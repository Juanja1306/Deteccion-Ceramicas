import os
import time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern #type: ignore[import]
from skimage.color import rgb2gray #type: ignore[import]
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Configuración inicial
ROOT_DIR = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"  # Ruta a la carpeta raíz con subcarpetas de etiquetas
NUM_CLASSES = len(os.listdir(ROOT_DIR))  # Número de clases (subcarpetas)
NUM_FOLDS = 5  # Número de folds para validación cruzada
IMAGE_SIZE = (512, 512)  # Tamaño deseado para las imágenes
MODEL_SAVE_PATH = "mejor_modelo_lbp.pkl"  # Ruta para guardar el mejor modelo
RANDOM_STATE = 42

# Parámetros para LBP
RADIUS = 3  # Radio del LBP
N_POINTS = 8 * RADIUS  # Número de puntos alrededor del pixel central
METHOD = 'uniform'  # Método de cálculo del LBP

# Función para calcular LBP de una imagen
def compute_lbp(image_path):
    image = Image.open(image_path).convert("L")  # Convertir a escala de grises
    image = image.resize(IMAGE_SIZE)  # Redimensionar
    gray_image = np.array(image)  # Convertir a matriz numpy
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# Cargar dataset completo
def load_dataset(root_dir):
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    features = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                features.append(compute_lbp(img_path))
                labels.append(class_to_idx[cls])
    return np.array(features), np.array(labels)

# Cargar datos
print("Cargando dataset...")
X, y = load_dataset(ROOT_DIR)

# Validación cruzada con KFold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
accuracies = []
best_accuracy = 0.0  # Para rastrear el mejor accuracy

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\nIniciando Fold {fold + 1}/{NUM_FOLDS}")
    fold_start_time = time.time()

    # Dividir datos de entrenamiento y prueba
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Entrenar un clasificador (SVM o Random Forest)
    clf = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)  # Usamos SVM lineal
    # clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)  # Alternativa: Random Forest
    clf.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy:.2f}%")

    # Guardar el mejor modelo si mejora la precisión
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        import joblib
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%")

    fold_end_time = time.time()
    fold_duration = fold_end_time - fold_start_time
    print(f"Tiempo total para Fold {fold + 1}: {fold_duration:.2f} segundos")

# Calcular precisión final promedio
final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy:.2f}%")