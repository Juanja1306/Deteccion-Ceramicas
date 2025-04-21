import os
import time
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image
import joblib
import warnings

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Configuración inicial
ROOT_DIR        = "/home/admingig/Deteccion-Ceramicas/DATA/Ruido/"
NUM_CLASSES     = len(os.listdir(ROOT_DIR))
NUM_FOLDS       = 5
IMAGE_SIZE      = (512, 512)
MODEL_SAVE_PATH = "mejor_modelo_lbp_NC.pkl"
RANDOM_STATE    = 42

# Parámetros para LBP
RADIUS   = 3
N_POINTS = 8 * RADIUS
METHOD   = 'uniform'

def compute_lbp(image_path):
    """Calcula el histograma LBP normalizado de una imagen."""
    img = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

def load_dataset(root_dir, json_file="dataset.json"):
    """Carga o genera el dataset de LBP desde imágenes y lo cachea en JSON."""
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
        except (json.JSONDecodeError, KeyError):
            print("⚠️ JSON inválido, regenerando desde las imágenes…")

    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    features, labels = [], []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        print(f"Procesando clase: {cls}")
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            features.append(compute_lbp(img_path))
            labels.append(class_to_idx[cls])

    features = np.array(features)
    labels = np.array(labels)

    with open(json_file, "w") as f:
        json.dump({
            "features": features.tolist(),
            "labels": labels.tolist()
        }, f)
    print("Datos preprocesados guardados en:", json_file)

    return features, labels

# === Cargar datos ===
print("Cargando dataset...")
X, y = load_dataset(ROOT_DIR)

# === Buscar checkpoint para reanudar ===
start_fold = 0
for fn in range(NUM_FOLDS, 0, -1):
    ck = f"checkpoint_fold{fn}_latest.pkl"
    if os.path.exists(ck):
        start_fold = fn - 1
        print(f"Se encontró checkpoint en fold {fn}, reanudando...")
        break

# === Validación cruzada ===
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
accuracies = []
best_acc = 0.0

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    if fold < start_fold:
        print(f"Saltando fold {fold+1} (ya completado)...")
        continue

    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    t0 = time.time()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ck_path = f"checkpoint_fold{fold+1}_latest.pkl"

    if os.path.exists(ck_path):
        clf = joblib.load(ck_path)['clf']
        print(f"  → Reanudado desde {ck_path}")
    else:
        clf = NearestCentroid()
        clf.fit(X_train, y_train)
        joblib.dump({'clf': clf}, ck_path)
        print(f"  → Checkpoint guardado en {ck_path}")

    # Evaluación
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"  Fold {fold+1} Accuracy: {acc*100:.2f}%")

    # Mejor modelo global
    if acc > best_acc:
        best_acc = acc
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"  ↑ Nuevo mejor modelo global guardado ({best_acc*100:.2f}%)")

    print(f"  Tiempo fold: {time.time()-t0:.2f}s")

    # Opcional: limpiar checkpoint
    if os.path.exists(ck_path):
        os.remove(ck_path)

# === Resultados finales ===
mean_acc = np.mean(accuracies)
print(f"\nAverage Accuracy: {mean_acc*100:.2f}%")
print(f"Mejor modelo final en: {MODEL_SAVE_PATH} ({best_acc*100:.2f}%)")
