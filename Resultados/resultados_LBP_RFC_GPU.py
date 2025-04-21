import os
import random
import numpy as np
import cupy as cp
import joblib
import pandas as pd
from skimage.feature import local_binary_pattern  # type: ignore
from PIL import Image
import warnings

# ========================= Configuraciones Iniciales =========================
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

ROOT_DIR    = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"                  # Carpetas por clase
MODEL_PATH  = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_RFC_GPU.pkl"
IMAGE_SIZE  = (512, 512)            # Igual que en entrenamiento
SAMPLE_PERCENT = 0.2                # Muestra del 20% de cada carpeta
OUTPUT_CSV  = "resultados_prediccion_lbp_RFC_GPU.csv"

# Parámetros LBP (idénticos a los usados en entrenamiento)
RADIUS   = 3
N_POINTS = 8 * RADIUS
METHOD   = 'uniform'

# ========================= Función LBP =========================
def compute_lbp(path):
    """Devuelve histograma LBP normalizado (NumPy array)."""
    img = Image.open(path).convert("L").resize(IMAGE_SIZE)
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

# ========================= Carga del modelo GPU =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
clf = joblib.load(MODEL_PATH)
print("Modelo GPU RandomForest cargado desde:", MODEL_PATH)

# ========================= Clases =========================
classes = sorted(d for d in os.listdir(ROOT_DIR)
                 if os.path.isdir(os.path.join(ROOT_DIR, d)))
print("Clases detectadas:", classes)

# ========================= Evaluación por carpeta =========================
results = {}
valid_ext = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")

for label in classes:
    dir_label = os.path.join(ROOT_DIR, label)
    imgs = [f for f in os.listdir(dir_label)
            if f.lower().endswith(valid_ext)]
    if not imgs:
        continue

    n_sample = max(1, int(len(imgs) * SAMPLE_PERCENT))
    sampled = random.sample(imgs, n_sample)

    total, correct = 0, 0
    confidences = []

    for fn in sampled:
        fp = os.path.join(dir_label, fn)
        feat = compute_lbp(fp).reshape(1, -1)

        # Llevar a GPU y predecir
        feat_gpu = cp.asarray(feat)
        pred_gpu = clf.predict(feat_gpu)
        proba_gpu = clf.predict_proba(feat_gpu)

        pred_idx = int(cp.asnumpy(pred_gpu)[0])
        pred_label = classes[pred_idx]
        conf = float(cp.asnumpy(proba_gpu)[0].max())

        total += 1
        if pred_label == label:
            correct += 1
        confidences.append(conf)

    acc = 100 * correct / total
    avg_conf = 100 * (sum(confidences) / len(confidences))
    results[label] = {
        "Total Imagenes Evaluadas": total,
        "Aciertos": correct,
        "Accuracy (%)": acc,
        "Confianza Promedio (%)": avg_conf
    }
    print(f"[{label}] eval:{total} → acc={acc:.2f}%, conf={avg_conf:.2f}%")

# ========================= DataFrame y CSV =========================
df = pd.DataFrame.from_dict(results, orient="index")\
       .sort_values("Accuracy (%)")
df.to_csv(OUTPUT_CSV, index_label="Clase")
print(f"\nResultados completos guardados en: {OUTPUT_CSV}")
print(df.to_string())
