import os
import numpy as np
import joblib
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore

# ========================= RUTAS HARDCODED =========================
# Carpeta raíz con subcarpetas de clases
DATA_ROOT   = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
# Modelo KNN entrenado
MODEL_PATH  = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_KNN.pkl"
# Imagen a predecir
IMAGE_PATH  = r"C:\Users\juanj\Pictures\florencia.jpg"

# ========================= PARÁMETROS LBP =========================
IMAGE_SIZE = (512, 512)
RADIUS     = 3
N_POINTS   = 8 * RADIUS
METHOD     = 'uniform'

# ========================= CARGAR MODELO =========================
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en:\n{MODEL_PATH}")
clf = joblib.load(MODEL_PATH)
print(f"✅ Modelo KNN cargado desde: {MODEL_PATH}\n")

# ========================= LISTADO DE CLASES =========================
classes = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)
print("Clases:", classes, "\n")

# ========================= FUNCIÓN compute_lbp =========================
def compute_lbp(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS+3),
        range=(0, N_POINTS+2)
    )
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

# ========================= PREDICCIÓN TOP‑5 =========================
def predict_top5(img_path: str):
    feat = compute_lbp(img_path).reshape(1, -1)
    probs = clf.predict_proba(feat)[0]
    idx5  = np.argsort(probs)[::-1][:5]
    return [(classes[i], probs[i]) for i in idx5]

# ========================= EJECUTAR PREDICCIÓN =========================
if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"No existe la imagen:\n{IMAGE_PATH}")

top5 = predict_top5(IMAGE_PATH)
print(f"Imagen: {IMAGE_PATH}\nTop‑5 Predicciones:")
for rank, (label, prob) in enumerate(top5, 1):
    print(f"{rank}) {label} — {prob*100:.2f}%")
