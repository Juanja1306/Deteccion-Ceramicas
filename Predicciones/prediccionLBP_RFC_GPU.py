import os
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import joblib

# Parámetros LBP (idénticos a los de entrenamiento)
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'
IMAGE_SIZE = (512, 512)

# Rutas que proporcionas
ROOT_DIR   = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_RFC.pkl"
IMAGE_PATH = r"C:\Users\juanj\Pictures\florencia.jpg"

def compute_lbp(image_path: str) -> np.ndarray:
    """Calcula el histograma LBP normalizado de una sola imagen."""
    img = Image.open(image_path).convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

if __name__ == "__main__":
    # 1) Cargar el modelo sklearn
    clf = joblib.load(MODEL_PATH)

    # 2) Reconstruir nombres de clases (las carpetas bajo ROOT_DIR)
    classes = sorted([
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
    ])

    # 3) Calcular LBP de la imagen y dar formato [1, n_features]
    feature = compute_lbp(IMAGE_PATH).reshape(1, -1)

    # 4) Predecir
    pred_idx = clf.predict(feature)[0]
    pred_class = classes[pred_idx]

    print(f"Imagen: {os.path.basename(IMAGE_PATH)}")
    print(f"Predicción → Índice: {pred_idx}, Clase: '{pred_class}'")
