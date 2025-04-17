import os
import random
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore
import warnings
from sklearn.cluster import KMeans

# ========================= Configuraciones Iniciales =========================
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

ROOT_DIR        = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
MODEL_PATH      = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_KMeans.pkl"
CSV_OUTPUT_PATH = "resultados_prediccion_lbp_KMeans.csv"

IMAGE_SIZE     = (512, 512)
SAMPLE_PERCENT = 0.2

RADIUS   = 3
N_POINTS = 8 * RADIUS
METHOD   = 'uniform'

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# ========================= Clase KMeansClassifier =========================
class KMeansClassifier:
    def __init__(self, n_clusters, random_state=None):
        self.n_clusters   = n_clusters
        self.kmeans       = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_labels = {}

    def fit(self, X, y):
        self.kmeans.fit(X)
        for cluster in range(self.n_clusters):
            idxs = np.where(self.kmeans.labels_ == cluster)[0]
            if len(idxs) > 0:
                lbls, counts = np.unique(y[idxs], return_counts=True)
                self.cluster_labels[cluster] = lbls[np.argmax(counts)]
            else:
                self.cluster_labels[cluster] = -1
        return self

    def predict(self, X):
        clusters = self.kmeans.predict(X)
        return np.array([self.cluster_labels[c] for c in clusters])

# ========================= Función para calcular LBP =========================
def compute_lbp(image_path):
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# ========================= Cargar el Modelo =========================
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: no se encontró el modelo en {MODEL_PATH}")
    exit(1)
clf = joblib.load(MODEL_PATH)
print("Modelo KMeans cargado desde:", MODEL_PATH)

# ========================= Detectar Clases =========================
classes = sorted([
    d for d in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, d))
])
print("Clases detectadas:", classes)

# ========================= Evaluación por Carpeta =========================
results = {}

for label in classes:
    folder = os.path.join(ROOT_DIR, label)
    images = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    total_imgs = len(images)
    if total_imgs == 0:
        continue

    sample_size = max(1, int(total_imgs * SAMPLE_PERCENT))
    sampled = random.sample(images, sample_size)

    processed    = 0
    correct      = 0
    confidences  = []

    for img_name in sampled:
        path = os.path.join(folder, img_name)
        feat = compute_lbp(path)
        if feat is None:
            continue
        feat = feat.reshape(1, -1)

        cluster = clf.kmeans.predict(feat)[0]
        pred    = clf.cluster_labels[cluster]

        # calcular confianza
        dists = clf.kmeans.transform(feat)[0]
        conf  = 1 - (dists[cluster] / dists.sum())
        confidences.append(conf * 100)

        processed += 1
        if pred == classes.index(label):
            correct += 1

    if processed == 0:
        continue

    accuracy = (correct / processed) * 100
    avg_conf = np.mean(confidences)

    # Guardar métricas
    results[label] = {
        "Total Imagenes Evaluadas": processed,
        "Aciertos": correct,
        "Accuracy (%)": round(accuracy, 2),
        "Confianza Promedio (%)": round(avg_conf, 2)
    }

    # Impresión detallada por subcarpeta
    print(f"\nResultados para la subcarpeta '{label}':")
    print(f"  Total imágenes evaluadas: {processed}")
    print(f"  Aciertos: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Confianza Promedio: {avg_conf:.2f}%")

# ========================= Resumen y Exportación de Resultados =========================
df = pd.DataFrame.from_dict(results, orient="index")
df = df.sort_values(by="Accuracy (%)", ascending=True)

print("\nTabla de resultados por etiqueta (ordenadas por Accuracy):")
print(df.to_string())

# Subcarpeta con peor desempeño
worst_label = df.index[0]
worst_metrics = df.loc[worst_label]
print("\nLa subcarpeta con peor desempeño es:")
print(f"  Subcarpeta: {worst_label}")
print(worst_metrics.to_string())

# Fila PROMEDIO e inserción
df_reset = df.reset_index().rename(columns={'index': ''})
prom = ['PROMEDIO'] + [round(df_reset[col].mean(), 2) for col in df_reset.columns[1:]]
df_final = pd.concat([pd.DataFrame([prom], columns=df_reset.columns), df_reset], ignore_index=True)

# Guardar CSV
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH) or '.', exist_ok=True)
df_final.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"\nCSV generado en: {CSV_OUTPUT_PATH}")
