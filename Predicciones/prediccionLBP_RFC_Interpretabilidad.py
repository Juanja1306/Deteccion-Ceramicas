import os
import joblib
import json
from json import JSONDecodeError

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import shap
from lime.lime_tabular import LimeTabularExplainer

# =============================== Configuraciones ===============================
RADIUS      = 3
N_POINTS    = 8 * RADIUS
METHOD      = 'uniform'
IMAGE_SIZE  = (512, 512)

ROOT_DIR    = r"C:\Users\juanj\Desktop\DATA FINAL\Ruido"
MODEL_PATH  = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp_RFC_FINAL.pkl"
IMAGE_PATH  = r"C:\Users\juanj\Pictures\florencia2.jpg"
JSON_FILE   = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\datasetLBP.json"

# =============================== Funciones ===============================
def compute_lbp(image_path):
    image = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    gray = np.array(image)
    lbp = local_binary_pattern(gray, P=N_POINTS, R=RADIUS, method=METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        range=(0, N_POINTS + 2)
    )
    hist = hist.astype(float)
    return hist / (hist.sum() + 1e-6)

def load_dataset(root_dir, json_file=JSON_FILE):
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            return np.array(data["features"]), np.array(data["labels"])
        except (JSONDecodeError, KeyError):
            print("⚠️ JSON inválido, regenerando…")

    classes = sorted(os.listdir(root_dir))
    cls2idx = {c: i for i, c in enumerate(classes)}
    feats, labs = [], []
    for cls in classes:
        folder = os.path.join(root_dir, cls)
        if not os.path.isdir(folder):
            continue
        for img in os.listdir(folder):
            h = compute_lbp(os.path.join(folder, img))
            feats.append(h)
            labs.append(cls2idx[cls])
    X = np.vstack(feats)
    y = np.array(labs)
    with open(json_file, "w") as f:
        json.dump({"features": X.tolist(), "labels": y.tolist()}, f)
    print("✅ Datos preprocesados guardados en", json_file)
    return X, y

def plot_feature_importance(clf):
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure()
    plt.title("Importancia de bins LBP (RFC)")
    plt.bar(range(len(imp)), imp[idx], align="center")
    plt.xlabel("Bins (ordenados)")
    plt.ylabel("Importancia")
    plt.show()
    print("Top 10 bins:")
    for i in idx[:10]:
        print(f"  bin_{i}: {imp[i]:.4f}")

def plot_partial_dependence(clf, X, bins, target):
    """
    PDP para un modelo multiclass: hay que pasar 'target' (índice de clase).
    """
    PartialDependenceDisplay.from_estimator(
        clf, X,
        features=bins,
        target=target,
        kind="average"
    )
    plt.show()

def shap_analysis(clf, X, instance, class_idx, feature_names):
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X)
    shap.summary_plot(sv, X, feature_names=feature_names)
    shap.initjs()
    shap.force_plot(
        expl.expected_value[class_idx],
        expl.shap_values(instance)[class_idx][0],
        instance[0],
        feature_names=feature_names
    )

def lime_analysis(clf, X_train, classes, instance, feature_names):
    expl = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=classes,
        mode="classification"
    )
    exp = expl.explain_instance(
        instance.ravel(),
        clf.predict_proba,
        num_features=5
    )
    exp.show_in_notebook(show_table=True)

def predict_image_top5(clf, classes, image_path):
    hist = compute_lbp(image_path)
    probs = clf.predict_proba(hist.reshape(1, -1))[0]
    idxs = np.argsort(probs)[::-1][:5]
    return [(classes[i], probs[i]) for i in idxs], hist

# =============================== Main ===============================
if __name__ == "__main__":
    # 1) Carga del modelo
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado.")

    # 2) Carga de datos preprocesados
    print("🔄 Cargando dataset preprocesado…")
    X, y = load_dataset(ROOT_DIR, JSON_FILE)
    classes = sorted(os.listdir(ROOT_DIR))
    feature_names = [f"bin_{i}" for i in range(X.shape[1])]

    # 3) Interpretabilidad global (feature importance)
    plot_feature_importance(clf)

    # 4) Predicción local + selección de clase para PDP
    top5, hist = predict_image_top5(clf, classes, IMAGE_PATH)
    print("\n🎯 Top‑5 Predicciones:")
    for lbl, p in top5:
        print(f"  {lbl}: {p*100:.2f}%")

    best_idx = classes.index(top5[0][0])

    # 5) PDP para los 3 bins más importantes y la clase predicha
    top_bins = np.argsort(clf.feature_importances_)[::-1][:3]
    print(f"\n🔍 PDP para bins {top_bins} y clase índice {best_idx} ({classes[best_idx]})")
    plot_partial_dependence(clf, X, top_bins, target=best_idx)

    # 6) SHAP local y global
    print("\n✨ Generando SHAP…")
    shap_analysis(clf, X, hist.reshape(1, -1), best_idx, feature_names)

    # 7) LIME
    print("\n✨ Generando LIME…")
    lime_analysis(clf, X, classes, hist.reshape(1, -1), feature_names)
