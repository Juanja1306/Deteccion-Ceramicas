import os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib
import warnings
import lbp_hist as lbp

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

Image.MAX_IMAGE_PIXELS = None  # Deshabilitar el límite

# Cargar datos
print("Cargando dataset...")
X, y = lbp.load_dataset(ROOT_DIR)

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
        # # Entrenar el clasificador SVM
        # clf = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
        
        # Usar RandomForestClassifier en su lugar
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        
        clf.fit(X_train, y_train)
        # Guardar el checkpoint para este fold
        joblib.dump({'clf': clf}, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path} para fold {fold + 1}.")

    # Evaluar el modelo en el fold actual
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold + 1} Test Accuracy: {accuracy:.2f}%")

    # Guardar el mejor modelo global si se mejora la accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        joblib.dump(clf, MODEL_SAVE_PATH)
        print(f"Nuevo mejor modelo guardado con accuracy: {best_accuracy:.2f}%")

    fold_end_time = time.time()
    print(f"Tiempo total para Fold {fold + 1}: {fold_end_time - fold_start_time:.2f} segundos")

    # (Opcional) Eliminar el checkpoint del fold si ya no es necesario
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

final_accuracy = np.mean(accuracies)
print(f"\nFinal Average Test Accuracy: {final_accuracy:.2f}%")
print(f"Mejor modelo guardado en: {MODEL_SAVE_PATH} con accuracy: {best_accuracy:.2f}%")