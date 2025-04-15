import os
import joblib
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern  # type: ignore

# =============================== Configuraciones ===============================
# Parámetros para LBP
RADIUS = 3             # Radio del LBP
N_POINTS = 8 * RADIUS  # Número de puntos alrededor del pixel central
METHOD = 'uniform'     # Método de cálculo del LBP
IMAGE_SIZE = (512, 512)  # Tamaño deseado para las imágenes

# Ruta a la carpeta raíz con subcarpetas (cada una es una etiqueta)
# Ruta del modelo guardado
MODEL_PATH = r"C:\Users\juanj\Desktop\Deteccion-Ceramicas\Paths\mejor_modelo_lbp.pkl"

# =============================== Función para extraer LBP ===============================
def compute_lbp(image_path):
    """
    Calcula y retorna el histograma normalizado del descriptor LBP de una imagen.
    """
    try:
        # Abrir la imagen en escala de grises y redimensionar
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None
    # Convertir la imagen a matriz numpy y calcular LBP
    gray_image = np.array(image)
    lbp = local_binary_pattern(gray_image, P=N_POINTS, R=RADIUS, method=METHOD)
    # Calcular histograma del LBP con número de bins adecuado
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalizar el histograma
    return hist

# =============================== Cargar el Modelo ===============================
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente desde:", MODEL_PATH)
else:
    print("No se encontró el modelo en la ruta especificada.")
    exit()

# =============================== Obtener las Clases ===============================
# Se asume que cada subcarpeta en ROOT_DIR corresponde a una etiqueta
classes = ['ALABAMA', 'ALMA', 'ALTAR', 'ALTEZZA', 'AMALFI', 'AMALTA', 'AMAZONIA', 'AMELIE', 
           'ANDROMEDA', 'ANTARES', 'AQUARIUM', 'ARGEL', 'ARIZONA', 'ARRECIFE', 'ARTICO', 
           'ASTI', 'ATHENAS', 'BALI', 'BASALTINA', 'BASALTO', 'BASIC', 'BELEK', 'BLACKSTONE', 
           'BLOSSOM', 'BOLONIA', 'BOREAL', 'BOSTON', 'BRESCIA', 'BROOKLYN', 'CALIFORNIA', 'CALIZA', 
           'CARRARA', 'CATANIA', 'CEMENTO', 'CEMENTO DECORADO', 'CEOL', 'CEPPO', 'CHARLOTE', 'CHLOE', 
           'CIPRES', 'CLOUDY_FOREST', 'CONCEPT', 'CONCRET', 'CONCRETO', 'CONTEMPO', 'CORAL', 'COTTO', 
           'CRETA', 'DAKAR', 'DAKARI', 'DAKOTA', 'DANAE', 'DESERT', 'DOLOMITE', 'DORNE', 'ECOSTONE', 
           'EMPERADOR', 'ESPACATOS', 'FACTORY', 'FENIX', 'FERRAN', 'FLOREANA', 'FLORENCIA', 'FLORENTINO', 
           'FLORIAN', 'FORESTA', 'FORMATO', 'FRANCINE', 'FUSION', 'GALENO', 'GALIA', 'GRANITY', 'GRECO', 
           'GRETA', 'GUAYACAN', 'HELI', 'HERMES', 'HYDRA', 'IBIZA', 'IPANEMA', 'ISABELA', 'IVORY', 'KAMANI', 
           'KANSAS', 'KAPUR', 'KYOTO_DREAM', 'LAJA', 'LEBLON', 'LIMESTONE', 'LINEN', 'LINO', 'LITIO', 'MADISON', 
           'MAJESTIC_BLUE', 'MALIBU', 'MANHATAN', 'MAUI', 'METAL', 'METALICOS', 'MIAMI', 'MONET', 'MONTANA', 'MORMONT', 
           'MURANO', 'NANTES', 'NARVI', 'NATURA', 'NATURAL', 'NATURE', 'NAVONA', 'NEBRASKA', 'NEO', 'OAKLAND', 
           'OKLAHOMA', 'OLIMPIA', 'OLIMPO', 'ONICE', 'OREGON', 'ORWELL', 'PADUA', 'PALIO', 'PALMIRA', 
           'PAMPLONA', 'PANDORA', 'PARQUET', 'PEBBLE', 'PERIGATAN', 'PLASTER', 'PLATINUM', 'POMPEYA', 
           'PRADO', 'QUARESTONE', 'QUEBEC', 'REFLECTION', 'RIVENDEL', 'ROCKY', 'ROMA', 'ROMA DECO', 
           'SANDSTONE', 'SANT', 'SANTA CLARA', 'SECOYA', 'SIENA', 'SIERRA', 'SILVERADO', 'SLATEWORK', 
           'SOFT_SABANA', 'SOTILE', 'STATUARIO', 'STELAR', 'STONE PRO', 'TDF', 'TERSO', 'TIRRENO', 
           'TORINO', 'TOULOUSE', 'TRAVERTINE', 'TRITON', 'TROPICAL_LINEN', 'TUNDRA', 'TURIN', 'URBA', 
           'URBAN_JUNGLE', 'UTAH', 'UYUNI', 'VENECIA', 'VENETTO', 'VESTRUM', 'WOODLAND', 'ZINERVA']

# =============================== Función de Predicción para una Imagen ===============================
def predict_image_top5(image_path):
    """
    Realiza la predicción de la clase para una imagen y retorna una lista con los 5 resultados
    más probables ordenados de mayor a menor probabilidad.
    
    Retorna:
      - Una lista de tuplas (etiqueta, probabilidad)
        donde el primer elemento es la predicción con mayor probabilidad y los siguientes son los 4 que le siguen.
    """
    features = compute_lbp(image_path)
    if features is None:
        print("No se pudo extraer las características de la imagen.")
        return None
    # Convertir el vector de características a formato 2D para la predicción
    features = features.reshape(1, -1)
    # Obtener el array de probabilidades para cada clase
    prob_array = clf.predict_proba(features)[0]
    # Ordenar los índices de mayor a menor probabilidad y obtener los 5 primeros
    top5_indices = np.argsort(prob_array)[::-1][:5]
    # Mapear cada índice a la etiqueta y su probabilidad
    top5 = [(classes[idx], prob_array[idx]) for idx in top5_indices]
    return top5

# =============================== Ejecución: Predecir para una Imagen ===============================
if __name__ == "__main__":
    image_path = input("Ingrese la ruta de la imagen a predecir: ").strip()
    if not os.path.exists(image_path):
        print("La ruta especificada no existe.")
    else:
        results = predict_image_top5(image_path)
        if results is not None:
            print("\nTop-5 Predicciones:")
            # Imprime la predicción principal y luego las 4 siguientes
            print(f"Mejor coincidencia: {results[0][0]} con una confianza de {results[0][1]*100:.2f}%")
            print("Otras coincidencias:")
            for label, conf in results[1:]:
                print(f"  {label}: {conf*100:.2f}%")
        else:
            print("No se pudo realizar la predicción.")
