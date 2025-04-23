# Detección Cerámicas

Deteccion Ceramicas es un proyecto orientado a la clasificación y detección de diferentes tipos de cerámicas basandonos unicamente en su diseño mediante técnicas de procesamiento de imágenes y aprendizaje automático. Se han implementado diversas metodologías que incluyen redes neuronales simples (NN), redes neuronales convolucionales (CNN), clasificadores tradicionales (SVM y Random Forest) y extracción de características basadas en LBP.

## Resultados
<div align="center">
    
| Algoritmo/Modelo | Acuracy | Confianza |
| :---: | :---: | :---: |
| LBP + Random Forest Classifier (RFC) |  95.95%  | 83.83% |
| LBP + KNN    | 91.63%  | 87.09%  |
| Convolutional Neural Network (CNN) |  59.32%  | 91.5%  |
| LBP + NN con Tensorflow    |  49.19%  | 55.32%  |
| LBP + NN con Pytorch    |  46.82%  | 51.24%  |
| LBP + KMeans    |  14.12%  | 99.97%  |
| LBP + Suport Vector Machine (SVM)    |  2.3%  | 10.86%  |


</div>

---

## Tabla de Contenidos

- [Algoritmos y Modelos](#algoritmos-y-modelos)
- [Checkpoints](#checkpoints)
- [Requisitos](#requisitos)
- [Dependencias](#dependencias)
- [Instalación](#instalación)
- [Estructura de Datos](#estructura-de-datos)
- [Predicción](#predicción)
- [Evaluación de Resultados](#evaluación-de-resultados)
- [Consideraciones](#consideraciones)

---

## Algoritmos y Modelos

Este proyecto tiene como objetivo detectar y clasificar imágenes de cerámicas basandonos unicamente en su diseño mediante varias técnicas de análisis visual. Para ello se han desarrollado múltiples módulos:

* Convolutional Neural Network (CNN): En `CNN.py` se emplea una CNN basada en ResNet50 preentrenada. Las capas convolucionales extraen características de alto nivel de las imágenes, y la última capa totalmente conectada se ajusta al número de clases. Esta arquitectura aprovecha transferencia de aprendizaje.

* Local Binary Pattern (LBP): Local Binary Pattern es una técnica de extracción de características basada en textura. Para cada píxel de la imagen en escala de grises, se comparan los valores de intensidad de los píxeles vecinos con el píxel central. Se genera un histograma que codifica patrones locales de textura normalizados. Se dispone de versiones tanto en CPU como aceleradas en GPU.

* Random Forest Classifier: Random Forest es un método de ensamble que construye múltiples árboles de decisión independientes y combina sus predicciones por votación. Permite robustez ante ruido y evita sobreajuste. Se usa en los scripts `LBP_RFC_o_SVM.py` y `LBP_RFC_GPU.py`

* K-Nearest Neighbors (KNN): clasifica una muestra nueva asignándole la etiqueta predominante entre sus k vecinos más cercanos en el espacio de características. En `LBP_KNN.py` se usa k=5 para predecir la clase basándose en histograma LBP.

* K-Means Classifier: es un algoritmo de clustering no supervisado que particiona los datos en k grupos según cercanía a centroides. En `LBP_KMeans.py`, tras extraer histogramas LBP, se agrupan en k clústeres y cada clúster se etiqueta con la clase mayoritaria de sus muestras. 

* Red Neuronal Feedforward (NN): En `LBP_NN_PyTorch.py` y `LBP_NN_Tensorflow.py` se define una red de perceptrones con dos capas ocultas (128 y 64 neuronas) y función de activación ReLU. La salida usa softmax para clasificación múltiple y se entrena con entropía cruzada.

Además, se proporcionan scripts para realizar predicciones sobre nuevas imágenes y evaluar el desempeño de los modelos mediante la generación de informes y tablas resumidas.

---

## Checkpoints

- Buscan archivos `checkpoint_fold{n}_latest.*` desde el último fold al primero.
- Guardan el estado del modelo (y optimizador en NN) periódicamente o al finalizar cada fold.
- Permiten interrumpir y reanudar el entrenamiento sin perder progreso.
- En algunos archivos, se eliminan automáticamente los folds cuya precisión (accuracy) sea inferior a la de los de mejor desempeño.

---

## Requisitos

`Python 3.12`

---

## Dependencias

* `PyTorch`

* `TorchVision`
* `torch`

* `NumPy`

* `scikit-learn`

* `scikit-image`

* `joblib`

* `Pandas`

* `Pillow (PIL)`

* `cupy` 

* `cuML` 

---

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Estructura de Datos

El proyecto asume que las imágenes se organizan en carpetas donde cada subcarpeta representa una clase (label) de cerámicas. Por ejemplo:
```
DATA/
└── Ruido/
    ├── Clase1/
    │   ├── imagen1.jpg
    │   ├── imagen2.jpg
    │   └── ...
    ├── Clase2/
    │   ├── imagen1.jpg
    │   ├── imagen2.jpg
    │   └── ...
    └── ...
```
Asegurarse de configurar la variable ROOT_DIR en cada script para apuntar a la ubicación correcta.

---

## Predicción

La prediccion es para una unica imagen, Para probar de uno en uno 

## Evaluación de Resultados

Los resultados en un `.csv` que nos da una tabla que compara cada label sobre su acuracy y confianza basado en el 20% de imagenes existentes

---

## Consideraciones

* **Parámetros de Transformación:** 
Asegurarse de que parámetros como el tamaño de imagen (IMAGE_SIZE) y las transformaciones (normalización, conversión a escala de grises o RGB) sean consistentes entre entrenamiento y predicción.
Ajustar parámetros (`NUM_FOLDS`, `BATCH_SIZE`, `IMAGE_SIZE`, `tasa de aprendizaje`) según hardware.

* **Uso de GPU:**
Si se cuenta con una GPU, utiliza LBP_GPU.py para aprovechar la aceleración mediante cupy y cuML.

* **Reanudación del Entrenamiento:**
Muchos scripts verifican la existencia de archivos checkpoint para reanudar el entrenamiento sin reiniciar desde cero.
El archivo JSON (`dataset.json`) acelera cargas en los archivos que usan LBP. Aquí se encuientra el histograma del LBP de las imágenes.

* **Espacio:**
Asegurarse de tener espacio en disco para checkpoints y modelos.