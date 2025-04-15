# Detección Cerámicas

Deteccion Ceramicas es un proyecto orientado a la clasificación y detección de diferentes tipos de cerámicas mediante técnicas de procesamiento de imágenes y aprendizaje automático. Se han implementado diversas metodologías que incluyen redes neuronales convolucionales, extracción de características basadas en LBP, reducción de dimensionalidad con PCA y clasificadores tradicionales (como SVM y Random Forest).


## Tabla de Contenidos

- [Descripción](#descripción)
- [Organización del Proyecto](#organización-del-proyecto)
- [Requisitos](#requisitos)
- [Dependencias](#dependencias)
- [Instalación](#instalación)
- [Estructura de Datos](#estructura-de-datos)
- [Instrucciones de Uso](#instrucciones-de-uso)
- [Predicción](#predicción)
- [Consideraciones](#consideraciones)

---

## <span style="color:red;">Descripción</span>

Este proyecto tiene como objetivo detectar y clasificar imágenes de cerámicas mediante varias técnicas de análisis visual. Para ello se han desarrollado múltiples módulos:

* CNN: Utiliza una red neuronal basada en ResNet50 preentrenada para extraer características y clasificar las imágenes usando validación cruzada (KFold) y checkpointing para reanudar entrenamientos.

* LBP (Local Binary Pattern): Extrae características locales de las imágenes mediante LBP para entrenar clasificadores con SVM o Random Forest. Se dispone de versiones tanto en CPU como aceleradas en GPU.

* PCA: Implementa una reducción de dimensionalidad con Análisis de Componentes Principales (PCA) seguida de regresión logística para la clasificación.

* Clasificadores Incrementales: Incluye scripts para entrenar modelos basados en Random Forest (con entrenamiento incremental vía warm_start) y SVM incremental (con SGDClassifier).

Además, se proporcionan scripts para realizar predicciones sobre nuevas imágenes y evaluar el desempeño de los modelos mediante la generación de informes y tablas resumidas.

---

## <span style="color:red;">Organización del Proyecto</span>

El repositorio incluye los siguientes archivos principales:

* CNN.py: Entrena una CNN basada en ResNet50 utilizando validación cruzada, checkpoints periódicos y reanudación de entrenamiento.

* LBP.py: Implementa la extracción de características basadas en LBP y entrena clasificadores (SVM / Random Forest) con validación cruzada.

* LBP_GPU.py: Versión acelerada en GPU para la extracción y clasificación basada en LBP, utilizando bibliotecas como cupy y cuML.

* PCA.py: Aplica PCA para reducir la dimensionalidad de las imágenes y entrena un clasificador basado en regresión logística.

* RandomTree.py: Entrena un modelo Random Forest de forma incremental usando la técnica de warm_start y validación cruzada.

* SMV.py: Utiliza un clasificador SVM incremental (SGDClassifier) basado en características extraídas por un modelo preentrenado.

* Prediccion.py: Script para cargar un modelo CNN entrenado y realizar predicciones sobre nuevas imágenes.

* prediccionLBP.py: Permite predecir la clase de una imagen utilizando un modelo entrenado con características LBP, mostrando las clasificaciones más probables junto con su nivel de confianza.

* resultados.py: Evalúa el desempeño de la CNN mediante el muestreo de imágenes de cada categoría y genera un informe estadístico.

* resultadosLBP.py: Similar a resultados.py pero para los modelos basados en LBP, generando un resumen de resultados en formato de tabla y exportándolo a CSV.

---

## <span style="color:red;">Requisitos</span>
Python 3.12

---

## Dependencias
* PyTorch

* TorchVision

* NumPy

* scikit-learn

* joblib

* Pandas

* Pillow (PIL)

* scikit-image

* cupy (para LBP_GPU.py)

* cuML (para LBP_GPU.py)

---

## Instalación
```bash
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
Asegúrate de configurar la variable ROOT_DIR en cada script para apuntar a la ubicación correcta de tus datos.

---

## Instrucciones de Uso
### Entrenamiento
Cada módulo tiene su propio script de entrenamiento:

* CNN.py:
Para entrenar la red neuronal CNN con ResNet50, usa:

```bash
python CNN.py
```

* LBP.py / LBP_GPU.py:
Para entrenar un modelo usando características LBP (en CPU o acelerado en GPU):

```bash
python LBP.py
```
o
```bash
python LBP_GPU.py
```

* PCA.py:
Para ejecutar la reducción de dimensionalidad y entrenamiento con PCA:

```bash
python PCA.py
```

RandomTree.py:
Para entrenar un clasificador Random Forest con entrenamiento incremental:

```bash
python RandomTree.py
```

* SMV.py:
Para entrenar un clasificador SVM incremental:

```bash
python SMV.py
```

Checkpointing:
La mayoría de los scripts implementan checkpoints para guardar el progreso y permitir la reanudación en caso de interrupciones.

---

## Predicción

* Prediccion.py:
Para realizar predicciones utilizando el modelo CNN entrenado, ejecuta:

```bash
python Prediccion.py
```

* prediccionLBP.py:
Para predecir usando el modelo basado en LBP y obtener las clasificaciones más probables:

```bash
python prediccionLBP.py
```

### Evaluación de Resultados
* resultados.py:
Evalúa el desempeño del modelo CNN sobre una muestra de imágenes para cada categoría y muestra estadísticas de precisión y confianza.

```bash
python resultados.py
```

* resultadosLBP.py:
Similar al anterior, pero para modelos basados en LBP. Genera un informe y exporta los resultados a un archivo CSV.

```bash
python resultadosLBP.py
```

---

## Consideraciones
* Parámetros de Transformación:
Asegúrate de que parámetros como el tamaño de imagen (IMAGE_SIZE) y las transformaciones (normalización, conversión a escala de grises o RGB) sean consistentes entre entrenamiento y predicción.

* Uso de GPU:
Si cuentas con una GPU, utiliza LBP_GPU.py para aprovechar la aceleración mediante cupy y cuML.

Reanudación del Entrenamiento:
Muchos scripts verifican la existencia de archivos checkpoint para reanudar el entrenamiento sin reiniciar desde cero.
