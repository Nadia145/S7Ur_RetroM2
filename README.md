# S7Ur_RetroM2
Este proyecto es una implementación de un clasificador K-Nearest Neighbors (KNN) para la clasificación de datos utilizando varios conjuntos de datos disponibles en sklearn. KNN es un algoritmo de aprendizaje supervisado que se utiliza para la clasificación de datos.

## Conjuntos de Datos

Este proyecto utiliza tres conjuntos de datos diferentes de sklearn:

1. **Iris Dataset**: Este conjunto de datos se utiliza como punto de partida y es un conjunto de datos clásico de clasificación multiclase que contiene características de flores de iris.

2. **Wine Dataset**: Este conjunto de datos se utiliza para la clasificación multiclase y contiene características relacionadas con la composición química de vinos de diferentes variedades.


3. **Breast Cancer Dataset**: Este conjunto de datos se utiliza para la clasificación binaria y contiene características relacionadas con el cáncer de mama.

## Dependencias

Este proyecto utiliza las siguientes bibliotecas de Python:

- `numpy`: Para cálculos numéricos.
- `scikit-learn`: Para cargar conjuntos de datos, dividir los datos y evaluar el rendimiento del modelo.
- `matplotlib`: Para visualizar los datos y resultados.

## Estructura del Proyecto

- `knn.py`: Contiene la implementación de la clase `KNN` que se utiliza para el clasificador KNN.
- `train1.py`: Entrena y evalúa el modelo KNN utilizando el conjunto de datos Iris.
- `train2.py`: Entrena y evalúa el modelo KNN utilizando el conjunto de datos Wine.
- `train3.py`: Entrena y evalúa el modelo KNN utilizando el conjunto de datos Breast Cancer.
- `README.md`: Este archivo que proporciona una descripción general del proyecto.

## Uso

Puedes ejecutar cada uno de los archivos `train1.py`, `train2.py` y `train3.py` para entrenar y evaluar el modelo KNN en los conjuntos de datos correspondientes.

```bash
python train1.py
python train2.py
python train3.py
