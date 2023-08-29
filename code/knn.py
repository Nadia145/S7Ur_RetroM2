# Nadia Paola Ferro Gallegos - A01752013
import numpy as np
from collections import Counter

# Pasos para el KNN:
# 1. Elegir el valor de k (debe ser impar)
# 2. Calcular distancias
# 3. Seleccionar los k vecinos mas cercanos
# 4. Realizar predicciones
# 5. Evaluar y repetir


# Se define la funcion de euclidiana entre dos puntos x1 y x2
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


# Se define la clase KNN
class KNN:
    def __init__(self, k=3):
        self.k = k  # numero de vecinos

    # Se guardan los datos de entrenamiento
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Se realizan predicciones para un conjunto de datos de entrada x
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    # Se calcula la distancia entre x y todos los puntos de entrenamiento
    def _predict(self, x):
        distances = [
            euclidean_distance(x, x_train) for x_train in self.X_train
            ]

        # Se obtienen los indices de los k puntos mas cercanos
        k_indices = np.argsort(distances)[:self.k]

        # Obtener las etiquetras de los k puntos mas cercanos
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Encontrar la etiqueta mas comun entre los vecinos cercanos
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
