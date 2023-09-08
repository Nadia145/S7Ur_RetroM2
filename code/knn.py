# Nadia Paola Ferro Gallegos - A01752013
# Se importan las bibliotecas necesarias
import numpy as np
from collections import Counter

# Pasos para el KNN:
# 1. Elegir el valor de k (debe ser impar)
# 2. Calcular distancias
# 3. Seleccionar los k vecinos mas cercanos
# 4. Realizar predicciones
# 5. Evaluar y repetir


# Se define la funcion de distancia euclidiana entre dos puntos x1 y x2
def euclidean_distance(x1, x2):
    """
    Calcula la distancia euclidiana entre dos puntos.

    Args:
        x1: (numpy.array): Primer punto.
        x2: (numpy.array): Segundo punto.

        Returns: float la distancia euclidiana entre x1 y x2.
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


# Se define la clase KNN
class KNN:
    def __init__(self, k=3):
        """
        Inicializa un clasificador KNN.

        Args:
            k (int): El número de vecinos a conciderar para la clasificación (debe ser impar).

        Raises:
            ValueError: Si k es par.
        """
        if k % 2 == 0:
            raise ValueError(
                "El valor de 'k' debe ser impar para garantizar una decision en caso de empate.")
        self.k = k  # numero de vecinos

    # Se guardan los datos de entrenamiento
    def fit(self, X, y):
        """
        Entrena el clasificador KNN con datos de entrenamiento.

        Args:
            X (numpy.ndarray): Los datos de entrenamiento, donde cada fila es una muestra y cada columna es una caracteristica.
            y (numpy.ndarray): Las etiquetas correspondientes a los datos de entrenamiento.

        Returns:
            None
        """
        self.X_train = X
        self.y_train = y

    # Se realizan predicciones para un conjunto de datos de entrada x
    def predict(self, X):
        """
        Realiza predicciones para un conjunto de datos de entrada.

        Args:
            X (numpy.ndarray): Los datos de entrada para los cuales se realizan predicciones.

        Returns:
            list: Una lista de las etiquetas predichas para cada dato de entrada en X.
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    # Se calcula la distancia entre x y todos los puntos de entrenamiento
    def _predict(self, x):
        """
        Realiza una prediccion para un solo dato de entrada.

        Args:
            x (numpy.ndarray): El dato de entrada para el cual se realizará la predicción.

            Returns:
                int: La etiqueta predicha para x.
        """
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
