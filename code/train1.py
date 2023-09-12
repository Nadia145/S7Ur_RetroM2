# Nadia Paola Ferro Gallegos - A01752013
# Se importan las bibliotecas necesarias
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Se carga el conjunto de datos
# Iris Dataset - Clasificacion Multiclase
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Se crea un mapa de colores para las etiquetas
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Se visualiza el conjunto de datos
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.title('Conjunto de datos')
plt.xlabel('Caracteristica X')
plt.ylabel('Caracteristica y')
plt.show()

# Se divide el conjunto de datos en conjuntos de entrenamiento(70%), validacion(15%) y prueba(15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=1234)

# Se visualiza el conjunto de datos de entrenamiento
plt.figure()
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train,
            cmap=cmap, edgecolors='k', s=20)
plt.title('Conjunto de entrenamiento')
plt.xlabel('Caracteristica X')
plt.ylabel('Caracteristica y')
plt.show()

# Se visualiza el conjunto de datos de validacion
plt.figure()
plt.scatter(X_val[:, 2], X_val[:, 3], c=y_val, cmap=cmap, edgecolors='k', s=20)
plt.title('Conjunto de Validacion')
plt.xlabel('Caracteristica X')
plt.ylabel('Caracteristica y')
plt.show()

# Se visualiza el conjutno de datos de prueba
plt.figure()
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test,
            cmap=cmap, edgecolors='k', s=20)
plt.title('Conjunto de Prueba')
plt.xlabel('Caracteristica X')
plt.ylabel('Caracteristica y')
plt.show()

# Crear y entrenar el modelo KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)

# Realizar predicciones en el conunto de entrenamiento
predictions_train = clf.predict(X_train)

# Calcular metricas en el conjunto de entrenamiento
accuracy_train = accuracy_score(y_train, predictions_train)
precision_train = precision_score(
    y_train, predictions_train, average='weighted')
recall_train = recall_score(y_train, predictions_train, average='weighted')
f1_train = f1_score(y_train, predictions_train, average='weighted')

# Imprimir metricas en el conjunto de entrenamiento
print("-------------------------------------------")
print("Metricas en el conjunto de entrenamiento:")
print("Precision:", accuracy_train)
print("Precision ponderada:", precision_train)
print("Recall ponderado:", recall_train)
print("F1-score ponderado:", f1_train)

# Realizar predicciones en el conjunto de validacion
predictions_val = clf.predict(X_val)

# Calcular metricas en el conjunto de validacion
accuracy_val = accuracy_score(y_val, predictions_val)
precision_val = precision_score(y_val, predictions_val, average='weighted')
recall_val = recall_score(y_val, predictions_val, average='weighted')
f1_val = f1_score(y_val, predictions_val, average='weighted')

# Imprimir metricas en el conjunto de validacion
print("-------------------------------------------")
print("Metricas en el conjunto de validacion:")
print("Precision:", accuracy_val)
print("Precision ponderada:", precision_val)
print("Recall ponderado:", recall_val)
print("F1-score ponderado:", f1_val)

# Realizar predicciones en el conjunto de prueba
predictions_test = clf.predict(X_test)

# Calcular metricas en el conjunto de prueba
accuracy_test = accuracy_score(y_test, predictions_test)
precision_test = precision_score(y_test, predictions_test, average='weighted')
recall_test = recall_score(y_test, predictions_test, average='weighted')
f1_test = f1_score(y_test, predictions_test, average='weighted')

# Imprimir metricas en el conjunto de prueba
print("-------------------------------------------")
print("Metricas en el conjunto de prueba:")
print("Precision:", accuracy_test)
print("Precision ponderada:", precision_test)
print("Recall ponderado:", recall_test)
print("F1-score ponderado:", f1_test)

# Se ajustan los hiperparametros utilizando el conjunto de validacion
k_values = [1, 3, 5, 7, 9]

# Se declaran las variables para determinar el mejor valor de k
best_k = None
best_precision = 0.0

# Listas para almacenar las metricas
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Utilizamos los hiperparametros en el modelo
for k in k_values:
    # Crear y entrenar el modelo KNN
    temporary_model = KNN(k=k)
    temporary_model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de validacion
    val_predictions = temporary_model.predict(X_val)

    # Calcular metricas
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average="weighted")
    val_recall = recall_score(y_val, val_predictions, average="weighted")
    val_f1 = f1_score(y_val, val_predictions, average="weighted")

    # Almacenar metricas en las listas
    accuracy_scores.append(val_accuracy)
    precision_scores.append(val_precision)
    recall_scores.append(val_recall)
    f1_scores.append(val_f1)

    # Determinar el mejor valor de k
    if val_accuracy > best_precision:
        best_precision = val_accuracy
        best_k = k

# Se notifica al usuario cual es el mejor valor de k
print("Mejor valor de 'k' en el conjunto de validacion:", best_k)

# Se crea una grafica comparativa
plt.plot(k_values, precision_scores, marker='o')
plt.xlabel('Valor de k')
plt.ylabel('precision ponderada')
plt.title('Precision ponderada vs. Valor de k')
plt.grid(True)
plt.show()

# Se entrena el modelo final con todos los datos de entrenamiento utilizando el mejor valor de k encontrado
modelo_final = KNN(k=best_k)
modelo_final.fit(X_train, y_train)

# Se realizan predicciones en datos de prueba (sin usar el conjunto de validacion)
predictions_test = modelo_final.predict(X_test)

# Evaluar el rendimiento del modelo final en datos prueba
accuracy_test = accuracy_score(y_test, predictions_test)
precision_test = precision_score(y_test, predictions_test, average='weighted')
recall_test = recall_score(y_test, predictions_test, average='weighted')
f1_test = f1_score(y_test, predictions_test, average='weighted')

# Imprimir metricas en el conjunto de prueba para el modelo final
print("-------------------------------------------")
print("Metricas en el conjunto de prueba para el modelo final:")
print("Precision:", accuracy_test)
print("Precision ponderada:", precision_test)
print("Recall ponderado:", recall_test)
print("F1-score ponderado:", f1_test)
print("-------------------------------------------")

# Imprimir matriz de confusion para el modelo final
confusion_final = confusion_matrix(y_test, predictions_test)
print("Matriz de confusion para el modelo final:")
print(confusion_final)

# Escala los conjuntos de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Entrena el modelo KNN con los conjuntos de datos escalados
clf = KNN(k=5)
clf.fit(X_train_scaled, y_train)

# Realiza predicciones en los conjuntos de validación y prueba escalados
predictions_val = clf.predict(X_val_scaled)
predictions_test = clf.predict(X_test_scaled)


# Definir la cuadrícula de valores de 'k' para buscar
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}

# Crear un clasificador KNN
knn = KNeighborsClassifier()

# Realizar una búsqueda de cuadrícula para encontrar el mejor valor de 'k'
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor valor de 'k' y entrenar el modelo con él
best_k = grid_search.best_params_['n_neighbors']
clf = KNN(k=best_k)
clf.fit(X_train_scaled, y_train)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Antes de la selección del mejor valor de 'k'
print("Antes de la selección del mejor valor de 'k':")
clf_before = KNeighborsClassifier(n_neighbors=3)  # Valor de 'k' arbitrario
clf_before.fit(X_train_scaled, y_train)
accuracy_before = clf_before.score(X_test_scaled, y_test)
print("Precisión antes de la selección de 'k':", accuracy_before)

# Selección del mejor valor de 'k' utilizando validación cruzada
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_k = grid_search.best_params_['n_neighbors']

# Después de la selección del mejor valor de 'k'
print("\nDespués de la selección del mejor valor de 'k':")
clf_after = KNeighborsClassifier(n_neighbors=best_k)
clf_after.fit(X_train_scaled, y_train)
accuracy_after = clf_after.score(X_test_scaled, y_test)
print("Mejor valor de 'k' encontrado:", best_k)
print("Precisión después de la selección de 'k':", accuracy_after)

# Validación cruzada para evaluar el modelo después de la selección de 'k'
cv_scores = cross_val_score(clf_after, X_train_scaled, y_train, cv=5)
mean_cv_accuracy = np.mean(cv_scores)

print("\nValidación cruzada (k-fold) después de la selección de 'k':")
print("Puntajes de Validación Cruzada:", cv_scores)
print("Precisión Promedio:", mean_cv_accuracy)
