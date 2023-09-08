# Nadia Paola Ferro Gallegos - A01752013
# Se importan las bibliotecas necesarias
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

# Se carga el conjunto de datos
# Wine Dataset - Clasificacion Multiclase
wine = datasets.load_wine()
X, y = wine.data, wine.target

# Se crea un mapa de colores para las etiquetas
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Se visualiza el conjunto de datos
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# Se divide el conjunto de datos en conjuntos de entrenamiento(70%), validacion(15%) y prueba(15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=1234)

# Crear y entrenar el modelo KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)

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
k_values = [3, 5, 7, 9]

best_k = None
best_precision = 0.0

for k in k_values:
    temporary_model = KNN(k=k)
    temporary_model.fit(X_train, y_train)
    val_predictions = temporary_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    if val_accuracy > best_precision:
        best_precision = val_accuracy
        best_k = k

print("Mejor valor de 'k' en el conjunto de validacion:", best_k)

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
print("Metricas en el conjunto de prueba:")
print("Precision:", accuracy_test)
print("Precision ponderada:", precision_test)
print("Recall ponderado:", recall_test)
print("F1-score ponderado:", f1_test)
print("-------------------------------------------")

# Imprimir matriz de confusion para el modelo final
confusion_final = confusion_matrix(y_test, predictions_test)
print("Matriz de confusion para el modelo final:")
print(confusion_final)
