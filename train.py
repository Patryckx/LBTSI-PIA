import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

dataPath = "C:/Users/User/3D Objects/LBTSI-PIA/Dataset_faces"
dir_list = os.listdir(dataPath)
print("Lista archivos:", dir_list)

labels = []
facesData = []

for label, name_dir in enumerate(dir_list):
    dir_path = os.path.join(dataPath, name_dir)
    
    for file_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, file_name)
        print(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer la imagen en escala de grises
        image = cv2.resize(image, (100, 100))  # Redimensionar la imagen a 100x100 píxeles

        facesData.append(image.flatten())  # Aplanar la imagen
        labels.append(label)

print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1))

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(facesData, labels, test_size=0.2, random_state=42)

# Definir y entrenar el clasificador SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Calcular la precisión en el conjunto de prueba
accuracy = svm_classifier.score(X_test, y_test)
print("Precisión del clasificador SVM:", accuracy)

# Guardar el modelo entrenado
joblib.dump(svm_classifier, "svm_model.pkl")
print("Modelo SVM almacenado exitosamente.")
