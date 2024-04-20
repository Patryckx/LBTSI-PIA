import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Función para cargar imágenes y etiquetas
def load_data(data_path):
    images = []
    labels = []
    label = 0
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (100, 100))  # Redimensionar todas las imágenes a 100x100
            images.append(image.flatten())  # Aplanar la imagen
            labels.append(label)
        label += 1
    return np.array(images), np.array(labels)

# Ruta de los datos
data_path = "C:/Users/User/3D Objects/LBTSI-PIA/Dataset_faces"
images, labels = load_data(data_path)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar los datos de píxeles a valores entre 0 y 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Definir y entrenar el clasificador SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = svm_classifier.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del clasificador SVM:", accuracy)

# Guardar el modelo entrenado
joblib.dump(svm_classifier, "svm_model.pkl")
print("Modelo SVM almacenado exitosamente.")
