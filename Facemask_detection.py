import cv2
import os
import mediapipe as mp
import joblib

mp_face_detection = mp.solutions.face_detection

LABELS = ["SIN_mascarilla", "CON_mascarilla"]

# Cargar el modelo SVM entrenado
svm_model = joblib.load("svm_model.pkl")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:  # Verificar si el fotograma está vacío
            break
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                if xmin < 0 and ymin < 0:
                    continue

                face_image = frame[ymin: ymin + h, xmin: xmin + w]
                if face_image is None:  # Verificar si la imagen de la cara está vacía
                    continue

                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_CUBIC)

                # Aplanar la imagen a un vector de características de longitud 10000
                face_image_vector = face_image.flatten()

                # Predecir utilizando el modelo SVM
                result = svm_model.predict([face_image_vector])[0]

                color = (0, 255, 0) if result == 0 else (0, 0, 255)
                cv2.putText(frame, "{}".format(LABELS[result]), (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
