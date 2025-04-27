# -*- coding: utf-8 -*-
"""Computer Vision Tutorial #1 - Face Detection & Recognition

Nama : Athaya Naura Khalilah
NIM : 23/512716/PA/21899

Original file is located at
    https://colab.research.google.com/drive/1_OIbqV7N3v7qAwQyEUftWKBWDQjUl0js
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# GANTI: path lokal ke dataset kamu
dataset_dir = r'D:\Kuliah\Semester 4\PKAC - Computer Vision\Tutorial CV #1\images'

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Could not load image.')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# Contoh load sample image
sample_image_path = os.path.join(dataset_dir, 'George_W_Bush', '1.jpg')
sample_image, sample_image_gray = load_image(sample_image_path)

# Tampilkan gambar RGB
sampleImg_RGB = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
plt.imshow(sampleImg_RGB)
plt.axis('off')
plt.show()

# Tampilkan gambar grayscale
cv2.imshow('Sample Gray Image', sample_image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load semua gambar untuk dataset
images = []
labels = []
for root, dirs, files in os.walk(dataset_dir):
    if len(files) == 0:
        continue
    for f in files:
        _, image_gray = load_image(os.path.join(root, f))
        if image_gray is None:
            continue
        images.append(image_gray)
        labels.append(os.path.basename(root))

print(f"Total images: {len(images)}")

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

sample_faces = detect_faces(sample_image_gray, min_size=(50, 50))

def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

cropped_faces, _ = crop_faces(sample_image_gray, sample_faces)

plt.imshow(cropped_faces[0], cmap='gray')
plt.axis('off')
plt.show()

# Face Recognition — Eigenface
face_size = (128, 128)

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_flattened = face_resized.flatten()
    return face_flattened

X = []
y = []

for image, label in zip(images, labels):
    faces = detect_faces(image)
    cropped_faces, _ = crop_faces(image, faces)
    if len(cropped_faces) > 0:
        face_flattened = resize_and_flatten(cropped_faces[0])
        X.append(face_flattened)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y length: {len(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

# Mean centering
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face

pipe = Pipeline([
    ('centering', MeanCentering()),
    ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
    ('svc', SVC(kernel='linear', random_state=177))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

n_components = len(pipe.named_steps['pca'].components_)

ncol = 4
nrow = (n_components + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5 * nrow),
                         subplot_kw={'xticks': [], 'yticks': []})

eigenfaces = pipe.named_steps['pca'].components_.reshape((n_components, X_train.shape[1]))
for i, ax in zip(range(len(eigenfaces)), axes.flat):
    ax.imshow(eigenfaces[i].reshape(face_size), cmap='gray')
    ax.set_title(f'Eigenface {i+1}')

plt.tight_layout()
plt.show()

# Save model
with open('eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Prediction helper
def get_eigenface_score(X):
    X_pca = pipe[:2].transform(X)
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1)
    return eigenface_scores

def eigenface_prediction(image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces)

    if len(cropped_faces) == 0:
        return 'No face detected.'

    X_face = []
    for face in cropped_faces:
        face_flattened = resize_and_flatten(face)
        X_face.append(face_flattened)

    X_face = np.array(X_face)
    labels = pipe.predict(X_face)
    scores = get_eigenface_score(X_face)

    return scores, labels, selected_faces

sample_scores, sample_labels, sample_faces = eigenface_prediction(sample_image_gray)

print(sample_scores)
print(sample_labels)
print(sample_faces)

# Drawing functions
def draw_text(image, label, score, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=0.6, font_thickness=2, text_color=(0, 0, 0), text_color_bg=(0, 255, 0)):
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y - h1 - h2 - 25), (x + max(w1, w2) + 20, y), text_color_bg, -1)
    cv2.putText(image, label, (x + 10, y - 10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x + 10, y - h2 - 15), font, font_scale, text_color, font_thickness)

def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

result_image = draw_result(sample_image, sample_scores, sample_labels, sample_faces)

cv2.imshow('Recognition Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Webcam recognition
def webcam_recognition():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Mengubah lebar frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Mengubah tinggi frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result = eigenface_prediction(gray_frame)

        if isinstance(result, str):
            continue
        else:
            scores, labels, faces = result

        result_frame = draw_result(frame, scores, labels, faces)

        cv2.imshow('Webcam Face Recognition', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Memanggil fungsi webcam recognition
if __name__ == "__main__":
    webcam_recognition()
