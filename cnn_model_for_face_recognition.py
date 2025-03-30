import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
import os

mtcnn = MTCNN()

# Function to extract face from image
def extract_face(image):
    faces = mtcnn.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        return image[y:y+h, x:x+w]
    return None


X, y = [], []
labels = {"Known Person": 0, "Unknown": 1}  
# download data set from ::   https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset

data_path = "face_dataset"
for label in os.listdir(data_path):
    label_path = os.path.join(data_path, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            face = extract_face(img)
            if face is not None:
                face = cv2.resize(face, (64, 64)) / 255.0
                X.append(face)
                y.append(labels[label])

X = np.array(X)
y = np.array(y)
y = to_categorical(y, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model 8 layers 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("face_recognition_cnn.h5")

# webcam 
video_capture = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

def predict_face(face):
    face = cv2.resize(face, (64, 64)) / 255.0
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)
    return "Known Person" if np.argmax(prediction) == 0 else "Unknown"

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn.detect_faces(rgb_frame)
        
        for face in faces:
            x, y, w, h = face['box']
            detected_face = rgb_frame[y:y+h, x:x+w]
            name = predict_face(detected_face)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Face Recognition")
        plt.pause(0.01)
        
except KeyboardInterrupt:
    print("Closing the webcam...")
finally:
    plt.ioff()
    plt.show()
    video_capture.release()
    cv2.destroyAllWindows()
