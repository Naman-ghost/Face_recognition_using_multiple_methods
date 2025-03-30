import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine

# Load pre-trained FaceNet model for face recognition
facenet_model = load_model("facenet_keras.h5")
mtcnn = MTCNN()

# Function to get face embedding
def get_embedding(model, face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = face_pixels.astype("float32") / 255.0
    face_pixels = np.expand_dims(face_pixels, axis=0)
    return model.predict(face_pixels)[0]

# Load and encode a known image
known_image = cv2.imread("C:\\Users\\Naman Singh\\Pictures\\Camera Roll\\WhatsAp.JPG")
known_faces = []
names = ["Naman"]
face_data = mtcnn.detect_faces(known_image)
if face_data:
    x, y, w, h = face_data[0]['box']
    face = known_image[y:y+h, x:x+w]
    known_faces.append(get_embedding(facenet_model, face))

# Start webcam
video_capture = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

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
            face_embedding = get_embedding(facenet_model, detected_face)
            
            name = "Unknown"
            for known_face, known_name in zip(known_faces, names):
                similarity = 1 - cosine(known_face, face_embedding)
                if similarity > 0.5:  # Threshold for recognition
                    name = known_name
                    break
            
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
