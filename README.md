# **Face Recognition Using Multiple Methods**

Welcome to the **Face Recognition Using Multiple Methods** repository. This project presents three distinct approaches to face recognition, each leveraging different levels of complexity and flexibility:

1. **Pre-trained Model (`face_recognition` library)** – A simple and effective solution with minimal setup.
2. **Deep Learning with TensorFlow** – A robust, scalable approach utilizing TensorFlow’s deep learning capabilities.
3. **Custom CNN Model** – A fully customizable neural network trained from scratch for face recognition.

---
## **1. Face Recognition with the `face_recognition` Library**
### **Overview**
This method utilizes the `face_recognition` Python library, which is built on dlib’s powerful facial recognition framework.

### **Key Features**
✔️ Quick and easy setup  
✔️ Pre-trained models ensure high efficiency  
✔️ Minimal computational resources required  

### **Installation**
```bash
pip install face_recognition numpy opencv-python
```

### **Basic Usage**
```python
import face_recognition
import cv2

image = face_recognition.load_image_file("sample.jpg")
face_locations = face_recognition.face_locations(image)
```

---
## **2. Face Recognition with TensorFlow**
### **Overview**
This method leverages TensorFlow and deep learning techniques to perform facial recognition with a more flexible and scalable approach.

### **Key Features**
✔️ Higher accuracy with deep learning models  
✔️ Suitable for large-scale applications  
✔️ Customizable architecture for specific use cases  

### **Installation**
```bash
pip install tensorflow keras opencv-python numpy mtcnn
```

### **Basic Usage**
```python
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("face_recognition_model.h5")
```

---
## **3. Training a CNN Model from Scratch**
### **Overview**
This approach involves designing and training a **custom Convolutional Neural Network (CNN)** for face recognition, providing full control over model architecture and dataset.

### **Key Features**
✔️ Fully customizable deep learning pipeline  
✔️ Can be fine-tuned for specific datasets  
✔️ Requires substantial training but yields high accuracy  

### **Installation**
```bash
pip install tensorflow keras numpy opencv-python mtcnn matplotlib scikit-learn
```

### **Dataset Requirement**
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset) and place it in the `face_dataset` directory.

### **Model Architecture**
The CNN consists of **8 layers**, including **convolutional, pooling, dense, and dropout layers**.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### **Training the Model**
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
```

---
## **Real-Time Face Recognition Using a Webcam**
This implementation detects and classifies faces in real-time using a webcam.
```python
import cv2
from mtcnn import MTCNN

def predict_face(face):
    face = cv2.resize(face, (64, 64)) / 255.0
    face = np.expand_dims(face, axis=0)
    prediction = model.predict(face)
    return "Known Person" if np.argmax(prediction) == 0 else "Unknown"
```

---
## **Performance Comparison**
| **Method** | **Ease of Use** | **Accuracy** | **Training Required** |
|------------|---------------|------------|------------------|
| `face_recognition` Library | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | No |
| TensorFlow Pre-trained | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Minimal |
| CNN from Scratch | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Yes |

---
## **Conclusion**
This repository provides a comprehensive overview of three distinct face recognition techniques, catering to different needs and skill levels. Whether you seek a **ready-made solution**, a **deep learning-based approach**, or **full control over training**, this project offers the right tools.

🚀 **Choose your method, implement, and enhance your facial recognition system today!**

---
## **License**
This project is licensed under the **MIT License**.
