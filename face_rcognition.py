import cv2
import face_recognition
import matplotlib.pyplot as plt

# Load a sample picture and learn how to recognize it
known_image = face_recognition.load_image_file("C:\\Users\\Naman Singh\\Pictures\\Camera Roll\\WhatsAp.JPG")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_faces = [known_encoding]
names = ["Naman"]

# Start webcam
video_capture = cv2.VideoCapture(0)
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue  # Keep looping even if a frame is not read correctly
        
        # Convert frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the resulting frame using Matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Face Recognition")
        plt.pause(0.01)
        
except KeyboardInterrupt:
    print("Closing the webcam...")
finally:
    plt.ioff()  # Disable interactive mode
    plt.show()
    video_capture.release()
    cv2.destroyAllWindows()
