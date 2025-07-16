import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('../models/final_acne_model.keras')

# Class labels
class_labels = ['Clear Skin', 'Mild Acne', 'Severe Acne']

# Load face detection Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing function
def preprocess_frame(frame, target_size=(224, 224)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_y_cr_cb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(img_y_cr_cb)
    y_channel_eq = cv2.equalizeHist(y_channel)
    img_y_cr_cb_eq = cv2.merge((y_channel_eq, cr, cb))
    frame_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)
    frame_rgb_eq = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb_eq, target_size)
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)
    return input_image

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Predict
    input_image = preprocess_frame(frame)
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]

    # Set background color based on prediction
    overlay_color = (0, 255, 0)  # Green
    if label == 'Mild Acne':
        overlay_color = (0, 255, 255)  # Yellow
    elif label == 'Severe Acne':
        overlay_color = (0, 0, 255)  # Red

    # Create a semi-transparent overlay
    overlay = frame.copy()
    output = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), overlay_color, -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Detect faces (or regions) for drawing bounding box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 255, 255), 2)  # White box
        cv2.putText(output, 'Detected Region', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw prediction label
    text = f"{label} ({confidence*100:.2f}%)"
    cv2.putText(output, text, (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the output frame
    cv2.imshow('Acne Detection - Live', output)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
webcam.release()
cv2.destroyAllWindows()
