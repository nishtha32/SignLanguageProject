import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging

# Ensure directories exist
os.makedirs("detected_text", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(filename="logs/detected_text.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s")

# Load the trained model
model = load_model("models/sign_language_model.h5")
label_map = np.load("models/label_map.npy", allow_pickle=True).item()
labels = {v: k for k, v in label_map.items()}

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open detected text file
detected_text_file = open("detected_text/detected_text.txt", "w")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Extract hand landmarks for prediction
            landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                        [lm.y for lm in hand_landmarks.landmark] + \
                        [lm.z for lm in hand_landmarks.landmark]
            prediction = model.predict(np.array(landmarks).reshape(1, -1))
            predicted_label = labels[np.argmax(prediction)]

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put label text above the box
            cv2.putText(frame, predicted_label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Log and save the prediction
            logging.info(f"Detected: {predicted_label}")
            detected_text_file.write(predicted_label + "\n")

    # Display the video feed
    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detected_text_file.close()
cv2.destroyAllWindows()
