import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
import threading
import os

# Load the trained model
model = load_model("models/sign_language_model.h5")
label_map = np.load("models/label_map.npy", allow_pickle=True).item()
labels = {v: k for k, v in label_map.items()}

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Sentence detection function
def detect_sentences(output_box):
    def detection_task():
        sentence = ""
        cap = cv2.VideoCapture(0)
        output_box.insert(tk.END, "Starting detection... Press 'q' in the camera window to quit.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks and predict
                    landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                                [lm.y for lm in hand_landmarks.landmark] + \
                                [lm.z for lm in hand_landmarks.landmark]
                    prediction = model.predict(np.array(landmarks).reshape(1, -1))
                    detected_label = labels[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100

                    # Append detected label to sentence
                    if confidence > 80:  # Confidence threshold
                        if detected_label == "SPACE":
                            sentence += " "
                        elif detected_label == "CLEAR":
                            sentence = ""
                        else:
                            sentence += detected_label

                        # Display sentence in GUI
                        output_box.delete(1.0, tk.END)
                        output_box.insert(tk.END, f"Sentence: {sentence}\n")
                        output_box.see(tk.END)

                        # Draw prediction on frame
                        h, w, _ = frame.shape
                        x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                        y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                        cv2.putText(frame, f"{detected_label} ({confidence:.1f}%)", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Sign Language Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        output_box.insert(tk.END, "Detection session ended.\n")

    threading.Thread(target=detection_task).start()

# GUI for Sentence Detection
def main():
    # Create main window
    root = tk.Tk()
    root.title("Sign Language to Sentence")

    # Create detection button
    output_box = tk.Text(root, width=80, height=20, wrap=tk.WORD, bg="lightyellow")
    output_box.grid(row=1, column=0, padx=10, pady=10)

    btn_detect = tk.Button(root, text="Start Detection", width=20, height=2, bg="lightcoral",
                           command=lambda: detect_sentences(output_box))
    btn_detect.grid(row=0, column=0, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
