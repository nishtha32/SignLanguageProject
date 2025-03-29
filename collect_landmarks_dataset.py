import cv2
import mediapipe as mp
import os
import csv

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def collect_landmark_data(output_dir, label):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    csv_file = os.path.join(output_dir, f"{label}_landmarks.csv")
    try:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)  # Add headers for 21 landmarks

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot access the webcam.")
                return

            print(f"Collecting landmarks for label: {label}. Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                                    [lm.y for lm in hand_landmarks.landmark] + \
                                    [lm.z for lm in hand_landmarks.landmark]
                        print(landmarks)  # Debugging: Print landmarks
                        writer.writerow(landmarks)

                cv2.imshow("Collect Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred during landmark collection: {e}")

if __name__ == "__main__":
    output_dir = input("Enter output directory for landmarks: ")
    label = input("Enter label for this gesture: ")
    collect_landmark_data(output_dir, label)
