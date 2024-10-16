import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7)

# Initialize OpenCV camera
cap = cv2.VideoCapture(0)  # Change the index if necessary (try 1 or 2 if 0 doesn't work)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Couldn't capture frame. Make sure the camera is accessible.")
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract hand landmarks
            x_ = []
            y_ = []
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            data_aux = []
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure data has the correct length (84)
            if len(data_aux) < 84:
                data_aux = np.pad(data_aux, (0, 84 - len(data_aux)))  # Pad with zeros if shorter
            elif len(data_aux) > 84:
                data_aux = data_aux[:84]  # Truncate if longer

            data_aux = np.array(data_aux).reshape(1, -1)
            prediction_proba = model.predict_proba(data_aux)
            prediction = model.predict(data_aux)

            # Get the index of the class with the highest probability
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[0][predicted_class]

            # Display the predicted class and confidence
            label = str(prediction[0])
            cv2.putText(frame, f'Hand Sign: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with hand landmarks and prediction
    cv2.imshow('Hand Sign Detection', frame)

    # Press 'q' to quit the camera stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
