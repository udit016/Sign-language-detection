import os
import pickle
import warnings
import mediapipe as mp
import cv2
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To show only errors

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory with images
DATA_DIR = './data'
print(f"Using data directory: {DATA_DIR}")

data = []
labels = []

# Define the expected feature length
expected_feature_length = 84

# Verify if the directory exists
if not os.path.exists(DATA_DIR):
    print(f"Directory '{DATA_DIR}' does not exist.")
else:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.listdir(dir_path):
            print(f"Skipping empty directory: {dir_}")
            continue

        print(f"Processing directory: {dir_path}")

        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_.clear()
                    y_.clear()
                    data_aux.clear()

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Ensure feature length is 84
                    if len(data_aux) == expected_feature_length:
                        data.append(data_aux)
                        labels.append(dir_)
                    else:
                        # If feature length is incorrect, pad or truncate the data
                        if len(data_aux) < expected_feature_length:
                            data_aux = np.pad(data_aux, (0, expected_feature_length - len(data_aux)))
                        else:
                            data_aux = data_aux[:expected_feature_length]
                        
                        data.append(data_aux)
                        labels.append(dir_)

    # Save data using pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print("Data collection and saving complete.")

