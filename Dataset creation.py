import os
import pickle
import warnings
import mediapipe as mp
import cv2
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To show only errors

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Base directory to store data
DATA_DIR = './data/asl'

print(f"Using data directory: {DATA_DIR}")

data = []
labels = []

# Define the expected feature length (21 landmarks * 2 for x and y)
expected_feature_length = 84

# Verify if the directory exists
if not os.path.exists(DATA_DIR):
    print(f"Directory '{DATA_DIR}' does not exist.")
else:
    # Loop through each letter/number folder (e.g., 'A', 'B', ..., 'Z', '0', '1', ...)
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)

        # Skip if the item is not a directory (we expect directories for each letter/number)
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing directory: {dir_path}")

        # Loop through each image in the subdirectory (e.g., 'A', 'B', etc.)
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)

            # Check if the file has a valid image extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            if not any(img_path.lower().endswith(ext) for ext in valid_extensions):
                print(f"Skipping non-image file: {img_path}")
                continue

            # Read the image
            img = cv2.imread(img_full_path)

            # If image is not read successfully, skip it
            if img is None:
                print(f"Failed to read image: {img_full_path}")
                continue

            # Convert the image to RGB for MediaPipe processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # If landmarks are detected, process them
            if results.multi_hand_landmarks:
                print(f"Landmarks detected in {img_full_path}")
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract x and y coordinates for each landmark
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                    # Normalize the coordinates by subtracting the minimum x and y values
                    data_aux = []
                    min_x, min_y = min(x_coords), min(y_coords)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)  # Normalize x
                        data_aux.append(landmark.y - min_y)  # Normalize y

                    # Ensure the feature length is 84 (21 landmarks * 2 for x and y)
                    if len(data_aux) < expected_feature_length:
                        # Pad with zeros if the length is shorter than expected
                        data_aux = np.pad(data_aux, (0, expected_feature_length - len(data_aux)), mode='constant')
                    elif len(data_aux) > expected_feature_length:
                        # Truncate if the length is longer than expected
                        data_aux = data_aux[:expected_feature_length]

                    # Append the feature data and label (class) to the dataset
                    labels.append(dir_)  # Use the folder name (A-Z, 0-9) directly as the label
                    data.append(data_aux)

    # Save the data and labels using pickle
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print("Data collection and saving complete.")
