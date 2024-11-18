import os
import cv2
import time
import string

# Base directory to store data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Class labels (A-Z, 0-9)
labels = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z', '0', '1', ..., '9']

# Ask for folder name to create a new folder
folder_name = input("Enter the name of the new folder for this dataset: ").strip()
lang_data_dir = os.path.join(DATA_DIR, folder_name)

# Create the folder if it doesn't exist
if not os.path.exists(lang_data_dir):
    os.makedirs(lang_data_dir)

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

# Set camera resolution 
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Data collection for each class (letter or number)
for label in labels:
    class_dir = os.path.join(lang_data_dir, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Preparing to collect data for class {label}. Press 'Q' to start!")

    # Wait for the user to press 'Q' before starting data collection for each class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the live camera feed with instructions
        message = f'Press "Q" to start data collection for class {label}'
        
        # Display the message with adjusted font size and position
        cv2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with the message
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            print(f"Starting data collection for class {label}")
            break

    # Capture images for the current class
    counter = 0
    dataset_size = 500  # Number of images to collect for each class
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the current frame
        cv2.imshow('frame', frame)

        # Save the frame as a JPEG image
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        print(f"Captured image {counter+1} for class {label}")
        counter += 1

        # Press 'q' again to stop data collection early
        if cv2.waitKey(1) == ord('q'):
            print(f"Early exit from data collection for class {label}.")
            break

    print(f"Completed data collection for class {label}.")
    time.sleep(2)  # Short delay to allow preparation for the next class

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
