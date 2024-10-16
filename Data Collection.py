import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Initialize camera
cap = cv2.VideoCapture(0)  # Change index to 1 or 2 if necessary

# Check if the camera opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

for j in range(number_of_classes):
    # Create a directory for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Preparing to collect data for class {j}. Press 'Q' to start!")

    # Wait for the user to press 'Q' before starting data collection for each class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the live camera feed with instructions
        cv2.putText(frame, 'Ready? Press "Q" to start for class {}'.format(j), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for user to press 'q' to start data collection
        if cv2.waitKey(1) == ord('q'):
            print(f"Starting data collection for class {j}")
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the current frame
        cv2.imshow('frame', frame)

        # Save the frame as a JPEG image
        image_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(image_path, frame)

        print(f"Captured image {counter+1} for class {j}")
        counter += 1

        # Press 'q' again to stop data collection early
        if cv2.waitKey(1) == ord('q'):
            print("Early exit from data collection for class {}.".format(j))
            break

    # Short pause between class collections
    print(f"Completed data collection for class {j}.")
    time.sleep(2)  # Short delay to allow preparation for the next class

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
