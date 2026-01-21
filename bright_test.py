import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Function to calculate the brightness of an image
def calculate_brightness(image):
    return np.mean(image)

# Function to adjust the brightness of an image
def adjust_brightness(image, target_brightness):
    current_brightness = calculate_brightness(image)
    if current_brightness < target_brightness:
        brightness_ratio = target_brightness / current_brightness
        image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)
    return image

# Preprocess images
def preprocess_image(image_path, target_brightness):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Image at path {image_path} could not be loaded.")
        return None
    img = adjust_brightness(img, 130)  # Adjust brightness if necessary
    img = cv2.resize(img, (450, 450), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    return img

def get_newest_image(folder_path):
    # List all files in the directory
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Get the newest file based on the timestamp
    newest_file = max(files, key=os.path.getmtime)
    return newest_file

# Load the model
model = load_model(r'E:\project\model\sun_detect.h5')  # Update this path

# Specify the folder containing the images
folder_path = r'E:\project\model\demo'  # Update this path

# File paths
angles_file_path = r"E:\project\model\results.json"  # File to store old angles
random_angles_path = r"E:\project\model\coordinates.json"

# Load the old angles
if os.path.exists(angles_file_path):
    try:
        with open(angles_file_path, "r") as json_file:
            read_angles = json.load(json_file)
    except json.JSONDecodeError:
        read_angles = {}
else:
    read_angles = {}

# Load the random angles
if os.path.exists(random_angles_path):
    try:
        with open(random_angles_path, "r") as json_file:
            loaded_angles = json.load(json_file)
    except json.JSONDecodeError:
        loaded_angles = {"angles": [], "counter": 0}
else:
    loaded_angles = {"angles": [], "counter": 0}

# Define the angles to test
angle_list = [(0, 0), (0, 60), (0, 120), (0, 180), (60, 0), (60, 60), (60, 120), (60, 180),
              (120, 0), (120, 60), (120, 120), (120, 180),]

# Determine the FOV
FOV = 66

# Function to calculate angles
def calculate_angles(x, y):
    # Convert resized coordinates to original resolution
    x_orig = x * (320 / 450)
    y_orig = y * (240 / 450)

    # Calculate the angles
    phi_x = (x_orig / 320 - 0.5) * FOV
    phi_y = (0.5 - y_orig / 240) * FOV

    return phi_x, phi_y

# Get the newest image from the folder
newest_image_path = get_newest_image(folder_path)

# Preprocess the newest image with a target brightness of 100 (adjust as needed)
target_brightness = 100
new_image = preprocess_image(newest_image_path, target_brightness)
if new_image is not None:
    new_image = new_image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    print("New image shape:", new_image.shape)

    # Make predictions
    predictions = model.predict(new_image)

    sun_prediction = predictions[0][0]
    bbox_prediction = predictions[1][0]

    # Multiply the bounding box predictions by 224
    bbox_prediction = bbox_prediction * 450

    # Calculate the center of the bounding box
    center_x = (bbox_prediction[0] + bbox_prediction[2]) / 2
    center_y = (bbox_prediction[1] + bbox_prediction[3]) / 2

    # Display predictions
    print(f"Image: {newest_image_path}")
    sun_presence = "Sun" if sun_prediction > 0.5 else "No Sun"
    print(f"Sun Presence Prediction: {sun_presence}")
    print(f"Center of Bounding Box: (x: {center_x}, y: {center_y})")

    # Check sun presence and perform actions
    if sun_presence == "Sun":
        x = center_x
        y = center_y
        angles = calculate_angles(x, y)
        print(f"x= {angles[0]:.2f}, y= {angles[1]:.2f}")

        if "CenterOfBoundingBox" in read_angles:
            x1 = read_angles["CenterOfBoundingBox"].get("phi_x_old", 0)
            y1 = read_angles["CenterOfBoundingBox"].get("phi_y_old", 0)
        else:
            x1 = 0
            y1 = 0

        phi_x_new = angles[0] + x1
        phi_y_new = angles[1] + y1

        # Apply transformations on phi_x
        if phi_x_new < 0:
            phi_x_new = 180 + angles[0] + x1
            phi_y_new = 180 - angles[1] - y1
        elif phi_x_new > 180:
            phi_x_new = x1 + angles[0] - 180
            phi_y_new = 180 - angles[1] - y1

        # Apply transformations on phi_y
        if phi_y_new < 0:
            phi_y_new = 0
        elif phi_y_new > 180:
            phi_y_new = 180

        print("x = ", phi_x_new)
        print("y =",  phi_y_new)

        # Store the angles
        store_angles = {"SunPresencePrediction": "Sun",
                        "CenterOfBoundingBox": {
                            "x": phi_x_new,
                            "y": phi_y_new
                        }}

        with open(angles_file_path, "w") as json_file:
            json.dump(store_angles, json_file, indent=4)
            print(f"Angles saved to {angles_file_path}")

        # Clear tested angles if sun is found
        loaded_angles = {"angles": [], "counter": 0}
        with open(random_angles_path, "w") as json_file:
            json.dump(loaded_angles, json_file, indent=4)
            print(f"Tested angles cleared in {random_angles_path}")

    elif sun_presence == "No Sun":
        # Get the counter
        counter = loaded_angles.get("counter", 0)

        if counter < len(angle_list):
            next_angle = angle_list[counter]
            print(f"Testing angle: {next_angle}")

            # Save the angle to the result file to move the motor
            store_angles = {"SunPresencePrediction": "No Sun",
                            "x": next_angle[0],
                            "y": next_angle[1]
                            }

            with open(angles_file_path, "w") as json_file:
                json.dump(store_angles, json_file, indent=4)
                print(f"Next angle to test saved to {angles_file_path}")

            # Update the counter and save
            counter += 1
            loaded_angles["counter"] = counter
            with open(random_angles_path, "w") as json_file:
                json.dump(loaded_angles, json_file, indent=4)
                print(f"Counter updated and saved to {random_angles_path}")

        else:
            print("All angles have been tested.")
            #make the counter = 0
            loaded_angles["counter"] = 0
            with open(random_angles_path, "w") as json_file:
                json.dump(loaded_angles, json_file, indent=4)
                print(f"Counter reset to 0 in {random_angles_path}")
else:
    print("Failed to preprocess the image.")
