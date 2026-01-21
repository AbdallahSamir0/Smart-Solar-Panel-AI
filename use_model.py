import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import matplotlib.pyplot as plt

# Preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Image at path {image_path} could not be loaded.")
        return None
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

def detect_brightest_spot(image_path):
    # Read the image in grayscale and resize to (224, 224)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image {image_path}")
        return None, None

    image = cv2.resize(image, (450, 450), interpolation=cv2.INTER_LINEAR)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour with the largest area (brightest spot)
        brightest_contour = max(contours, key=cv2.contourArea)
        # Get the centroid of the brightest contour
        M = cv2.moments(brightest_contour)
        brightest_spot_x = int(M["m10"] / M["m00"])
        brightest_spot_y = int(M["m01"] / M["m00"])
        return brightest_spot_x, brightest_spot_y
    else:
        return None, None

def draw_circle_and_plot(image_path, coordinates, label, bbox=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    image = cv2.resize(image, (450, 450), interpolation=cv2.INTER_LINEAR)
    x, y = map(int, coordinates)
    cv2.circle(image, (x, y), 15, (0, 0, 255), -1)  # Draw a larger red circle

    if bbox is not None:
        # Draw the bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(label)
    plt.show()

# Load the model
model = load_model(r'E:\project\model\sun_detect.h5')  # Use pretrained DL model

# Specify the folder containing the images
folder_path = r'E:\project\model\demo'  # folder of images

# Get the newest image from the folder
newest_image_path = get_newest_image(folder_path)

# Preprocess the newest image
new_image = preprocess_image(newest_image_path)
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
    print(f"Sun Prediction: {sun_prediction}")
    print(f"Sun Presence Prediction: {sun_presence}")
    print(f"Center of Bounding Box: (x: {center_x}, y: {center_y})")

    data = {
        "SunPresencePrediction": sun_presence,
        "CenterOfBoundingBox": {"x": center_x, "y": center_y}
    }

    coordinates = None
    label = ""
    bbox = None

    if sun_presence == "No Sun":
        # Detect the brightest spot in the image
        brightest_spot_x, brightest_spot_y = detect_brightest_spot(newest_image_path)
        if brightest_spot_x is not None and brightest_spot_y is not None:
            print(f"Brightest Spot: (x: {brightest_spot_x}, y: {brightest_spot_y})")
            data["BrightestSpot"] = {"x": brightest_spot_x, "y": brightest_spot_y}
            coordinates = (brightest_spot_x, brightest_spot_y)
            label = "Brightest Spot"
        else:
            print("No bright spot detected in the image.")
    else:
        coordinates = (center_x, center_y)
        bbox = bbox_prediction
        label = "Sun Center"

    # Draw the circle and plot the image
    if coordinates:
        draw_circle_and_plot(newest_image_path, coordinates, label, bbox)

    # Define the path to the JSON file
    json_file_path = r"E:\project\model\coor.json"  # Update this path if necessary

    # Write the data to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        print(f"Predictions saved to {json_file_path}")
else:
    print("Failed to preprocess the image.")
