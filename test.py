# Import necessary library:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime


# Dummy function for trackbar
def nothing(x):
    pass

# Helper function to create a trackbar
def create_trackbar(trackbar_name, window_name, default_value, max_value):
    """Helper function to create a trackbar."""
    cv2.createTrackbar(trackbar_name, window_name, default_value, max_value, nothing)

# Initialize webcam
cap = cv2.VideoCapture(0)

# A flag to toggle between color and grayscale
grayscale_mode = False

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Create a window and trackbars for brightness and contrast
window_name = 'Webcam Feed'
cv2.namedWindow(window_name)
# Brightness: range 0-200, default 100 (0 is -100, 200 is +100)
create_trackbar('Brightness', window_name, 100, 200)
# Contrast: range 1-200, default 100 (1 is 0.01, 200 is 2.0)
create_trackbar('Contrast', window_name, 100, 200)


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a smaller size to fit the screen
    new_width = 640
    new_height = 480
    frame = cv2.resize(frame, (new_width, new_height))

    # Get current trackbar positions
    brightness = cv2.getTrackbarPos('Brightness', window_name) - 100
    contrast = cv2.getTrackbarPos('Contrast', window_name) / 100.0

    # Apply brightness and contrast
    # The formula is: new_image = alpha * original_image + beta
    # alpha is contrast, beta is brightness
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If '1' is pressed, toggle the grayscale mode
    if key == ord('1'):
        grayscale_mode = not grayscale_mode
    # If 'q' is pressed, break the loop
    elif key == ord('q'):
        break

    # If grayscale mode is on, convert the adjusted frame
    if grayscale_mode:
        display_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
    else:
        display_frame = adjusted_frame

    # Add text instructions to the frame
    cv2.putText(display_frame, 'Press 1: Toggle Grayscale | Press q: Quit', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow(window_name, display_frame)

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

