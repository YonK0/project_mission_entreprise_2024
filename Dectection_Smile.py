import cv2
import numpy as np
import random
import os
from running_model import *
from mqtt_test import * 

# Load pre-trained face and smile detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to enhance the image
def enhance_image(image):
    # Convert to grayscale and apply histogram equalization for better contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

# Function to adjust brightness
def adjust_brightness(image, beta):
    # Convert to YUV color space and adjust the brightness of the Y channel
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.add(yuv[:, :, 0], beta)
    bright_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return bright_image

# Function to detect faces and eyes
def detect_faces_and_eyes(frame, face_cascade, eye_cascade, directory_path, counter):
    # Enhance the image
    enhanced_frame = enhance_image(frame)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(enhanced_frame, scaleFactor=1.2, minNeighbors=10, minSize=(22, 22))
    
    if len(faces) > 0:
        # Process only the first detected face
        (x, y, w, h) = faces[0]
        margin = 0.2
        x_expanded = max(0, int(x - margin * w))
        y_expanded = max(0, int(y - margin * h))
        w_expanded = min(frame.shape[1] - x_expanded, int(w * (1 + 2 * margin)))
        h_expanded = min(frame.shape[0] - y_expanded, int(h * (1 + 2 * margin)))

        # Draw rectangle around the face
        cv2.rectangle(frame, (x_expanded, y_expanded), (x_expanded + w_expanded, y_expanded + h_expanded), (255, 0, 0), 2)
        
        # Region of interest for face
        roi_color = frame[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]
        
        # Adjust brightness
        bright_face_image = adjust_brightness(roi_color, beta=50)

        eyes = eye_cascade.detectMultiScale(roi_color)
        if len(eyes) > 0 and counter[0] < config['number_of_images']:
            ext = random.randint(1, 9999999)
            # Save the detected face as an image
            cv2.imwrite(f'{directory_path}face_{ext}.png', bright_face_image)
            counter[0] += 1
            print(f'Screenshot taken and saved as face_{ext}.png')
            if counter[0] >= config['number_of_images']:
                return True  # Indicate that the limit has been reached
    return False

#############Main#################

# Create output directory
dir_ = random.randint(1, 9999999999999)
home = config['home_dir']
directory_path = f'{home}/{dir_}/output_images/'
os.makedirs(directory_path, exist_ok=True)

# Start MQTT service with sudo 
start_mqtt()

# Capture video from webcam
webcam = cv2.VideoCapture(0)
counter = [0]

while True:
    # Read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()
    
    # If there's an error, abort
    if not successful_frame_read:
        break
    
    # Detect faces and eyes
    if detect_faces_and_eyes(frame, face_cascade, eye_cascade, directory_path, counter):
        if (Detection(directory_path)) : 
            message = "happy" 
            mqtt_connect(topic ,message)
            break 
        
    # Display the image
    cv2.imshow('Face Detector', frame)
    
    # Stop if Q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()

print('Code Completed')
