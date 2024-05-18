import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

# Define the model architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Load the model weights
model.load_weights('/home/aero/Downloads/model_weights.h5')

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define a function to make a prediction
def predict_emotion(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return np.argmax(prediction), prediction

# Define a function to map the predicted class index to the emotion label
def get_emotion_labels():
    return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



# Path to the directory containing the test images
test_images_dir = '/home/aero/output_images/'

# List of test images
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('png', 'jpg', 'jpeg'))]


emotion_labels = get_emotion_labels()

for i, img_path in enumerate(test_images):
    class_index, prediction = predict_emotion(img_path)
    emotion_label = emotion_labels[class_index]

    # Load the original image for annotation
    original_img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    font = ImageFont.load_default()

    # Annotate the image with the predicted emotion and probabilities
    text = f'Predicted Emotion: {emotion_label}\n' + "\n".join([f"{label}: {percentage:.2f}%" for label, percentage in zip(emotion_labels, prediction[0] * 100)])
    draw.text((10, 10), text, fill="red", font=font)

    # Save the annotated image
    annotated_img_path = os.path.join(test_images_dir, f'annotated_{i}.png')
    original_img.save(annotated_img_path)

    