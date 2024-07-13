import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import os

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array.astype(np.float32)

# Define a function to make a prediction
def predict_emotion(img_path):
    img_array = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction), prediction

# Define a function to map the predicted class index to the emotion label
def get_emotion_labels():
    return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def Detection(dir_path):

    # List of test images
    test_images = [os.path.join(dir_path, img) for img in os.listdir(dir_path) if img.endswith(('png', 'jpg', 'jpeg'))]

    emotion_labels = get_emotion_labels()

    for i, img_path in enumerate(test_images):
        class_index, prediction = predict_emotion(img_path)
        emotion_label = emotion_labels[class_index]

        # Check if the predicted emotion is "Happy" with a probability greater than 60%
        if emotion_label == 'Happy' and prediction[0][class_index] > 0.62:
            # Load the original image for annotation
            original_img = Image.open(img_path)
            annotated_img_path = os.path.join(dir_path, f'happy_{i}.png')
            original_img.save(annotated_img_path)
            
            # Annotate the image with the predicted emotion and probabilities
            text = f'Predicted Emotion: {emotion_label}\n' + "\n".join([f"{label}: {percentage:.2f}%" for label, percentage in zip(emotion_labels, prediction[0] * 100)])
            print("the client is happy")
            return True

    print("the client is not happy")
    return False
