import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from matplotlib.pyplot import imshow, show, axis
import rembg
from PIL import Image
from queue import Queue

images = Queue ## Replace with actual queue
responses = {}

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1) / 255.0

# Save the trained model
model_path = 'fashion_mnist_cnn_model.keras'
model = keras.models.load_model(model_path)


while True:
    image_path = images.get()
    input_image = Image.open(image_path)
    # Convert the image to RGBA mode (if not already in RGBA mode)
    input_image = input_image.convert('RGBA')
    # Convert the RGBA image to a NumPy array
    input_array = np.array(input_image)
    # Remove the background using rembg
    output_array = rembg.remove(input_array)
    # Convert the output array back to an RGBA image
    output_image = Image.fromarray(output_array)
    # Save the output image with the background removed

    # Load and preprocess your own image(s) for classification
    image = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)
    # Resize the image using INTER_AREA interpolation
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)

    # Make predictions on the custom image
    prediction = model.predict(np.array([image]))
    predicted_class = np.argmax(prediction)

    # Display the predicted class
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    responses[image_path.removesuffix(".jpg")] = class_names[predicted_class]
