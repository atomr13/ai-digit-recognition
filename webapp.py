import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# Load Model
model = load_model('mnist_cnn_model.h5')

# Define Prediction Function
def predict_digit(image):
    """
    Preprocesses the input image, runs the digit prediction using the trained model, 
    and returns the predicted digit.
    
    Args:
        image (numpy array): Image to be processed and predicted. The image is expected
        to be a grayscale image in NumPy array format.
    
    Returns:
        int: Predicted digit (0-9), or a message if no image is provided.
    """

    if image is None:
        return "No image provided"

    
    image = Image.fromarray(image) # Convert the NumPy array to a PIL Image for further processing

    image = ImageOps.grayscale(image) # Convert the image to grayscale (if it's not already)
    image = ImageOps.invert(image)  # Invert the colors to match MNIST format (white digits on black background)
    image = image.resize((28,28)) # Resize the image to 28x28 pixels as required by the model
    
    # Enhance the contrast of the image to make the digit more prominent
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    image = np.array(image) # Convert the image back to a NumPy array for further processing

    # Apply thresholding to enhance digit clarity (optional)
    threshold_value = 128 # Pixel values greater than 128 become white (255), others become black (0)
    image = np.where(image > threshold_value, 255, 0).astype(np.uint8)

    # Reshape the image to match the input shape expected by the CNN model (28x28 pixels, 1 channel)
    # Normalize the pixel values to the range [0, 1] as required by the model
    image = image.reshape(1,28,28,1) / 255.0

    prediction = model.predict(image) # Use the model to predict the digit in the image
    predicted_digit = np.argmax(prediction) # Get the digit with the highest probability (output as an integer)

    print("Prediction array:", prediction) # Print the prediction array for debugging purposes

    return int(predicted_digit) # Return the predicted digit

# Gradio Interface
interface = gr.Interface(
    fn = predict_digit,
    inputs = gr.Image(image_mode = 'L'),
    outputs= "label",
    live = True
)

interface.launch()

