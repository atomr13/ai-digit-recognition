# Handwritten Digit Recognition with MNIST

This project implements a Convolutional Neural Network (CNN) model for recognizing handwritten digits (0-9) using the MNIST dataset. The model achieves high accuracy (99.07%) on the MNIST test set and can predict handwritten digits from images that are preprocessed to match the MNIST dataset format.

## Project Overview

This project demonstrates:
- Training a CNN model using TensorFlow and Keras.
- Preprocessing images to match the MNIST dataset format.
- Evaluating the model's accuracy on the MNIST test dataset.
- Testing the model on custom images after appropriate preprocessing.

## Model Summary

The CNN Model consist of:
- Two convolutional layers with ReLU activation and max pooling.
- A fully connected dense layer with 64 units.
- A softmax output layer for classifying 10 digits (0-9).

The project is designed to **work best with images similar to the MNIST dataset**, which consists of 28x28 grayscale images of single digits, white on a black background.

## Requirements

To run the project, you will need the following Python packages:
- `tensorflow`
- `numpy`
- `Pillow` (for image preprocessing)
- `matplotlib` (for image visualization)

Install the dependencies using:

```bash
pip install -r requirements.txt
```
# How to Run the Project

## 1. Train the Model

To train the model, simply run the `model.py` script: 

```bash
python model.py
```
The script will: 
- Load and preprocess the MNIST dataset.
- Define and train a CNN model on the MNIST data.
- Save the trained model as `mnist_cnn_model.h5`

## 2. Run the Web Application

Once the model is trained, you can use the `webapp.py` to launch a web interface where users can upload images and get predictions.

To run the web application:
```bash
python webapp.py
```

This will start a web server where you can upload handwritten digit images and get predictions from the trained model.

# Important Notes

- **Best with MNIST Images**: This model works best with images resembling those in the MNIST dataset (28x28 pixels, grayscale, white digit on a black background). Custom images must be preprocessed accordingly.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

