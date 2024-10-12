import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Preprocess the data (normalize and reshape)
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


# Define CNN Model
model = tf.keras.models.Sequential([
    # First Convolutional Layer
    layers.Conv2D(32,(3, 3), activation = 'relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    # Second Convolutional Layer
    layers.Conv2D(64,(3,3), activation = 'relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    # Flatten the Output of Previous Layers
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(64, activation = 'relu'),

    # Output Layer (10 Classes for 0-9)
    layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model (Trained once to avoid overfitting)
model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test))

# Save Model (Saved after trained)
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")


# Evaluate the Trained Model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Trained model test accuracy: {test_accuracy}")

