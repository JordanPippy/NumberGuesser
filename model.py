import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from PIL import Image
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def train_model():
    from keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_x, test_x = train_x / 255.0, test_x / 255.0
    

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28,28)),
    #     tf.keras.layers.Dense(512, activation="relu"),
    #     tf.keras.layers.Dense(256, activation="relu"),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(10, activation="softmax")
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=["accuracy"]
    )

    #model.fit(train_x, train_y, epochs=20, validation_data=(test_x, test_y))
    model.fit(train_x, train_y, epochs=5, validation_data=(test_x, test_y))

    model.evaluate(test_x,  test_y, verbose=2)

    tf.keras.models.save_model(
        model,
        filepath="./model"
    )

def test_model():
    from keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0

    testingX = test_x[0]
    testingY = test_y[0]

    model = tf.keras.models.load_model('./model')

    testingX = image.img_to_array(testingX)
    testingX = np.expand_dims(testingX, axis=0)

    #predictions = model.evaluate(test_x, test_y, verbose=2)
    predictions = model.predict(testingX)

    print(np.argmax(predictions))

def plot(X):
    import matplotlib.pyplot as plt
    image = X
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()


def make_predictions():
    model = tf.keras.models.load_model('./model')
    image = Image.open("image1.png")

    image = np.asarray(image)
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    
    prediction = model.predict(image)

    print(np.argmax(prediction))
    print(max(prediction[0]))