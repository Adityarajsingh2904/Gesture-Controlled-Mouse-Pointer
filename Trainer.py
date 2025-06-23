from tensorflow.keras import layers, models
import numpy as np
import cv2
from sklearn.utils import shuffle


def train():
    # Load Images from Swing
    loadedImages = []
    for i in range(0, 1000):
        image = cv2.imread("Dataset/SwingImages/swing_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Load Images From Palm
    for i in range(0, 1000):
        image = cv2.imread("Dataset/PalmImages/palm_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Load Images From Fist
    for i in range(0, 1000):
        image = cv2.imread("Dataset/FistImages/fist_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Load Images From Peace
    for i in range(0, 1000):
        image = cv2.imread("Dataset/PeaceImages/peace_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Load Images From ThumbsUp
    for i in range(0, 1000):
        image = cv2.imread("Dataset/TriImages/tri_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Load Images From ThumbsDown
    for i in range(0, 1000):
        image = cv2.imread("Dataset/YoImages/yo_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loadedImages.append(gray_image.reshape(89, 100, 1))

    # Create OutputVector
    outputVectors = []
    for i in range(0, 1000):
        outputVectors.append([1, 0, 0, 0, 0, 0])

    for i in range(0, 1000):
        outputVectors.append([0, 1, 0, 0, 0, 0])

    for i in range(0, 1000):
        outputVectors.append([0, 0, 1, 0, 0, 0])

    for i in range(0, 1000):
        outputVectors.append([0, 0, 0, 1, 0, 0])

    for i in range(0, 1000):
        outputVectors.append([0, 0, 0, 0, 1, 0])

    for i in range(0, 1000):
        outputVectors.append([0, 0, 0, 0, 0, 1])

    testImages = []

    for i in range(0, 100):
        image = cv2.imread("Dataset/SwingTest/swing_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    for i in range(0, 100):
        image = cv2.imread("Dataset/PalmTest/palm_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    for i in range(0, 100):
        image = cv2.imread("Dataset/FistTest/fist_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    for i in range(0, 100):
        image = cv2.imread("Dataset/PeaceTest/peace_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    for i in range(0, 100):
        image = cv2.imread("Dataset/TriTest/tri_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    for i in range(0, 100):
        image = cv2.imread("Dataset/YoTest/yo_" + str(i) + ".png")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        testImages.append(gray_image.reshape(89, 100, 1))

    testLabels = []

    for i in range(0, 100):
        testLabels.append([1, 0, 0, 0, 0, 0])

    for i in range(0, 100):
        testLabels.append([0, 1, 0, 0, 0, 0])

    for i in range(0, 100):
        testLabels.append([0, 0, 1, 0, 0, 0])

    for i in range(0, 100):
        testLabels.append([0, 0, 0, 1, 0, 0])

    for i in range(0, 100):
        testLabels.append([0, 0, 0, 0, 1, 0])

    for i in range(0, 100):
        testLabels.append([0, 0, 0, 0, 0, 1])

    # Build model using Keras
    model = models.Sequential(
        [
            layers.Input(shape=(89, 100, 1)),
            layers.Conv2D(32, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 2, activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(1000, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(6, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)
    model.fit(
        np.array(loadedImages),
        np.array(outputVectors),
        epochs=50,
        validation_data=(np.array(testImages), np.array(testLabels)),
    )
    model.save("TrainedNewModel/GestureRecogModel.h5")


if __name__ == "__main__":
    train()
