import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import cv2
import imutils
import pyautogui


class GestureMouseController:
    """Controller that maps hand gestures to mouse actions."""

    def __init__(self):
        # tracking variables
        self.bg = None
        self.n = 0
        self.cX = 0
        self.cY = 0
        self.nX = 0
        self.nY = 0
        self.i = 0
        self.model = self._load_model()

    def _load_model(self):
        """Load the gesture recognition model."""
        model = models.Sequential([
            layers.Input(shape=(89, 100, 1)),
            layers.Conv2D(32, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 2, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(1000, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('TrainedNewModel/GestureRecogModel.h5')
        return model

    @staticmethod
    def resize_image(image_name):
        basewidth = 100
        img = Image.open(image_name)
        wpercent = basewidth / float(img.size[0])
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(image_name)

    def run_avg(self, image, aWeight):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, aWeight)

    def segment(self, image, threshold=25):
        if self.bg is None:
            return None
        diff = cv2.absdiff(self.bg.astype("uint8"), image)
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return None
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented

    def get_predicted_class(self):
        image = cv2.imread('Temp.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prediction = self.model.predict(gray_image.reshape(1, 89, 100, 1))
        return np.argmax(prediction), (np.amax(prediction) / np.sum(prediction))

    def show_statistics(self, predicted_class, confidence):
        pyautogui.FAILSAFE = False
        textImage = np.zeros((300, 512, 3), np.uint8)
        className = ""
        if predicted_class == 0:
            className = "Scroll Down - Swing"
            if self.i == 8:
                self.i = 0
            else:
                pyautogui.scroll(-10)
                self.n = 1
                self.i = 0
        elif predicted_class == 1:
            className = "Right Click - Palm"
            if self.n != 2:
                pyautogui.click(button='right')
                self.n = 2
            self.i = 0
        elif predicted_class == 2:
            className = "Mouse Movement - Fist"
            if self.i < 8:
                pyautogui.move((self.cX - self.nX) * 10, (self.cY - self.nY) * 10)
                if self.nX < 375:
                    pyautogui.move(-30, 0)
                if self.nX > 565:
                    pyautogui.move(30, 0)
                if self.nY < 215:
                    pyautogui.move(0, -30)
                if self.nY > 305:
                    pyautogui.move(0, 30)
                if abs(self.cX - self.nX) < 2 and abs(self.cY - self.nY) < 5:
                    self.i += 1
                else:
                    self.i = 0
            self.n = 3
        elif predicted_class == 3:
            className = "Left Click - Peace"
            if self.n != 4:
                pyautogui.click()
                self.n = 4
                self.i = 0
        elif predicted_class == 4:
            className = "Double Click - Three Finger"
            if self.n != 5:
                pyautogui.doubleClick()
                self.n = 5
            self.i = 0
        elif predicted_class == 5:
            className = "Scroll Up - Yo"
            pyautogui.scroll(10)
            self.n = 6
            self.i = 0
        self.nX = self.cX
        self.nY = self.cY
        print(className)
        cv2.putText(textImage, "Gesture : " + className, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(textImage, "Precision : " + str(confidence * 100) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Statistics", textImage)

    def run(self):
        aWeight = 0.5
        camera = cv2.VideoCapture(0)
        top, right, bottom, left = 110, 350, 325, 590
        num_frames = 0
        start_recording = False
        self.n = 0
        while True:
            grabbed, frame = camera.read()
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            if num_frames < 30:
                self.run_avg(gray, aWeight)
            else:
                hand = self.segment(gray)
                if hand is not None:
                    thresholded, segmented = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    try:
                        M = cv2.moments(segmented + (right, top))
                        self.cX = int(M["m10"] / M["m00"])
                        self.cY = int(M["m01"] / M["m00"])
                        if self.nX == 0 and self.nY == 0:
                            self.nX = self.cX
                            self.nY = self.cY
                        cv2.circle(clone, (self.cX, self.cY), 3, (255, 255, 255), -1)
                    except Exception:
                        print("Empty")
                    if start_recording:
                        cv2.imwrite('Temp.png', thresholded)
                        self.resize_image('Temp.png')
                        predicted_class, confidence = self.get_predicted_class()
                        self.show_statistics(predicted_class, confidence)
                    cv2.imshow("Thesholded", thresholded)
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(clone, (375, 215), (565, 305), (255, 0, 0), 1)
            num_frames += 1
            cv2.imshow("Video Feed", clone)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
            if keypress == ord("s"):
                start_recording = True
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureMouseController().run()
