import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import cv2
import imutils
import pyautogui
import logging
import mediapipe as mp

logging.basicConfig(level=logging.ERROR)


class GestureMouseController:
    """Controller that maps hand gestures to mouse actions."""

    def __init__(self):
        self.bg = None
        self.n = 0
        self.cX = 0
        self.cY = 0
        self.nX = 0
        self.nY = 0
        self.i = 0
        self.model = self._load_model()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,
                                          min_detection_confidence=0.7,
                                          min_tracking_confidence=0.7)
        self.drawing = mp.solutions.drawing_utils

    def _load_model(self):
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
        try:
            img = Image.open(image_name)
        except Exception:
            logging.exception("Failed to open image %s", image_name)
            return
        wpercent = basewidth / float(img.size[0])
        hsize = int(float(img.size[1]) * wpercent)
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(image_name)

    def extract_hand_roi(self, frame):
        """Return cropped hand image using mediapipe detection."""
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x1, x2 = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
        y1, y2 = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        self.cX = int((x1 + x2) / 2)
        self.cY = int((y1 + y2) / 2)
        self.drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame[y1:y2, x1:x2]

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

    def get_predicted_class(self, image):
        if image is None:
            return 0, 0
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (100, 89))
        except Exception:
            logging.exception('Failed to preprocess frame for prediction')
            return 0, 0
        try:
            prediction = self.model.predict(gray_image.reshape(1, 89, 100, 1), verbose=0)
        except Exception:
            logging.exception("Model prediction failed")
            return 0, 0
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
        try:
            camera = cv2.VideoCapture(0)
        except Exception:
            logging.exception("Failed to access camera")
            return
        if not camera.isOpened():
            logging.error("Camera could not be opened")
            return

        start_recording = False
        self.n = 0

        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                logging.error("Failed to grab frame from camera")
                break
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            roi = self.extract_hand_roi(clone)
            if start_recording and roi is not None:
                predicted_class, confidence = self.get_predicted_class(roi)
                self.show_statistics(predicted_class, confidence)

            cv2.imshow("Video Feed", clone)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
            if keypress == ord("s"):
                start_recording = True

        camera.release()
        self.hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureMouseController().run()
