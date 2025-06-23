import tensorflow as tf
import cv2
import imutils
import logging

logging.basicConfig(level=logging.ERROR)

bg = None


def augment_image(img, zoom_range=0.1):
    """Apply random flip, brightness and zoom to the captured image."""
    tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    if len(tensor.shape) == 2:
        tensor = tf.expand_dims(tensor, -1)
    tensor = tf.image.random_flip_left_right(tensor)
    tensor = tf.image.random_brightness(tensor, 0.2)
    orig_shape = tf.shape(tensor)[:2]
    scale = tf.random.uniform([], 0.9, 1.0)
    new_size = tf.cast(tf.cast(orig_shape, tf.float32) * scale, tf.int32)
    tensor = tf.image.random_crop(
        tensor, tf.concat([new_size, [tf.shape(tensor)[-1]]], 0)
    )
    tensor = tf.image.resize(tensor, orig_shape)
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
    return tf.squeeze(tensor).numpy()


def save_augmented(img, folder, prefix, idx, augments=2):
    """Save image and several augmented versions."""
    cv2.imwrite(f"{folder}/{prefix}_{idx}.png", img)
    for i in range(augments):
        aug = augment_image(img)
        cv2.imwrite(f"{folder}/{prefix}_{idx}_aug{i}.png", aug)


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    # use global background from run_avg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(
        thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    aWeight = 0.5
    try:
        camera = cv2.VideoCapture(0)
    except Exception:
        logging.exception("Failed to access camera")
        return
    if not camera.isOpened():
        logging.error("Camera could not be opened")
        return
    top, right, bottom, left = 100, 400, 300, 600
    num_frames = 0
    image_num = 0
    start_recording = False

    while True:
        (grabbed, frame) = camera.read()
        if grabbed:
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            (height, width) = frame.shape[:2]
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)
            else:
                hand = segment(gray)
                if hand is not None:
                    (thresholded, segmented) = hand
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    if start_recording:
                        save_augmented(thresholded, "Dataset/YoImages", "yo", image_num)
                        image_num += 1
                    cv2.imshow("Thesholded", thresholded)
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            num_frames += 1
            cv2.imshow("Video Feed", clone)
            keypress = cv2.waitKey(1) & 0xFF
            print(image_num)

            if keypress == ord("q") or image_num > 999:
                break
            if keypress == ord("s"):
                start_recording = True

        else:
            print("Error, Check Camera")
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
