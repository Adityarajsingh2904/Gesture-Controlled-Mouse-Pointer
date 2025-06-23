# Project Overview

Gesture Controlled Mouse Pointer uses a webcam to detect hand gestures and map them to mouse actions. MediaPipe provides the hand landmarks while a lightweight CNN classifies gestures.

# Requirements & Installation

- Python 3.8+
- Install dependencies using:

```bash
pip install -r requirements.txt
```

# How to Train Your Gestures

1. Capture gesture images using `ImageGenerator.py`.
2. Optionally resize the captured images with `Resize.py` which reads `config.json` for dimensions.
3. Train the CNN model with:

```bash
python run.py --train
```

The resulting model is saved to `TrainedNewModel/GestureRecogModel.h5`.

# Running the Gesture-Controlled Mouse

Start the controller:

```bash
python run.py --run
```

Press `s` in the video window to start classifying gestures and `q` to quit.

# Folder Structure

```
Dataset/           # Training and test images
TrainedNewModel/   # Saved neural network weights
ImageGenerator.py  # Tool to capture training images
Resize.py          # Utility to resize image dataset
Trainer.py         # Training script used by run.py
run.py             # Entry point for training or running the mouse controller
```

# Known Issues & TODOs

- No automated tests are provided.
- Paths in `config.json` may need updating when running on a different system.
- The accuracy of gesture recognition could be improved with more training data.
