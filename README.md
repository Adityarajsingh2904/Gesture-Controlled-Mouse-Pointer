# Gesture Controlled Mouse Pointer

This project allows controlling the mouse pointer using hand gestures. The repository includes scripts for
collecting training images, training a recognition model and running the gesture based controller.

## Usage

All entry points are now exposed through `run.py` using a simple command line interface:

```bash
python run.py --train   # train the gesture recognition model
python run.py --run     # start the gesture controlled mouse
python run.py --resize  # resize images according to config.json
```

The training and resize steps expect the dataset paths defined in the repository. Adjust `config.json`
for image resizing settings if required.
