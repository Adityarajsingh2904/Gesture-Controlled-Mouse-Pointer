import argparse

from Main import GestureMouseController
import Resize
import Trainer


def main():
    parser = argparse.ArgumentParser(description="Gesture Controlled Mouse Pointer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train the gesture model")
    group.add_argument("--run", action="store_true", help="Run the mouse controller")
    group.add_argument("--resize", action="store_true", help="Resize dataset images")

    args = parser.parse_args()

    if args.train:
        Trainer.train()
    elif args.run:
        GestureMouseController().run()
    elif args.resize:
        Resize.main()


if __name__ == "__main__":
    main()
