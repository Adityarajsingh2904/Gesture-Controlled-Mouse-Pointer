import json
from pathlib import Path
from PIL import Image


def load_config(path: str = "config.json"):
    with open(path, "r") as f:
        return json.load(f)


def resize_image(image_name: str, basewidth: int, height: int):
    img = Image.open(image_name)
    img = img.resize((basewidth, height), Image.ANTIALIAS)
    img.save(image_name)


def main():
    config = load_config()
    basewidth = int(config.get("basewidth", 100))
    height = int(config.get("height", 89))
    base_path = config.get("image_path", "")
    image_count = int(config.get("image_count", 0))
    extension = config.get("extension", ".png")

    for i in range(image_count):
        image_name = f"{base_path}{i}{extension}"
        if Path(image_name).exists():
            resize_image(image_name, basewidth, height)


if __name__ == "__main__":
    main()
