import os

import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST


DATA_DIR = "data"
OUTPUT_DIR = "data/sample_images"

CLASS_NAMES = [
    "tshirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot"
]

SAMPLE_CLASSES = [0, 1, 2, 3, 5]


test_dataset = FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


saved_classes = set()

for image_index in range(len(test_dataset)):

    image, label = test_dataset[image_index]

    if label not in SAMPLE_CLASSES:
        continue

    if label in saved_classes:
        continue

    image_array = np.array(image)

    filename = f"{len(saved_classes) + 1:02d}_{CLASS_NAMES[label]}.png"
    file_path = os.path.join(OUTPUT_DIR, filename)

    Image.fromarray(image_array).save(file_path)

    print(f"Saved: {file_path}")

    saved_classes.add(label)

    if len(saved_classes) == len(SAMPLE_CLASSES):
        break


print(f"Sample images exported: {len(saved_classes)}")