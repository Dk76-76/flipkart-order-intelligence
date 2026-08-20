from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST


DATA_DIR = Path("part2_product_classifier/data")


def load_fashion_mnist():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True
    )

    test_dataset = FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True
    )

    labels = np.asarray(train_dataset.targets)

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=6000,
        stratify=labels,
        random_state=42
    )

    print("Train images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Test images:", len(test_dataset))

    return train_dataset, test_dataset, train_indices, val_indices


if __name__ == "__main__":
    load_fashion_mnist()