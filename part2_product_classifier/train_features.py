from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

from model import build_model
from preprocessing import get_image_transform


DATA_DIR = Path("part2_product_classifier/data")
FEATURE_DIR = DATA_DIR / "features"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets():
    transform = get_image_transform()

    train_dataset = FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=False
    )

    labels = np.asarray(train_dataset.targets)

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=6000,
        stratify=labels,
        random_state=42
    )

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    return train_data, val_data


def extract_features(backbone, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_features = []
    all_labels = []

    backbone.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            features = backbone(images)
            features = features.view(features.size(0), -1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    return features, labels


def train_classifier(
    train_features,
    train_labels,
    val_features,
    val_labels
):
    classifier = nn.Linear(512, 10).to(DEVICE)

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        classifier.train()

        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = classifier(features)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        classifier.eval()

        with torch.no_grad():
            val_outputs = classifier(val_features.to(DEVICE))
            predictions = torch.argmax(val_outputs, dim=1)

            accuracy = (
                predictions == val_labels.to(DEVICE)
            ).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f} "
            f"- Validation accuracy: {accuracy:.4f}"
        )

    return classifier, best_accuracy


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    train_data, val_data = get_datasets()

    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))

    model = build_model().to(DEVICE)

    backbone = nn.Sequential(
        *list(model.children())[:-1]
    ).to(DEVICE)

    train_feature_file = FEATURE_DIR / "train_features.pt"
    val_feature_file = FEATURE_DIR / "val_features.pt"

    if train_feature_file.exists() and val_feature_file.exists():
        print("Loading cached features...")

        train_cache = torch.load(
            train_feature_file,
            map_location="cpu"
        )

        val_cache = torch.load(
            val_feature_file,
            map_location="cpu"
        )

        train_features = train_cache["features"]
        train_labels = train_cache["labels"]

        val_features = val_cache["features"]
        val_labels = val_cache["labels"]

    else:
        print("Extracting train features...")
        train_features, train_labels = extract_features(
            backbone,
            train_data
        )

        print("Extracting validation features...")
        val_features, val_labels = extract_features(
            backbone,
            val_data
        )

        torch.save(
            {
                "features": train_features,
                "labels": train_labels
            },
            train_feature_file
        )

        torch.save(
            {
                "features": val_features,
                "labels": val_labels
            },
            val_feature_file
        )

        print("Features cached.")

    print("Train feature shape:", train_features.shape)
    print("Validation feature shape:", val_features.shape)

    classifier, best_accuracy = train_classifier(
        train_features,
        train_labels,
        val_features,
        val_labels
    )

    torch.save(
        classifier.state_dict(),
        FEATURE_DIR / "classifier_head.pt"
    )

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("Classifier head saved.")


if __name__ == "__main__":
    main()