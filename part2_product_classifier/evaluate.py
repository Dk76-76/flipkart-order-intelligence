import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_score, recall_score


DATA_DIR = "data"
CLASSIFIER_PATH = "part2_product_classifier/data/features/classifier_head.pt"
FINAL_MODEL_PATH = "models/product_classifier.pt"
OUTPUT_DIR = "part2_product_classifier/outputs"

BATCH_SIZE = 64
NUM_CLASSES = 10

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


backbone = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

for parameter in backbone.parameters():
    parameter.requires_grad = False


feature_size = backbone.fc.in_features
backbone.fc = nn.Identity()

backbone = backbone.to(device)
backbone.eval()


classifier = nn.Linear(
    feature_size,
    NUM_CLASSES
)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=device
    )
)

classifier = classifier.to(device)
classifier.eval()


all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        features = backbone(images)
        outputs = classifier(features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


accuracy = sum(
    prediction == label
    for prediction, label in zip(
        all_predictions,
        all_labels
    )
) / len(all_labels)


matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


precision = precision_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average=None,
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    average=None,
    zero_division=0
)


os.makedirs(
    "models",
    exist_ok=True
)

model_state = {
    "backbone": backbone.state_dict(),
    "classifier": classifier.state_dict(),
    "class_names": CLASS_NAMES
}

torch.save(
    model_state,
    FINAL_MODEL_PATH
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_results.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Test images: {len(test_dataset)}\n"
    )

    file.write(
        f"Test accuracy: {accuracy:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(matrix)
    )

    file.write("\n\n")

    file.write(
        "Per-class precision and recall:\n"
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        file.write(
            f"{class_name}: "
            f"Precision={precision[i]:.4f}, "
            f"Recall={recall[i]:.4f}\n"
        )


print(
    f"Test images: {len(test_dataset)}"
)

print(
    f"Test accuracy: {accuracy:.4f}"
)

print(
    "Evaluation results saved."
)

print(
    f"Final model saved: {FINAL_MODEL_PATH}"
)