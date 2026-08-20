import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


MODEL_PATH = "models/product_classifier.pt"

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


def classify_product_image(image_path: str) -> dict:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_data = torch.load(
        MODEL_PATH,
        map_location=device
    )

    backbone = models.resnet18(
        weights=None
    )

    backbone.fc = nn.Identity()

    backbone.load_state_dict(
        model_data["backbone"]
    )

    classifier = nn.Linear(
        512,
        10
    )

    classifier.load_state_dict(
        model_data["classifier"]
    )

    backbone = backbone.to(device)
    classifier = classifier.to(device)

    backbone.eval()
    classifier.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(image)
        outputs = classifier(features)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[0, predicted_index].item()

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    image_path = "data/sample_images/01_pullover.png"

    result = classify_product_image(image_path)

    print(f"Image: {image_path}")
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']}")