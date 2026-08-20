import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART2_PATH = PROJECT_ROOT / "part2_product_classifier"

sys.path.insert(0, str(PART2_PATH))

from predict import classify_product_image as classify_image


def classify_product_image(image_path: str) -> dict:
    result = classify_image(image_path)

    return {
        "image_path": image_path,
        "class": result["class"],
        "confidence": result["confidence"],
    }


if __name__ == "__main__":
    image_path = "data/sample_images/01_pullover.png"

    result = classify_product_image(image_path)

    print("Image:", result["image_path"])
    print("Predicted class:", result["class"])
    print("Confidence:", result["confidence"])