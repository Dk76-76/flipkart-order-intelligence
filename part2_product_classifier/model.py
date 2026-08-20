import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_CLASSES = 10


def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features

    model.fc = nn.Linear(input_features, NUM_CLASSES)

    return model


if __name__ == "__main__":
    model = build_model()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print("Backbone: ResNet-18")
    print("Pretrained weights: ImageNet")
    print("Output classes:", NUM_CLASSES)
    print("Trainable parameters:", trainable_parameters)
    print("Frozen parameters:", frozen_parameters)