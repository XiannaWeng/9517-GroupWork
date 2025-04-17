import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name='resnet18', num_classes=15, pretrained=True, freeze_layers=True):
    """Loads a pre-trained model and modifies the final layer."""
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    # Add more models like SENet etc. (might need external libraries like pretrainedmodels)
    else:
        raise ValueError("Unsupported model name")

    if model is None:
         raise ValueError("Model could not be loaded.")

    # Freeze layers if specified
    if pretrained and freeze_layers:
        print(f"Freezing layers for {model_name}...")
        for param in model.parameters():
            param.requires_grad = False

    # Modify the final classifier layer
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"Replaced ResNet fc layer. New out_features: {num_classes}")
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(f"Replaced EfficientNet classifier layer. New out_features: {num_classes}")
    # Add modifications for other architectures if needed

    return model

if __name__ == '__main__':
    # Example Usage
    model_res18 = get_model('resnet18', num_classes=15, pretrained=True, freeze_layers=True)
    model_effnet = get_model('efficientnet_b0', num_classes=15, pretrained=True, freeze_layers=False) # Example: Fine-tune all

    # Print model structure (optional)
    # print(model_res18)

    # Check trainable parameters
    print("\nResNet18 Trainable Params:")
    for name, param in model_res18.named_parameters():
        if param.requires_grad:
            print(name)

    print("\nEfficientNet B0 Trainable Params (Fine-tuning all):")
    count = 0
    for name, param in model_effnet.named_parameters():
        if param.requires_grad:
            # print(name) # This will print a lot
            count +=1
    print(f"Total trainable parameters in EfficientNet: {count}")