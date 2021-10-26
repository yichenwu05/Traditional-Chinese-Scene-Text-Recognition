import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

# ['resnet50', 'resnet101', 'densenet201', 'efficientnetb1']
def model_select(model_name, num_class, device, load_weigth=''):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, num_class)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048, num_class)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(2048, num_class)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = nn.Linear(1920, num_class, bias=True)
    elif model_name == 'efficientnetb1':
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_class)
    else:
        return None
    if load_weigth:
        model.load_state_dict(torch.load(f'./model/weight/{load_weigth}.ckpt', map_location=device), strict=False)
    model.to(device)
    return model