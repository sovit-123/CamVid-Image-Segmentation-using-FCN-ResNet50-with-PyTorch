"""
We will use the FCN ResNet50 from the PyTorch model. We will not
use any pretrained weights. Training from scratch.
"""

import torchvision.models as models

def model(pretrained, requires_grad):
    model = models.segmentation.fcn_resnet50(
        pretrained=pretrained, progress=True,
        num_classes=32
    )
    
    if requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    elif requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    
    return model


model = model(pretrained=False, requires_grad=True)
print(model)