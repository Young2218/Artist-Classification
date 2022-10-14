import torch.nn as nn
import torchvision.models as models

from transformers import ViTForImageClassification


class VitModel(nn.Module):
    def __init__(self, num_classes, device):
        super(VitModel, self).__init__()
        self.device = device

        self.backbone = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
        
    def forward(self, x):
        x = self.backbone(x)
        # print(x[0].shape)
        x = self.classifier(x[0])
        return x