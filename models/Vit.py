import torch.nn as nn
import torchvision.models as models
import timm
from transformers import ViTForImageClassification


class VitModel(nn.Module):
    def __init__(self, num_classes, device):
        super(VitModel, self).__init__()
        self.device = device

        self.backbone = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')



        # self.backbone =  timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        # self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
        
    def forward(self, x):
        x = self.backbone(x)
        # print(x['logits'])
        
        x = self.classifier(x['logits'])
        return x