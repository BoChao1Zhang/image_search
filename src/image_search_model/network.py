import torch
import torch.nn as nn
import torchvision.models as models


class NetWork(nn.Module):
    def __init__(self,):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad = False
        vgg16.classifier[3] = nn.Linear(in_features=4096, out_features=128)
        vgg16.classifier[6] = nn.Linear(in_features=128, out_features=2)

        self.vgg16 = vgg16

    def forward(self, x):
        return self.vgg16(x)
