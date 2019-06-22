import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet18
import numpy as np
import torch.nn.init as init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias:
            init.constant_(m.bias.data, 0.0)


class Normal_Linear(nn.Linear):
    def __init__(self,in_features,out_features):
        super(Normal_Linear,self).__init__(in_features,out_features,False)

    def forward(self, input):
        return F.linear(input, F.normalize(self.weight), self.bias)

class ResNet_50(nn.Module):
    def __init__(self,args):
        super(ResNet_50,self).__init__()


        resnet = resnet50(pretrained=True)

        self.backbone0=nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
        )

        self.pool=nn.AdaptiveAvgPool2d((1,1))

        bottleneck = nn.BatchNorm1d(1024)
        bottleneck.bias.requires_grad_(False)

        self.feater=nn.Sequential(
            nn.Linear(2048,1024),
            bottleneck
        )
        self.feater.apply(weights_init_kaiming)

        # self.classifier=nn.Linear(1024,args.num_classes,bias=False)
        self.classifier=Normal_Linear(1024,args.num_classes)
        self.classifier.weight.data.normal_(std=0.001)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


        self.params_groups=[{'params':self.parameters()}]

        self.mode='train'

    def forward(self, x):
        """

        :param x: input
        :return: dict {name: output}
        """
        f0=self.pool(self.backbone0(x)).view(x.size(0),-1)
        f0=self.feater(f0)
        if self.mode == 'train':
            sc=self.classifier(f0)
            return {'CE_Loss':[sc]}

        return f0





