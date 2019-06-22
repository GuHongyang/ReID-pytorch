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
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias:
            init.constant_(m.bias.data, 0.0)


class Strong_ReID(nn.Module):
    def __init__(self,args):
        super(Strong_ReID,self).__init__()


        resnet = resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
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

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(2048, args.num_classes, bias=False)


        self.center1=nn.Parameter(torch.Tensor(args.num_classes,2048).normal_(std=0.001))
        self.centers=[self.center1]

        init.constant_(self.bottleneck.weight.data, 1)
        init.constant_(self.bottleneck.bias.data, 0.0)
        init.normal_(self.classifier.weight.data, std=0.001)


        self.params_groups=[{'params':self.parameters()}]

        self.mode='train'

    def forward(self, x):
        """

        :param x: input
        :return: dict {name: output}
        """
        f0=self.pool(self.backbone0(x)).view(x.size(0),-1)
        f1=self.bottleneck(f0)
        if self.mode == 'train':
            sc=self.classifier(f1)
            return {'CE_Loss':[sc],'Triplet_Loss':[f0],'Center_Loss':[f0]}

        return f1





