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

class ResNet_50_2(nn.Module):
    def __init__(self,args):
        super(ResNet_50_2,self).__init__()


        resnet = resnet50(pretrained=True)

        self.backbone0=nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
        )
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4

        self.pool=nn.AdaptiveAvgPool2d((1,1))

        self.featers=nn.ModuleList()
        for i in range(3):
            bottleneck = nn.BatchNorm1d(1024//(2**(2-i)))
            bottleneck.bias.requires_grad_(False)

            feater=nn.Sequential(
                nn.Linear(2048//(2**(2-i)),1024//(2**(2-i))),
                bottleneck
            )
            feater.apply(weights_init_kaiming)
            self.featers.append(feater)

        # self.classifier=nn.Linear(1024,args.num_classes,bias=False)
        self.classifiers=nn.ModuleList()
        for i in range(3):
            classifier=Normal_Linear(1024//(2**(2-i)),args.num_classes)
            classifier.weight.data.normal_(std=0.001)
            if classifier.bias is not None:
                classifier.bias.data.zero_()
            self.classifiers.append(classifier)


        self.params_groups=[{'params':self.parameters()}]

        self.mode='train'

    def forward(self, x):
        """

        :param x: input
        :return: dict {name: output}
        """
        f0=self.backbone0(x)
        f2=self.layer2(f0)
        f3=self.layer3(f2)
        f4=self.layer4(f3)

        f2_=self.featers[0](self.pool(f2).view(x.size(0),-1))
        f3_=self.featers[1](self.pool(f3).view(x.size(0),-1))
        f4_=self.featers[2](self.pool(f4).view(x.size(0),-1))

        if self.mode == 'train':
            sc1=self.classifiers[0](f2_)
            sc2=self.classifiers[1](f3_)
            sc3=self.classifiers[2](f4_)
            return {'CE_Loss':[sc1,sc2,sc3]}

        return torch.cat([f2_,f3_,f4_],1)





