import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet18
import numpy as np
import torch.nn.init as init
import itertools

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


class PCB(nn.Module):
    def __init__(self,args):
        super(PCB,self).__init__()


        resnet = resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1,1)
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

        self.pool=nn.AdaptiveAvgPool2d((6,1))

        self.featers=nn.ModuleList()
        for i in range(6):
            feater=nn.Sequential(
                nn.Conv2d(2048,256,kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
            feater.apply(weights_init_kaiming)
            self.featers.append(feater)

        self.classifiers=nn.ModuleList()
        for i in range(6):
            classifier=nn.Linear(256,args.num_classes)
            classifier.weight.data.normal_(std=0.001)
            if classifier.bias is not None:
                classifier.bias.data.zero_()
            self.classifiers.append(classifier)


        self.params_groups=[{'params':self.backbone0.parameters()},
                            {'params':itertools.chain(self.featers.parameters(),self.classifiers.parameters())}]

        self.mode='train'

    def forward(self, x):
        """

        :param x: input
        :return: dict {name: output}
        """
        f0=self.pool(self.backbone0(x))
        fs=[]
        for i in range(6):
            fs.append(self.featers[i](f0[:,:,i:i+1,:]).squeeze(2).squeeze(2))

        if self.mode == 'train':
            sc=[]
            for i in range(6):
                sc.append(self.classifiers[i](fs[i]))
            return {'CE_Loss':sc}

        return torch.cat(fs,1)





