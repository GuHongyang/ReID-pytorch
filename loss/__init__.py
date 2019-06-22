from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch.nn as nn

def Loss(args):
    l=args.loss.split('+')
    loss_fn={}
    for i in range(len(l)):
        l_=l[i].split('*')
        if l_[1]=='CE_Loss':
            loss_fn[l_[1]]={
                'function':nn.CrossEntropyLoss(),
                'weight':float(l_[0])
            }
        elif l_[1].split('[')[0]=='Triplet_Loss':
            loss_fn[l_[1].split('[')[0]]={
                'function':TripletLoss(float(l_[1].split('[')[1].split(']')[0])),
                'weight':float(l_[0])
            }
        elif l_[1]=='Center_Loss':
            loss_fn[l_[1]]={
                'function': CenterLoss(),
                'weight': float(l_[0])
            }


    return loss_fn