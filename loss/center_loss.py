import torch.nn as nn
import torch

class CenterLoss(nn.Module):

    def __init__(self):
        super(CenterLoss, self).__init__()

    def forward(self, inputs, targets, centers):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True)
        dist = dist + torch.pow(centers, 2).sum(dim=1, keepdim=True).t()
        dist.addmm_(1, -2, inputs, centers.t())


        mask = targets.view(n,1).eq(torch.arange(centers.size(0)).view(1,-1).cuda().long())

        dist_ap = dist[mask]
        loss=0.5*torch.sum(dist_ap)
        return loss