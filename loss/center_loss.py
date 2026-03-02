from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    This implementation is:
    - device agnostic (CPU / GPU safe)
    - numerically stable
    - compatible with modern PyTorch
    """

    def __init__(self, num_classes=751, feat_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # DO NOT force CUDA here
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature tensor of shape [B, feat_dim]
            labels: ground truth labels of shape [B]
        """
        assert x.size(0) == labels.size(0), \
            "features.size(0) must equal labels.size(0)"

        # ensure centers are on same device as features
        if self.centers.device != x.device:
            self.centers.data = self.centers.data.to(x.device)

        batch_size = x.size(0)

        # compute pairwise distance between features and centers
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        # select correct class centers
        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat[mask]
        loss = dist.clamp(min=1e-12, max=1e12).mean()

        return loss
