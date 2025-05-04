import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """
    Computes focal loss for a training process
    
    ** Params:
    - gamma: Rate to downweight easier examples, defaults to 2.0
    - weight: Weight tensor indicating the corresponding class weights, None by default 
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


catalog = {
    "crossentropy": (nn.CrossEntropyLoss, True),
    "focal": (FocalLoss, True)
}