import warnings

import torch.nn as nn
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class CLRecall(nn.Module):
    def __init__(self):
        super(CLRecall, self).__init__()

    def forward(self, pred, cl):
        pred = torch.softmax(pred, dim=1)
        pred = pred[:, 2:3, :, :, :]

        inter =  torch.sum(cl * pred)
        sum_ = torch.sum(cl)

        recall = inter / (sum_  + 1e-8)

        return 1 - recall


class CustomLoss(nn.Module):
    def __init__(self, loss, weights):
        super(CustomLoss, self).__init__()
        print(f"==> Using custom loss from {__file__}")

        self.loss = DeepSupervisionWrapper(loss, weights)
        self.loss_add = CLRecall()
        self.first_iter = True

    def forward(self, net_output, target):
        if self.first_iter:
            warnings.warn(f"==> Model output shape is {net_output[0].shape}")
            self.first_iter = False

        loss = self.loss(net_output, target[:-1])
        loss_add = self.loss_add(net_output[0], target[-1])
        return loss + loss_add
