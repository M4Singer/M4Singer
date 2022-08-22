import torch
from torch import nn


class GDLLoss(nn.Module):
    def __init__(self, pNorm=2):
        super(GDLLoss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)

        filterX = torch.FloatTensor([[[[-1, 1]]]])  # 1x2
        filterY = torch.FloatTensor([[[[1], [-1]]]])  # 2x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt, nonpadding):
        """

        :param pred: [B, T, n_bins]
        :param gt: [B, T, n_bins]
        :param nonpadding: [B, T]
        :return:
        """
        nonpadding = nonpadding[:, None, :, None]
        pred = pred[:, None]
        gt = gt[:, None]
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())

        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)[:, :, :-1]

        # print(">>> ", grad_diff_y.shape, grad_diff_x.shape, gt.shape, nonpadding.shape)

        mat_loss_x = grad_diff_x ** self.pNorm * nonpadding

        mat_loss_y = grad_diff_y ** self.pNorm * nonpadding  # Batch x Channel x width x height

        shape = gt.shape

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (nonpadding.sum() * shape[3])

        return mean_loss
