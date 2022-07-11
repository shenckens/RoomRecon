import torch
import torch.nn as nn


class PlanarityNet(nn.Module):
    def __init__(self, cfg):
        super(PlanarityNet, self).__init__()
        self.cfg = cfg

    def forward(self, input, outputs, loss_dict):
        planar_loss = self.calc_mean_planar_loss()
        loss_dict.update({'planar_loss': planar_loss})
        return outputs, loss_dict

    def calc_mean_planar_loss(self):
        return 0
