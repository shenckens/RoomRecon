import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import PointTensor
from loguru import logger

from models.modules import SPVCNN
from utils import apply_log_transform
from .gru_fusion import GRUFusion
from ops.back_project import back_project
from ops.generate_grids import generate_grid
from tools.bin_mean_shift import Bin_Mean_Shift


class RoomNet(nn.Module):
    '''
    Coarse-to-fine network.
    '''

    def __init__(self, cfg):
        super(RoomNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        alpha = int(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 1
                 + 9, 48 + 24 * alpha + 1 + 9, 24 + 24 + 1 + 9]
        channels = [96, 48, 24]

        # ---- to Edit for both NR and PR ----

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion: follow the network design in NeuralRecon
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict tsdf, occupancy and plane.
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        self.plane_class = nn.ModuleList()
        self.plane_residual = nn.ModuleList()
        self.plane_distance = nn.ModuleList()
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))
            self.plane_class.append(nn.Linear(channels[i], 7))
            self.plane_distance.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i], bias=True),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(),
                    nn.Linear(channels[i], 1, bias=True)
                )
            )
            self.plane_residual.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i], bias=True),
                    nn.BatchNorm1d(channels[i]),
                    nn.ReLU(),
                    nn.Linear(channels[i], 7 * 3, bias=True)
                ))

        self.offset_center_preds = nn.Sequential(
            nn.Linear(channels[self.n_scales],
                      channels[self.n_scales], bias=True),
            nn.BatchNorm1d(channels[self.n_scales]),
            nn.ReLU(),
            nn.Linear(channels[self.n_scales], 3, bias=True)
        )

        self.normal_anchors = torch.from_numpy(
            np.load(self.cfg.NORMAL_ANCHOR_PATH)).float()
        self.normal_anchors = torch.nn.parameter.Parameter(
            self.normal_anchors, requires_grad=False)
        self.mean_shift = Bin_Mean_Shift(device='cuda')


    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: label_target: (Tensor), label ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            tsdf_target = inputs['tsdf_list'][scale]
            label_target = inputs['label_list'][scale]
            occ_target = inputs['occ_list'][scale]
            coords_down = coords.detach().clone().long()
            # 2 ** scale == interval
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            tsdf_target = tsdf_target[coords_down[:, 0],
                                    coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            label_target = label_target[coords_down[:, 0],
                                    coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

            occ_target = occ_target[coords_down[:, 0],
                                    coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return tsdf_target, label_target, occ_target

    def forward(self):
        return 0
