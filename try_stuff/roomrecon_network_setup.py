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
        # check for channels_in
        ch_in = [80 * alpha + 1, 96 + 40 * alpha + 1
                 + 9, 48 + 24 * alpha + 1 + 9, 24 + 24 + 1 + 9]
        channels = [96, 48, 24]

        if self.cfg.FUSION.FUSION_ON:
            # GRU Fusion: follow the network design in NeuralRecon
            self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv
        self.sp_convs = nn.ModuleList()
        # MLPs that predict occupancy, tsdf, and plane.
        self.occ_preds = nn.ModuleList()
        self.tsdf_preds = nn.ModuleList()
        self.plane_class = nn.ModuleList()
        self.plane_residual = nn.ModuleList()
        self.plane_distance = nn.ModuleList()

        # i = [0, 1, 2]
        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(in_channels=ch_in[i],
                       pres=1,
                       cr=1 / 2 ** i,
                       vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                       dropout=self.cfg.SPARSEREG.DROPOUT)
            )
            self.occ_preds.append(nn.Linear(channels[i], 1))
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            # Seven plane classes??
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
            # end of for-loop

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
        # Check how this works
        self.mean_shift = Bin_Mean_Shift(device='cuda')

        def get_target(self, coords, inputs, scale):
            '''
            Won't be used when 'fusion_on' flag is turned on
            :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
            :param inputs: (List), inputs['label_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
            :param scale:
            :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
            :return: label_target: (Tensor), label ground truth for each predicted voxels, (N,)
            :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
            '''
            with torch.no_grad():
                # Added tsdf_list (needs to come from transforms.py)
                tsdf_target = inputs['tsdf_list'][scale]
                label_target = inputs['label_list'][scale]
                occ_target = inputs['occ_list'][scale]

                coords_down = coords.detach().clone().long()
                # 2 ** scale == interval
                coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
                label_target = label_target[coords_down[:, 0],
                                            coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
                occ_target = occ_target[coords_down[:, 0],
                                        coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
                tsdf_target = tsdf_target[coords_down[:, 0],
                                          coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]

                return label_target, occ_target, tsdf_target

        def upsample(self, pre_feat, pre_coords, interval, num=8):
            '''

            :param pre_feat: (Tensor), features from last level, (N, C)
            :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
            :param interval: interval of voxels, interval = scale ** 2
            :param num: 1 -> 8
            :return: up_feat : (Tensor), upsampled features, (N*8, C)
            :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
            '''
            with torch.no_grad():
                pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
                n, c = pre_feat.shape
                up_feat = pre_feat.unsqueeze(
                    1).expand(-1, num, -1).contiguous()
                up_coords = pre_coords.unsqueeze(
                    1).repeat(1, num, 1).contiguous()
                for i in range(num - 1):
                    up_coords[:, i + 1, pos_list[i]] += interval

                up_feat = up_feat.view(-1, c)
                up_coords = up_coords.view(-1, 4)

            return up_feat, up_coords

        def forward(self, features, inputs, outputs):
            '''

            :param features: list: features for each image: eg. list[0] : pyramid features for image0 : [(B, C0, H, W), (B, C1, H/2, W/2), (B, C2, H/2, W/2)]
            :param inputs: meta data from dataloader
            :param outputs: {}
            :return: outputs: dict: {
                'coords_':                  (Tensor), coordinates of voxels,
                                        (number of voxels, 4) (4 : batch ind, x, y, z)
                'feat':                     (Tensor), voxel features,
                                        (number of voxels, 24)
                'embedding':                (Tensor), planes normal (, 3), center_points (, 3), planes dis (, 1), pre_occ (, 1)
                                        (number of voxels, 8)
                others: target info.
            }
            :return: loss_dict: dict: {
                'multi_level_loss_X':         (Tensor), multi level loss
            }
            '''
            bs = features[0][0].shape[0]
            if "plane_anchors" in inputs.keys():
                anchors_gt = inputs['plane_anchors']
                residual_gt = inputs['residual']
                planes_gt = inputs['planes_trans']
                mean_xyz_gt = inputs['mean_xyz']
            else:
                anchors_gt = residual_gt = planes_gt = mean_xyz_gt = None

            pre_feat = None
            pre_coords = None
            outputs['label_target'] = []
            outputs['occ_target'] = []
            outputs['coords_'] = []
            loss_dict = {}
            # ----coarse to fine----
            for i in range(self.cfg.N_LAYER):
                interval = 2 ** (self.n_scales - i)
                scale = self.n_scales - i

                if i == 0:
                    # ----generate new coords----
                    coords = generate_grid(self.cfg.N_VOX, interval)[0]
                    up_coords = []
                    for b in range(bs):
                        up_coords.append(
                            torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * b, coords]))
                    up_coords = torch.cat(
                        up_coords, dim=1).permute(1, 0).contiguous()
                else:
                    # ----upsample coords----
                    up_feat, up_coords = self.upsample(
                        pre_feat, pre_coords, interval)
                # ----back project----
                feats = torch.stack([feat[scale] for feat in features])
                KRcam = inputs['proj_matrices'][:, :, scale].permute(
                    1, 0, 2, 3).contiguous()
                volume, count = back_project(up_coords, inputs['vol_origin_partial'], self.cfg.VOXEL_SIZE, feats,
                                             KRcam)
                grid_mask = count > 1
                # ----concat feature from last stage----
                if i != 0:
                    feat = torch.cat([volume, up_feat], dim=1)
                else:
                    feat = volume
# change get_target function to obtain tsdf gt as well
                if not self.cfg.FUSION.FUSION_ON:
                    label_target, occ_target = self.get_target(up_coords,
                                                               inputs,
                                                               scale)

                # ----convert to aligned camera coordinate----
                r_coords = up_coords.detach().clone().float()
                for b in range(bs):
                    batch_ind = torch.nonzero(
                        up_coords[:, 0] == b, as_tuple=False).squeeze(1)
                    coords_batch = up_coords[batch_ind][:, 1:].float()
                    coords_batch = coords_batch * self.cfg.VOXEL_SIZE + \
                        inputs['vol_origin_partial'][b].float()
                    coords_batch = torch.cat(
                        (coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)
                    coords_batch = coords_batch @ inputs['world_to_aligned_camera'][b, :3, :].permute(
                        1, 0).contiguous()
                    r_coords[batch_ind, 1:] = coords_batch

                # batch index is in the last position
                r_coords = r_coords[:, [1, 2, 3, 0]]

                # ----sparse conv 3d backbone----
                point_feat = PointTensor(feat, r_coords)
                feat = self.sp_convs[i](point_feat)
                # ----gru fusion----
                if self.cfg.FUSION.FUSION_ON:
                    up_coords, r_coords, feat, label_target, occ_target = self.gru_fusion(
                        up_coords, feat, inputs, i)
                    grid_mask = torch.ones_like(feat[:, 0]).bool()
                    if label_target is not None:
                        label_target = label_target.squeeze(1).long()
                        occ_target = occ_target.squeeze(1)
                # -----get occupancy and offset(for instance segmentation)----
                occ = self.occ_preds[i](feat)
# edit by me
                tsdf = self.tsdf_preds[i](feat)
#
                # class and regress plane
                class_logits = self.plane_class[i](feat)
                residuals = self.plane_residual[i](feat)
                residuals = residuals.view(-1, 7, 3)
                distance = self.plane_distance[i](feat)
                if i == self.n_scales:
                    off_center = self.offset_center_preds(feat)
                else:
                    off_center = None
                # ------define the sparsity for the next stage-----
                occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]
                occupancy[grid_mask == False] = False

                num = int(occupancy.sum().data.cpu())

                # -------compute loss-------
                if anchors_gt is not None and self.training:
                    loss = self.compute_loss(occ, class_logits, residuals, distance, off_center, occ_target, label_target,
                                             anchors_gt, residual_gt, planes_gt, mean_xyz_gt, r_coords,
                                             mask=grid_mask,
                                             pos_weight=self.cfg.POS_WEIGHT)
                else:
                    loss = torch.Tensor([0])[0]
                loss_dict.update({f'multi_level_loss_{i}': loss})
                # ------avoid out of memory: sample points if num of points is too large-----
                if self.training and num > self.cfg.TRAIN_NUM_SAMPLE[i] * bs:
                    logger.info('larger points: scale {}'.format(i))
                    choice = np.random.choice(num, num - self.cfg.TRAIN_NUM_SAMPLE[i] * bs,
                                              replace=False)
                    ind = torch.nonzero(occupancy, as_tuple=False)
                    occupancy[ind[choice]] = False

                pre_coords = up_coords[occupancy]
                for b in range(bs):
                    batch_ind = torch.nonzero(
                        pre_coords[:, 0] == b, as_tuple=False).squeeze(1)
                    if len(batch_ind) == 0:
                        logger.warning(
                            'no valid points: scale {}, batch {}'.format(i, b))
                        return outputs, loss_dict

                feat = feat[occupancy]
                pre_occ = occ[occupancy]
                pre_class = class_logits[occupancy]
                pre_distance = distance[occupancy]

                pre_feat = torch.cat(
                    [feat, pre_occ, pre_class, pre_distance], dim=1)

                output_coord_ = pre_coords.detach().clone()
                output_coord_[:, 1:] = output_coord_[:, 1:] // 2 ** scale
                outputs['coords_'].append(output_coord_.long())
                if anchors_gt is not None:
                    outputs['label_target'].append(label_target[occupancy])
                    outputs['occ_target'].append(occ_target[occupancy])
                if i == self.cfg.N_LAYER - 1:
                    # ----convert class, residuals to normals -----
                    class_probs = F.softmax(class_logits[occupancy], dim=-1)
                    class_ids = class_probs.argmax(-1)
                    idx = torch.arange(
                        class_ids.shape[0], device=class_ids.device).long()
                    residuals_pred = residuals[occupancy][idx, class_ids]
                    normals = self.normal_anchors[class_ids] + residuals_pred
                    offset_points = r_coords[occupancy, :3] + 0.12 * distance[occupancy] * normals / torch.norm(normals, dim=1,
                                                                                                                keepdim=True)
                    D = -(offset_points.unsqueeze(1) @
                          normals.unsqueeze(2)).squeeze(1)
                    planes = torch.cat([normals, D], dim=1)

                    center_points = r_coords[occupancy,
                                             :3] + off_center[occupancy]
                    center_points = torch.cat(
                        (center_points, torch.ones_like(center_points[:, :1])), dim=1)
                    for b in range(bs):
                        # convert coordinate
                        ind = torch.nonzero(
                            pre_coords[:, 0] == b, as_tuple=False).squeeze(1)
                        planes[ind] = (inputs['world_to_aligned_camera'][b].transpose(0,
                                                                                      1) @ planes[ind].transpose(
                            0, 1)).transpose(0, 1)
                        if anchors_gt is not None:
                            planes_gt[b] = (inputs['world_to_aligned_camera'][b].transpose(0,
                                                                                           1) @ planes_gt[b].transpose(
                                0, 1)).transpose(0, 1)

                        center_points[ind] = center_points[ind] @ torch.inverse(
                            inputs['world_to_aligned_camera'])[b].permute(1, 0).contiguous()
                    center_points = center_points[:, :3]

                    # A,B,C,D,X,Y,Z,OCC for vote
                    # planes = planes[:, :3] # / planes[:, 3:]
                    embedding = torch.cat(
                        [planes[:, :3], center_points, planes[:, 3:], pre_occ], dim=1)

                    outputs['embedding'] = embedding
                    outputs['planes_gt'] = planes_gt
                    outputs['feat'] = feat

            return outputs, loss_dict
