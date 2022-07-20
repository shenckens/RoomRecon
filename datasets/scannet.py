# This file is derived from [NeuralRecon] (https://github.com/zju3dv/NeuralRecon)
# and [PlanarRecon] (https://github.com/neu-vi/PlanarRecon).
# Originating Author: Yiming Xie (https://ymingxie.github.io)
# Modified for [RoomRecon] (https://github.com/shenckenns/PlanarRecon) by Stijn Henckens.

import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.planes_file = 'planes_{}'.format(self.n_views)
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.planes_cashe = {}
        # ---- max_cache = 100 in NeuralRecon, 1 in PlanarRecon. ----
        self.max_cache = 100
        self.max_cashe = 1

    def build_list(self):
        with open(os.path.join(self.datapath, self.planes_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        # ---- Check if fragment files are equal or make a difference. ----
        # with open(os.path.join(self.datapath, self.planes_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
        #     metas = pickle.load(f)

        if self.mode != 'train' and len(metas) != 0:
            # make sure to save results for all scenes (utils.py:SaveScene)
            metas.append(metas[0])
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(
            filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(
            filepath, 'pose', '{}.txt'.format(vid)))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_planes(self, data_path, scene):
        if scene not in self.planes_cashe.keys():
            if len(self.planes_cashe) > self.max_cache:
                self.planes_cashe = {}
            planes = np.load(os.path.join(
                data_path, scene, 'annotation', 'planes.npy'))
            # a, b, c, -1
            planes = np.concatenate(
                [planes, - np.ones_like(planes[:, :1])], axis=-1)
            plane_points = list(
                np.load(os.path.join(data_path, scene, 'annotation', 'plane_points.npy'), allow_pickle=True))

            indices = np.load(os.path.join(data_path, scene, 'indices.npz'))
            indices = indices.f.arr_0
            self.planes_cashe[scene] = [planes, plane_points, indices]
        return self.planes_cashe[scene]

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(
            os.path.join(self.datapath, self.tsdf_file), meta['scene'])
        planes, plane_points, indices = self.read_scene_planes(
            os.path.join(self.datapath, self.planes_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(
                os.path.join(self.datapath, self.source_path, meta['scene']), vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'planes': planes,
            'plane_points': plane_points,
            'indices': indices,
            'vol_origin': meta['vol_origin'],
            'vol_dim': meta['vol_dim'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': np.array([self.epoch], dtype='int8'),
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
