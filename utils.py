import os
import torch
from torch.serialization import save
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
import cv2
# from tools.bin_mean_shift import Bin_Mean_Shift
# from tools.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
# from scipy.spatial import Delaunay
# from tools.generate_planes import furthest_point_sampling, project2plane, writePointCloudFace
# from tools.random_color import random_color
# import trimesh


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))
