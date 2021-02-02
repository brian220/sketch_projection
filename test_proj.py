import torch
import numpy as np
from proj_codes_pytorch import world2cam, perspective_transform, cont_proj
from proj_losses_pytorch import *
import scipy.misc as sc
import cv2

batch_size = 1
OUTPUT_PCL_SIZE = 2048
az = -87
el = -15
grid_h = 64
grid_w = 64
SIGMA_SQ = 0.5

PCL_PATH = "/home/caig/Desktop/sketch3d/capnet/data/ShapeNet_pointclouds/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/pointcloud_2048.npy"
PROJ_PATH = "/home/caig/Downloads/blenderRenderPreprocess/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/depth_3.png"

def get_test_batch_gt(proj_path):
    batch_gt = []
    model_gt = []
    ip_proj = cv2.imread(proj_path)[:,:,0]
    ip_proj = cv2.resize(ip_proj, (grid_w,grid_h))
    ip_proj[ip_proj<254] = 1
    ip_proj[ip_proj>=254] = 0
    ip_proj = ip_proj.astype(np.float32)
    model_gt.append(ip_proj)
    batch_gt.append(model_gt)

    return np.array(batch_gt).astype(np.float32)

# Test pcl
pcl = np.load(PCL_PATH).astype(np.float32)
gt_pcl_batch = []
gt_pcl_batch.append(pcl.astype(np.float32))
gt_pcl_batch = np.array(gt_pcl_batch)
gt_pcl_batch = torch.from_numpy(gt_pcl_batch)

batchx = []
batchy = []
batchx.append(float(az))
batchy.append(float(el))
batchx = np.array(batchx).astype(np.float32)
batchy = np.array(batchy).astype(np.float32)
batchx = torch.from_numpy(batchx)
batchy = torch.from_numpy(batchy)

grid_dist_tensor = torch.from_numpy(grid_dist(grid_h, grid_w).astype(np.float32))
print("grid_dist_tensor", grid_dist_tensor.dtype)

proj_gt = torch.from_numpy(get_test_batch_gt(PROJ_PATH))
print("proj_gt", proj_gt)

# World co-ordinates to camera co-ordinates
pcl_out_rot = world2cam(gt_pcl_batch, batchx, batchy, 
                        batch_size, OUTPUT_PCL_SIZE)
pcl_out_persp = perspective_transform(pcl_out_rot, batch_size)

proj_pred = cont_proj(pcl_out_persp, grid_h, grid_w, SIGMA_SQ)
print("proj_pred", proj_pred)

loss_bce, fwd, bwd = get_loss_proj(proj_pred, 
            proj_gt[:,0], 'bce_prob', 1.0, True, grid_dist_tensor, args=None, grid_h=grid_h, grid_w=grid_w)

print(loss_bce)
# print(fwd)
# print(bwd)

"""
pcl_out_rot = pcl_out_rot.numpy()
sc.imsave('test_proj_mask_pytorch.png', pcl_out_rot[0])
"""




