#reference: https://github.com/val-iisc/capnet/blob/master/src/proj_codes.py

from __future__ import division
import math
import numpy as np

import torch



def perspective_transform(xyz, batch_size):
    '''
    Perspective transform of pcl; Intrinsic camera parameters are assumed to be
    known (here, obtained using parameters of GT image renderer, i.e. Blender)
    Here, output grid size is assumed to be (64,64) in the K matrix
    TODO: use output grid size as argument
    args:
            xyz: float, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
    returns:
            xyz_out: float, (BS,N_PTS,3); perspective transformed point cloud 
    '''
    K = np.array([
            [120., 0., -32.],
            [0., 120., -32.],
            [0., 0., 1.]]).astype(np.float32)
    K = np.expand_dims(K, 0)
    K = np.tile(K, [batch_size,1,1])
    
    xyz_out = torch.matmul(K, xyz.permute(0, 2, 1))
    xy_out = xyz_out[:,:2]/abs(torch.unsqueeze(xyz[:,:,2],1))

    xyz_out = torch.cat([xy_out, abs(xyz_out[:,2:])],dim=1)
    return xyz_out.permute(0, 2, 1)


def world2cam(xyz, az, el, batch_size, N_PTS=1024):
    '''
    Convert pcl from world co-ordinates to camera co-ordinates
    args:
            xyz: float tensor, (BS,N_PTS,3); input point cloud
                     values assumed to be in (-1,1)
            az: float tensor, (BS); azimuthal angle of camera in radians
            elevation: float tensor, (BS); elevation of camera in radians
            batch_size: int, (); batch size
            N_PTS: float, (); number of points in point cloud
    returns:
            xyz_out: float tensor, (BS,N_PTS,3); output point cloud in camera
                        co-ordinates
    '''
    # Camera origin calculation - az,el,d to 3D co-ord

    # Rotation
    rotmat_az=[
                [torch.ones_like(az),torch.zeros_like(az),torch.zeros_like(az)],
                [torch.zeros_like(az),torch.cos(az),-torch.sin(az)],
                [torch.zeros_like(az),torch.sin(az),torch.cos(az)]
                ]
    rotmat_az = [ torch.stack(x) for x in rotmat_az ]

    rotmat_el=[
                [torch.cos(el),torch.zeros_like(az), torch.sin(el)],
                [torch.zeros_like(az),torch.ones_like(az),torch.zeros_like(az)],
                [-torch.sin(el),torch.zeros_like(az), torch.cos(el)]
                ]
    rotmat_el = [ torch.stack(x) for x in rotmat_el ]

    rotmat_az = torch.stack(rotmat_az, 0) # [3,3,B]
    rotmat_el = torch.stack(rotmat_el, 0) # [3,3,B]
    rotmat_az = rotmat_az.permute(2, 0, 1) # [B,3,3]
    rotmat_el = rotmat_el.permute(2, 0, 1) # [B,3,3]
    rotmat = torch.matmul(rotmat_el, rotmat_az)
    
    # Transformation(t)
    # Distance of object from camera - fixed to 2
    d = 2.
    # Calculate translation params
    tx, ty, tz = [0, 0, d]
    
    tr_mat = torch.unsqueeze(torch.tensor([tx, ty, tz]), 0).repeat(batch_size,1) # [B,3]
    tr_mat = torch.unsqueeze(tr_mat,2) # [B,3,1]
    tr_mat = tr_mat.permute(0, 2, 1) # [B,1,3]
    tr_mat = tr_mat.repeat(1, N_PTS, 1) # [B,1024,3]

    xyz_out = torch.matmul(rotmat, xyz.permute(0, 2, 1)) - tr_mat.permute(0, 2, 1)
    
    return xyz_out.permute(0, 2, 1)

