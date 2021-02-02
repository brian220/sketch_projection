from __future__ import division
import math
import numpy as np

import torch
from scipy.spatial.distance import cdist as np_cdist

def get_loss_proj(pred, gt, loss_type='bce', w=1., min_dist_loss=None,
        dist_mat=None, args=None, grid_h=64, grid_w=64):
    
    if loss_type == 'bce':
        print ('\nBCE Logits Loss\n')
        loss_function = torch.nn.BCEWithLogitsLoss(weight = None, reduction='none')
        loss = loss_function(pred, gt)
    """
    if loss == 'weighted_bce':
        print '\nWeighted BCE Logits Loss\n'
        loss = tf.nn.weighted_cross_entropy_with_logits(targets=gt, logits=pred, 
                        pos_weight=0.5)
    if loss == 'l2_sq':
        print '\nL2 Squared Loss\n'
        loss = (pred-gt)**2

    if loss == 'l1':
        print '\nL1 Loss\n'
        loss = abs(pred-gt)
    """
    if loss_type == 'bce_prob':
        print ('\nBCE Loss\n')
        epsilon = 1e-8
        loss = -gt*torch.log(pred+epsilon)*w - (1-gt)*torch.log(torch.abs(1-pred-epsilon))

    if min_dist_loss != None:
        # Affinity loss - essentially 2D chamfer distance between GT and 
        # predicted masks
        dist_mat += 1.
        gt_white = torch.unsqueeze(torch.unsqueeze(gt, 3), 3)
        gt_white = gt_white.repeat(1, 1, 1, 64, 64)
        
        pred_white = torch.unsqueeze(torch.unsqueeze(pred, 3), 3)
        pred_white = pred_white.repeat(1, 1, 1, 64, 64)
        
        pred_mask = (pred_white) + ((1.-pred_white))*1e6*torch.ones_like(pred_white)
        dist_masked_inv = gt_white * dist_mat * (pred_mask)
        
        gt_white_th = gt_white + (1.-gt_white)*1e6*torch.ones_like(gt_white)
        dist_masked = gt_white_th * dist_mat * pred_white
        
        #min_dist = torch.amin(dist_masked, dim=(3,4))
        #min_dist_inv = torch.amin(dist_masked_inv, dim=(3,4))

    return torch.mean(loss), dist_masked, dist_masked_inv


def grid_dist(grid_h, grid_w):
    '''
    Compute distance between every point in grid to every other point
    '''
    x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
    grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
    grid_dist = np_cdist(grid,grid)
    grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
    return grid_dist
