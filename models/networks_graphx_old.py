# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn

from models.graphx import CNN18Encoder, PointCloudEncoder, PointCloudGraphXDecoder, PointCloudDecoder
from models.projection import Projector
from models.edge_detection import EdgeDetector 

from losses.proj_losses import *
from losses.earth_mover_distance import EMD
from utils.point_cloud_utils import Scale, Scale_one

import utils.network_utils

class Pixel2Pointcloud_GRAPHX(nn.Module):
    def __init__(self, cfg, in_channels, in_instances, activation=nn.ReLU(), optimizer=None, scheduler=None, use_graphx=True, **kwargs):
        super().__init__()
        self.cfg = cfg
        
        # Graphx
        self.img_enc = CNN18Encoder(in_channels, activation)

        out_features = [block[-2].out_channels for block in self.img_enc.children()]
        self.pc_enc = PointCloudEncoder(3, out_features, cat_pc=True, use_adain=True, use_proj=True, 
                                        activation=activation)
        
        deform_net = PointCloudGraphXDecoder if use_graphx else PointCloudDecoder
        self.pc = deform_net(2 * sum(out_features) + 3, in_instances=in_instances, activation=activation)
        
        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)
        self.kwargs = kwargs

        # 2D supervision part
        self.projector = Projector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)

        # scale
        self.scale = Scale(self.cfg)
        self.scale_one = Scale_one(self.cfg)

        # emd loss
        self.emd = EMD()
        
        if torch.cuda.is_available():
            self.img_enc = torch.nn.DataParallel(self.img_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc_enc = torch.nn.DataParallel(self.pc_enc, device_ids=cfg.CONST.DEVICE).cuda()
            self.pc = torch.nn.DataParallel(self.pc, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.scale = torch.nn.DataParallel(self.scale, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()

        # edge detector
        if self.cfg.EDGE_LOSS.USE_EDGE_LOSS:
            self.edge_detector = EdgeDetector(self.cfg)
            self.edge_proj_loss = ProjectLoss(self.cfg)
            if torch.cuda.is_available():
                self.edge_detector = torch.nn.DataParallel(self.edge_detector, device_ids=cfg.CONST.DEVICE).cuda()
                self.edge_proj_loss = torch.nn.DataParallel(self.edge_proj_loss, device_ids=cfg.CONST.DEVICE).cuda()

    def forward(self, input, init_pc):
        img_feats = self.img_enc(input)
        pc_feats = self.pc_enc(img_feats, init_pc)
        return self.pc(pc_feats)

    def loss(self, input, init_pc, gt_pc, view_az, view_el, proj_gt, edge_gt):
        pred_pc = self(input, init_pc)
        
        if self.cfg.SUPERVISION_2D.USE_AFFINITY:
           grid_dist_np = grid_dist(grid_h=self.cfg.PROJECTION.GRID_H, grid_w=self.cfg.PROJECTION.GRID_W).astype(np.float32)
           grid_dist_tensor = utils.network_utils.var_or_cuda(torch.from_numpy(grid_dist_np))
        else:
           grid_dist_tensor = None

        # Use 2D projection loss to train
        proj_pred = {}
        loss_bce = {}
        fwd = {}
        bwd = {}
        loss_fwd = {}
        loss_bwd = {}
        loss_2d = 0.
        if not self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            loss_2d = torch.tensor(loss_2d)

        # For edge 3d
        loss_3d = 0.
        if not self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            loss_3d = torch.tensor(loss_3d)

        # For edge loss
        edge_proj_pred = {}
        edge_loss_bce = {}
        edge_fwd = {}
        edge_bwd = {}
        edge_loss_fwd = {}
        edge_loss_bwd = {}
        edge_loss = 0.
        
        # for 3d supervision (EMD)
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            for idx in range(0, self.cfg.PROJECTION.NUM_VIEWS):
                # Projection
                proj_pred[idx] = self.projector(pred_pc, view_az[:,idx], view_el[:,idx])
                
                # Projection loss
                loss_bce[idx], fwd[idx], bwd[idx] = self.proj_loss(preds=proj_pred[idx], gts=proj_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
                loss_fwd[idx] = 1e-4 * torch.mean(fwd[idx])
                loss_bwd[idx] = 1e-4 * torch.mean(bwd[idx])
    
                 # Loss = projection loss + edge projection loss
                loss_2d += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(loss_bce[idx]) +\
                           self.cfg.PROJECTION.LAMDA_AFF_FWD * loss_fwd[idx] +\
                           self.cfg.PROJECTION.LAMDA_AFF_BWD * loss_bwd[idx]
                    
                if self.cfg.EDGE_LOSS.USE_EDGE_LOSS:
                    # Edge prediction of projection
                    proj_pred[idx] = proj_pred[idx].unsqueeze(1) # (BS, 1, H, W)
                    edge_proj_pred[idx] = self.edge_detector(img=proj_pred[idx])
                    edge_proj_pred[idx] = edge_proj_pred[idx].squeeze(1) # (BS, H, W)
        
                    # Edge projection loss
                    edge_loss_bce[idx], edge_fwd[idx], edge_bwd[idx] = self.proj_loss(preds=edge_proj_pred[idx], gts=edge_gt[:,idx], grid_dist_tensor=grid_dist_tensor)
                    edge_loss_fwd[idx] = 1e-4 * torch.mean(edge_fwd[idx])
                    edge_loss_bwd[idx] = 1e-4 * torch.mean(edge_bwd[idx])
    
                    edge_loss += self.cfg.PROJECTION.LAMDA_BCE * torch.mean(edge_loss_bce[idx]) +\
                                 self.cfg.PROJECTION.LAMDA_AFF_FWD * edge_loss_fwd[idx] +\
                                 self.cfg.PROJECTION.LAMDA_AFF_BWD * edge_loss_bwd[idx]    

        # 3d loss
        if cfg.REFINE.USE_REFINE == True:
            loss_3d = torch.mean(self.emd(pred_pc, gt_pc))
        else:
            loss_3d = torch.mean(self.emd(pred_pc, gt_pc))

        # Total loss
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = self.cfg.SUPERVISION_2D.LAMDA_2D_LOSS * (loss_2d/self.cfg.PROJECTION.NUM_VIEWS) +\
                         self.cfg.SUPERVISION_3D.LAMDA_3D_LOSS * loss_3d

        elif self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            total_loss = loss_2d / self.cfg.PROJECTION.NUM_VIEWS

        elif self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = loss_3d
            
        return total_loss, (loss_2d/self.cfg.PROJECTION.NUM_VIEWS), loss_3d, pred_pc

    def learn(self, input, init_pc, gt_pc, view_az, view_el, proj_gt, edge_gt):
        self.train(True)
        self.optimizer.zero_grad()
        total_loss, loss_2d, loss_3d, _ = self.loss(input, init_pc, gt_pc, view_az, view_el, proj_gt, edge_gt)
        total_loss.backward()
        self.optimizer.step()
        total_loss_np = total_loss.detach().item()
        loss_2d_np = loss_2d.detach().item()
        loss_3d_np = loss_3d.detach().item()
        del total_loss, loss_2d, loss_3d
        return total_loss_np, loss_2d_np, loss_3d_np
