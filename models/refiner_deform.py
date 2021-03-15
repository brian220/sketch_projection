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

from models.graphx import PointCloudGraphXDecoder
from models.projection import Projector
from models.cdt_encoder import Encoder as CDT_Encoder
from models.cdt_generator import Generator as CDT_Generator

from losses.proj_losses import *
from losses.earth_mover_distance import EMD

import utils.network_utils

Conv = nn.Conv2d

class Refiner(nn.Module):
    def __init__(self, cfg, in_instances, activation=nn.ReLU(), optimizer=None, scheduler=None):
        super().__init__()
        self.cfg = cfg
        
        """
        deform_net = PointCloudGraphXDecoder
        # input a point cloud in_instances * 3
        self.deformer = deform_net(3, in_instances=in_instances, activation=activation)
        """
        encoder = Encoder(input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, z_size=args.z_size).cuda()
        generator = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support, z_size=args.z_size).cuda()
        

        self.cdt_encoder = CDT_Encoder(input_channels=cfg.CDT_REFINE.INPUT_CHANNELS,
                                       relation_prior=cfg.CDT_REFINE.RELATION_PRIOR,
                                       use_xyz=True,
                                       z_size=cfg.CDT_REFINE.Z_SIZE)

        self.cdt_generator = CDT_Generator(features=cfg.CDT_REFINE.G_FEAT, 
                                           degrees=cfg.CDT_REFINE.DEGREE,
                                           support=cfg.CDT_REFINE.SUPPORT,
                                           z_size=cfg.CDT_REFINE.Z_SIZE)

        self.optimizer = None if optimizer is None else optimizer(self.parameters())
        self.scheduler = None if scheduler or optimizer is None else scheduler(self.optimizer)
        
        """
        # 2D supervision part
        self.projector = Projector(self.cfg)

        # proj loss
        self.proj_loss = ProjectLoss(self.cfg)
        
        """
        
        # emd loss
        self.emd = EMD()

        if torch.cuda.is_available():
            # self.deformer = torch.nn.DataParallel(self.deformer, device_ids=cfg.CONST.DEVICE).cuda()
            self.cdt_encoder = torch.nn.DataParallel(self.cdt_encoder, device_ids=cfg.CONST.DEVICE).cuda()
            self.cdt_generator = torch.nn.DataParallel(self.cdt_generator, device_ids=cfg.CONST.DEVICE).cuda()
            self.projector = torch.nn.DataParallel(self.projector, device_ids=cfg.CONST.DEVICE).cuda()
            self.proj_loss = torch.nn.DataParallel(self.proj_loss, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
    
    def forward(self, input_pc):
        return self.deformer(input_pc)

    def loss(self, input_pc, gt_pc, view_az, view_el, proj_gt):
        pred_pc = self(input_pc)
        # pred_pc = input_pc + displacement
        
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

        # For 3d loss
        loss_3d = 0.
        if not self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            loss_3d = torch.tensor(loss_3d)
        
        # for 2d supervision
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
        
         # 3d loss
        loss_3d = torch.mean(self.emd(pred_pc, input_pc))

        # Total loss
        if self.cfg.SUPERVISION_2D.USE_2D_LOSS and self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = self.cfg.SUPERVISION_2D.LAMDA_2D_LOSS * (loss_2d/self.cfg.PROJECTION.NUM_VIEWS) +\
                         self.cfg.SUPERVISION_3D.LAMDA_3D_LOSS * loss_3d

        elif self.cfg.SUPERVISION_2D.USE_2D_LOSS:
            total_loss = loss_2d / self.cfg.PROJECTION.NUM_VIEWS

        elif self.cfg.SUPERVISION_3D.USE_3D_LOSS:
            total_loss = loss_3d
            
        return total_loss, (loss_2d/self.cfg.PROJECTION.NUM_VIEWS), loss_3d, pred_pc

    def learn(self, input_pc, gt_pc, view_az, view_el, proj_gt):
        self.train(True)
        self.optimizer.zero_grad()
        total_loss, loss_2d, loss_3d, _ = self.loss(input_pc, gt_pc, view_az, view_el, proj_gt)
        total_loss.backward()
        self.optimizer.step()
        total_loss_np = total_loss.detach().item()
        loss_2d_np = loss_2d.detach().item()
        loss_3d_np = loss_3d.detach().item()
        del total_loss, loss_2d, loss_3d
        return total_loss_np, loss_2d_np, loss_3d_np




