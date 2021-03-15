# -*- coding: utf-8 -*-
#
# Developed by Chao Yu Huang <b608390@gmail.com>
#
# References:
# - https://github.com/hzxie/Pix2Vox/blob/master/models/encoder.py
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

from collections import OrderedDict
from functools import partial
from itertools import chain
import numpy as np
import torch
import torch.nn as nn

from models.cdt_encoder import Encoder as CDT_Encoder
from models.cdt_generator import Generator as CDT_Generator

from losses.earth_mover_distance import EMD

import utils.network_utils

Conv = nn.Conv2d

class Refiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cdt_encoder = CDT_Encoder(input_channels=cfg.CDT_REFINE.INPUT_CHANNELS,
                                       relation_prior=cfg.CDT_REFINE.RELATION_PRIOR,
                                       use_xyz=True,
                                       z_size=cfg.CDT_REFINE.Z_SIZE)

        self.cdt_generator = CDT_Generator(features=cfg.CDT_REFINE.G_FEAT, 
                                           degrees=cfg.CDT_REFINE.DEGREE,
                                           support=cfg.CDT_REFINE.SUPPORT,
                                           z_size=cfg.CDT_REFINE.Z_SIZE)
        
        self.optimizer = torch.optim.Adam(chain(self.cdt_encoder.parameters(), self.cdt_generator.parameters()), lr=cfg.TRAIN.CDT_REFINE_LEARNING_RATE, betas=(0, 0.99))
        
        # emd loss
        self.emd = EMD()

        if torch.cuda.is_available():
            self.cdt_encoder = torch.nn.DataParallel(self.cdt_encoder, device_ids=cfg.CONST.DEVICE).cuda()
            self.cdt_generator = torch.nn.DataParallel(self.cdt_generator, device_ids=cfg.CONST.DEVICE).cuda()
            self.emd = torch.nn.DataParallel(self.emd, device_ids=cfg.CONST.DEVICE).cuda()
            self.cuda()
    
    def forward(self, input_pc):
        z_vector = self.cdt_encoder(input_pc)
        pred_pc = self.cdt_generator(z_vector)
        return pred_pc

    def loss(self, input_pc, gt_pc, view_az, view_el, proj_gt):
        pred_pc = self(input_pc)

        loss_2d = 0.
        loss_2d = torch.tensor(loss_2d)

        # 3d loss
        loss_3d = torch.mean(self.emd(pred_pc, gt_pc))

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




