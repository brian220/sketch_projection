# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/rec.json'

__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_sketch/%s/%s/render_%d.png'
__C.DATASETS.SHAPENET.DEPTH_PATH            = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_sketch/%s/%s/depth_%d.png'
# __C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_pointclouds/%s/%s/pointcloud_2048.npy'
__C.DATASETS.SHAPENET.POINT_CLOUD_PATH      = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/shape_net_core_uniform_samples_2048/%s/%s.ply'
__C.DATASETS.SHAPENET.VIEW_PATH             = '/media/caig/FECA2C89CA2C406F/sketch3D/dataset/capnet_data/data/ShapeNet_sketch/%s/%s/view.txt'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.TOTAL_VIEWS                     = 10
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
__C.DATASET.CLASS                           = 'chair'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = [0]
__C.CONST.DEVICE_NUM                        = 1
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 64       # Image width for input
__C.CONST.IMG_H                             = 64       # Image height for input
__C.CONST.BATCH_SIZE                        = 2
__C.CONST.CROP_IMG_W                        = 200       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 200       # Dummy property for Pascal 3D
__C.CONST.BIN_SIZE                          = 15
__C.CONST.NUM_POINTS                        = 2048
__C.CONST.WEIGHTS                           = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/output_results/output_xy_plane_init/checkpoints/best-reconstruction-ckpt.pth'

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/output'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.REC_MODEL                       = 'GRAPHX' # GRAPHX or PSGN_FC
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False

#
# GraphX
#
__C.GRAPHX                                 = edict()
__C.GRAPHX.USE_GRAPHX                      = True
__C.GRAPHX.NUM_INIT_POINTS                 = 2048

#
# 2d supervision
#
__C.SUPERVISION_2D                         = edict()
__C.SUPERVISION_2D.LOSS_TYPE               = 'bce_prob'
__C.SUPERVISION_2D.USE_AFFINITY            = False
__C.SUPERVISION_2D.USE_2D_LOSS             = False
__C.SUPERVISION_2D.LAMDA_2D_LOSS           = 10.

#
# 3d supervision
#
__C.SUPERVISION_3D                         = edict()
__C.SUPERVISION_3D.USE_3D_LOSS             = True
__C.SUPERVISION_3D.LAMDA_3D_LOSS           = 1.

# refine model
# deformation
__C.REFINE                                 = edict()
__C.REFINE.USE_REFINE                      = True
__C.REFINE.REFINE_MODEL                    = 'GRAPHX'
# CDT refine
# encoder
__C.CDT_REFINE                             = edict()
__C.CDT_REFINE.INPUT_CHANNELS              = 0
__C.CDT_REFINE.RELATION_PRIOR              = 1
# generator
__C.CDT_REFINE.G_FEAT                      = [256, 512, 256, 256, 128, 128, 64, 3]
__C.CDT_REFINE.DEGREE                      = [2, 2, 2, 2, 2, 4, 16]
__C.CDT_REFINE.SUPPORT                     = 10
# both
__C.CDT_REFINE.Z_SIZE                      = 256


#
# Edge loss
# 
__C.EDGE_LOSS                              = edict()
__C.EDGE_LOSS.USE_EDGE_LOSS                = False
__C.EDGE_LOSS.LAMDA_EDGE_LOSS              = 5e-1

#
# Continuous Projection
#
__C.PROJECTION                             = edict()
__C.PROJECTION.GRID_H                      = 64
__C.PROJECTION.GRID_W                      = 64
__C.PROJECTION.SIGMA_SQ                    = 0.5
__C.PROJECTION.NUM_VIEWS                   = 4
# __C.PROJECTION.NUM_VIEWS                   = 1 # only for test time optimization
__C.PROJECTION.LAMDA_BCE                   = 1.
__C.PROJECTION.LAMDA_AFF_FWD               = 1.
__C.PROJECTION.LAMDA_AFF_BWD               = 1.

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = True
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 1000

# __C.TRAIN.VIEW_ESTIMATOR_LEARNING_RATE      = 5e-4
__C.TRAIN.MILESTONES                        = [400]
__C.TRAIN.VIEW_ESTIMATOR_LR_MILESTONES      = [400]

# train parameters for graphx
__C.TRAIN.CDT_REFINE_LEARNING_RATE          = 4e-4

# train parameters for graphx
__C.TRAIN.GRAPHX_LEARNING_RATE              = 5e-5
__C.TRAIN.GRAPHX_WEIGHT_DECAY               = 1e-5

# train parameters for psgn fc
__C.TRAIN.PSGN_FC_LEARNING_RATE             = 5e-5
__C.TRAIN.PSGN_FC_CONV_WEIGHT_DECAY         = 1e-5
__C.TRAIN.PSGN_FC_FC_WEIGHT_DECAY           = 1e-3

__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .3
__C.TRAIN.SAVE_FREQ                         = 100            # weights will be overwritten every save_freq epoch
__C.TRAIN.TRANS_LAMDA                       = 10

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.NUM_WORKER                         = 4             # number of data workers
__C.TEST.RECONSTRUCTION_WEIGHTS             = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/output_3d/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST.VIEW_ENCODER_WEIGHTS               = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST.VIEW_ESTIMATION_WEIGHTS            = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-view-ckpt.pth'
__C.TEST.RESULT_PATH                        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/output_3d/test.txt'

#
# Evaluating options
#
__C.EVALUATE                                = edict()
__C.EVALUATE.TAXONOMY_ID                    = '03001627'
__C.EVALUATE.INPUT_IMAGE_FOLDER             = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/compare_result_3_18_senior_meeting/output_2d_224_pretrained/evaluate/evaluate_input_img'
__C.EVALUATE.OUTPUT_FOLDER                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/compare_result_3_18_senior_meeting/output_2d_224_pretrained/evaluate/evaluate_output'
__C.EVALUATE.INFO_FILE                      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/compare_result_3_18_senior_meeting/output_2d_224_pretrained/evaluate/eval_chair.txt'
__C.EVALUATE.RECONSTRUCTION_WEIGHTS         = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/compare_result_3_18_senior_meeting/output_2d_224_pretrained/checkpoints/best-reconstruction-ckpt.pth'

__C.EVALUATE.VIEW_ENCODER_WEIGHTS           = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-reconstruction-ckpt.pth'
__C.EVALUATE.VIEW_ESTIMATION_WEIGHTS        = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v3/checkpoints/best-view-ckpt.pth'

#
# Evaluate on the true habd draw image
#
__C.EVALUATE_HAND_DRAW                                = edict()
__C.EVALUATE_HAND_DRAW.INPUT_IMAGE_FOLDER             = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/hand_draw_input_img/'
__C.EVALUATE_HAND_DRAW.OUTPUT_FOLDER                  = '/media/itri/Files_2TB/chaoyu/pointcloud3d/pc3d/evaluate/hand_draw_output/'
__C.EVALUATE_HAND_DRAW.RECONSTRUCTION_WEIGHTS         = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-reconstruction-ckpt.pth'
__C.EVALUATE_HAND_DRAW.VIEW_ESTIMATION_WEIGHTS        = '/media/itri/Files_2TB/chaoyu/pointcloud3d/output/new_output/checkpoints/2020-12-17T11:56:09.024186/best-view-ckpt.pth'

#
# Evaluating options
#
__C.EVALUATE_FIXED_VIEW                             = edict()
__C.EVALUATE_FIXED_VIEW.TAXONOMY_ID                 = '03001627'
__C.EVALUATE_FIXED_VIEW.RESULT_DIR                  = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/'
__C.EVALUATE_FIXED_VIEW.SAMPLE_FILE                 = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/evaluate_sample.txt'
__C.EVALUATE_FIXED_VIEW.VIEW_FILE                   = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/evaluate_fixed_view/fixed_view.txt'
__C.EVALUATE_FIXED_VIEW.RECONSTRUCTION_WEIGHTS      = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch2pointcloud/output_v4/checkpoints/best-reconstruction-ckpt.pth'

#
# Test time optimization
#
__C.TEST_OPT                                        = edict()
__C.TEST_OPT.RECONSTRUCTION_WEIGHTS                 = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/output_xy_plane_init_l210/checkpoints/best-reconstruction-ckpt.pth'
__C.TEST_OPT.OUT_PATH                               = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_projection/test_opt'