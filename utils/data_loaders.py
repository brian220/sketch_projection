# -*- coding: utf-8 -*-
#
# Developed by Chaoyu Huang 
# email: b608390.cs04@nctu.edu.tw
# Lot's of code reference to Pix2Vox: 
# https://github.com/hzxie/Pix2Vox

import cv2
import json
import numpy as np
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset

from datetime import datetime as dt
from enum import Enum, unique

from pyntcloud import PyntCloud


@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


# //////////////////////////////// = End of DatasetType Class Definition = ///////////////////////////////// #
def init_pointcloud_loader(num_points):
    Z = np.random.rand(num_points) + 1.
    h = np.random.uniform(10., 214., size=(num_points,))
    w = np.random.uniform(10., 214., size=(num_points,))
    X = (w - 111.5) / 248. * -Z
    Y = (h - 111.5) / 248. * Z
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    XYZ = np.concatenate((X, Y, Z), 1)
    return XYZ.astype('float32')


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, init_num_points, proj_num_views, grid_h, grid_w, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.init_num_points = init_num_points
        self.proj_num_views = proj_num_views
        self.transforms = transforms
        self.grid_h = grid_h
        self.grid_w = grid_w
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, \
        model_gt, model_x, model_y, \
        init_point_cloud, ground_truth_point_cloud = self.get_datum(idx)

        if self.transforms:
            rendering_images = self.transforms(rendering_images)

        return (taxonomy_name, sample_name, rendering_images,
                model_gt, model_x, model_y, 
                init_point_cloud, ground_truth_point_cloud)

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        depth_image_paths = self.file_list[idx]['depth_images']
        ground_truth_point_cloud_path = self.file_list[idx]['point_cloud']
        radian_x = self.file_list[idx]['radian_x']
        radian_y = self.file_list[idx]['radian_y']

        # get data of rendering images (sample 1 image from paths)
        if self.dataset_type == DatasetType.TRAIN:
            rand_id = random.randint(0, len(rendering_image_paths) - 1)
            selected_rendering_image_path = rendering_image_paths[rand_id]
        else:
        # test, valid with the first image
            selected_rendering_image_path = rendering_image_paths[1]
        
        # read the test, train image
        # print(selected_rendering_image_path)
        rendering_images = []
        # rendering_image =  cv2.imread(selected_rendering_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        rendering_image = cv2.imread(selected_rendering_image_path).astype(np.float32) / 255.
        rendering_image = cv2.resize(rendering_image, (self.grid_w, self.grid_h))
        rendering_image = cv2.cvtColor(rendering_image, cv2.COLOR_BGR2RGB)

        if len(rendering_image.shape) < 3:
            print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                     (dt.now(), image_path))
            sys.exit(2)
        rendering_images.append(rendering_image)
        
        # read the ground truth proj images and views (multi-views)
        model_gt = []
        model_x = []
        model_y = []
        for idx in range(0, self.proj_num_views):
            # read the proj imgs
            proj_path = depth_image_paths[idx]
            ip_proj = cv2.imread(proj_path)[:,:,0]
            ip_proj = cv2.resize(ip_proj, (self.grid_h,self.grid_w))
            ip_proj[ip_proj<254] = 1
            ip_proj[ip_proj>=254] = 0
            ip_proj = ip_proj.astype(np.float32)
            model_gt.append(ip_proj)

            # read the views
            model_x.append(radian_x[idx])
            model_y.append(radian_y[idx])
        
        # get data of point cloud
        _, suffix = os.path.splitext(ground_truth_point_cloud_path)

        if suffix == '.ply':
            ground_truth_point_cloud = PyntCloud.from_file(ground_truth_point_cloud_path)
            
        # convert to np array
        rendering_images = np.array(rendering_images).astype(np.float32)
        model_gt = np.array(model_gt).astype(np.float32)
        model_x = np.array(model_x).astype(np.float32)
        model_y = np.array(model_y).astype(np.float32)
        ground_truth_point_cloud = np.array(ground_truth_point_cloud.points).astype(np.float32)

        return (taxonomy_name, sample_name, rendering_images,
                model_gt, model_x, model_y,
                init_pointcloud_loader(self.init_num_points), ground_truth_point_cloud)


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #


class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.depth_image_path_template = cfg.DATASETS.SHAPENET.DEPTH_PATH
        self.point_cloud_path_template = cfg.DATASETS.SHAPENET.POINT_CLOUD_PATH
        self.view_path_template = cfg.DATASETS.SHAPENET.VIEW_PATH

        self.class_name = cfg.DATASET.CLASS
        self.total_views = cfg.DATASET.TOTAL_VIEWS
        self.init_num_points = cfg.GRAPHX.NUM_INIT_POINTS
        self.proj_num_views = cfg.PROJECTION.NUM_VIEWS
        self.grid_h = cfg.PROJECTION.GRID_H 
        self.grid_w = cfg.PROJECTION.GRID_W
        
        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())
        
        # Get the class data (in dict) from taxonomy
        self.dataset_class_data_taxonomy = self.dataset_taxonomy[self.class_name]

    def get_dataset(self, dataset_type, transforms=None):
        taxonomy_folder_name = self.dataset_class_data_taxonomy['taxonomy_id']
        print('[INFO] %s Collecting files of Taxonomy[ID=%s, Name=%s]' %
                (dt.now(), self.dataset_class_data_taxonomy['taxonomy_id'], self.class_name))
        
        samples = []
        if dataset_type == DatasetType.TRAIN:
            samples = self.dataset_class_data_taxonomy['train']
        elif dataset_type == DatasetType.TEST:
            samples = self.dataset_class_data_taxonomy['test']
        elif dataset_type == DatasetType.VAL:
            samples = self.dataset_class_data_taxonomy['val']

        files = self.get_files_of_taxonomy(taxonomy_folder_name, samples)

        print('[INFO] %s Complete collecting files of the dataset. Total files: %d.' % (dt.now(), len(files)))
        return ShapeNetDataset(dataset_type, files, self.init_num_points, self.proj_num_views, self.grid_h, self.grid_w, transforms)
        
    def get_files_of_taxonomy(self, taxonomy_folder_name, samples):
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file paths of pointcloud
            point_cloud_file_path = self.point_cloud_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(point_cloud_file_path):
                print('[WARN] %s Ignore sample %s/%s since point cloud file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue
            
            # Get file paths of rendering images
            rendering_image_indexes = range(self.total_views)
            rendering_image_file_paths = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue
                rendering_image_file_paths.append(img_file_path)
            
            if len(rendering_image_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since image files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get file list of depth images
            depth_image_indexes = range(self.total_views)
            depth_image_file_paths = []
            for image_idx in depth_image_indexes:
                depth_image_file_path = self.depth_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(depth_image_file_path):
                    continue
                depth_image_file_paths.append(depth_image_file_path)

            if len(depth_image_file_paths) == 0:
                print('[WARN] %s Ignore sample %s/%s since depth files not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue

            # Get views of objects (azimuth, elevation)
            view_path = self.view_path_template % (taxonomy_folder_name, sample_name)
            if not os.path.exists(view_path):
                print('[WARN] %s Ignore sample %s/%s since view file not exists.' %
                      (dt.now(), taxonomy_folder_name, sample_name))
                continue
            
            angles = []
            with open(view_path) as f:
                angles = [item.split('\n')[0] for item in f.readlines()]
            
            radian_x = [] # azi
            radian_y = [] # ele
            for angle in angles:
                angle_x = float(angle.split(' ')[0])
                angle_y = float(angle.split(' ')[1])
                # convert angles to radians
                radian_x.append(angle_x*np.pi/180.)
                radian_y.append(angle_y*np.pi/180.)
                
            # Append to the list of rendering images
            files_of_taxonomy.append({
                'taxonomy_name': taxonomy_folder_name,
                'sample_name': sample_name,
                'rendering_images': rendering_image_file_paths,
                'depth_images' : depth_image_file_paths,
                'point_cloud': point_cloud_file_path,
                'radian_x' : radian_x,
                'radian_y' : radian_y
            })

            # Report the progress of reading dataset
            # if sample_idx % 500 == 499 or sample_idx == n_samples - 1:
            #     print('[INFO] %s Collecting %d of %d' % (dt.now(), sample_idx + 1, n_samples))
            
        return files_of_taxonomy


# /////////////////////////////// = End of ShapeNetDataLoader Class Definition = /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader
    # 'Pascal3D': Pascal3dDataLoader, # not implemented
    # 'Pix3D': Pix3dDataLoader # not implemented
} 

