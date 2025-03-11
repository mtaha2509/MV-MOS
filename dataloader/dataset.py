#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data


class SemKITTI(data.Dataset):
    def __init__(self, data_config_path, data_path, imageset='train', return_ref=False, residual=1,
                 residual_path=None, drop_few_static_frames=True,movable = False):
        self.return_ref = return_ref
        self.movable = movable
        with open(data_config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.moving_learning_map = semkittiyaml['moving_learning_map']
        self.movable_learning_map = semkittiyaml['movable_learning_map']
        self.imageset = imageset
        if imageset == 'train':
            self.split = semkittiyaml['split']['train']
        elif imageset == 'val':
            self.split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            self.split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.scan_files = {}
        self.residual = residual
        self.residual_files = {}
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files = []
            scan_files += absoluteFilePaths('/'.join([data_path, str(seq).zfill(2), 'velodyne']))
            scan_files.sort()
            self.scan_files[seq] = scan_files
            if self.residual > 0:
                residual_files = []
                residual_files += absoluteFilePaths('/'.join(
                    [residual_path, str(seq).zfill(2), 'residual_images']))  # residual_images_4  residual_images
                residual_files.sort()
                self.residual_files[seq] = residual_files

        if imageset == 'train' and drop_few_static_frames:
            self.remove_few_static_frames()

        scan_files = []
        residual_files = []
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files += self.scan_files[seq]
            if self.residual > 0:
                residual_files += self.residual_files[seq]
        self.scan_files = scan_files
        if self.residual > 0:
            self.residual_files = residual_files

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # This function is used to clear some frames, because too many static frames will lead to a long training time

        remove_mapping_path = "config/train_split_dynamic_pointnumber.txt"
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}  # 加载到dict中
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if int(seq) in self.split:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        total_raw_len = 0
        total_new_len = 0
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.scan_files[seq])

                # lidar scan files
                scan_files = self.scan_files[seq]
                useful_scan_paths = [path for path in scan_files if os.path.split(path)[-1][:-4] in pending_dict[seq]]
                self.scan_files[seq] = useful_scan_paths

                if self.residual:
                    residual_files = self.residual_files[seq]
                    useful_residual_paths = [path for path in residual_files if
                                             os.path.split(path)[-1][:-4] in pending_dict[seq]]
                    self.residual_files[seq] = useful_residual_paths
                    print("seq",seq)
                    # print("useful_scan_paths",len(useful_scan_paths))
                    # print("useful_residual_paths",len(useful_residual_paths))
                    assert (len(useful_scan_paths) == len(useful_residual_paths))
                new_len = len(self.scan_files[seq])
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")
                total_raw_len += raw_len
                total_new_len += new_len
        print(f"Totally drop {total_raw_len - total_new_len}: {total_raw_len} -> {total_new_len}")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scan_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.scan_files[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            moving_labels = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            movable_labels = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            labels = np.fromfile(self.scan_files[index].replace('velodyne', 'labels')[:-3] + 'label',dtype=np.int32).reshape((-1, 1))
            labels = labels & 0xFFFF  # delete high 16 digits binary
            moving_labels = np.vectorize(self.moving_learning_map.__getitem__)(labels)
            movable_labels = np.vectorize(self.movable_learning_map.__getitem__)(labels)
        
        # add movable
        data_tuple = (raw_data[:, :3], moving_labels.astype(np.uint8),movable_labels.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3],)

        if self.residual > 0:
            residual_data = np.load(self.residual_files[index])
            data_tuple += (residual_data,)  # (x y z), label, ref, residual_n

        # print("len(data_tuple)",len(data_tuple))
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size,
                 rotate_aug=False, flip_aug=False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 return_test=False,
                 fixed_volume_space=True,
                 max_volume_space=[50.15, np.pi, 2], min_volume_space=[1.85, -np.pi, -4],
                 ignore_label = 255):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        # max_volume_space = [50.15, np.pi, 2]  # 最里面一格和最外面用来收集ignore的点，防止与不ignore的点放在一起
        # min_volume_space = [1.85, -np.pi, -4]
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.ignore_label = ignore_label
        self.fov_up=3.0
        self.fov_down=-25.0
        self.max_range=50.15
        self.min_range=1.85
        self.proj_H=64
        self.proj_W=2048
        self.max_points = 130000
        self.sizeWH = (self.proj_W, self.proj_H)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def p2r_flow_matrix(self, range_idx, polar_idx):
        """
        range_idx: [H, W] indicates the location of each range pixel on point clouds
        polar_idx: [N, 3] indicates the location of each points on polar grids
        """
        H, W = range_idx.shape
        N, K = polar_idx.shape
        flow_matrix = torch.full(size=(H, W, K), fill_value=-10, dtype=torch.float)
        if self.valid_range.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(range_idx+1).transpose(0, 1)
        valid_value = range_idx[valid_idx[0], valid_idx[1]].long()
        flow_matrix[valid_idx[0], valid_idx[1], :] = polar_idx[valid_value, :]
        return flow_matrix
    
    def r2p_flow_matrix(self, polar_idx, range_idx):
        """
        polar_idx: [H, W, C] indicates the location of each range pixel on point clouds
        range_idx: [N, 2] indicates the location of each points on polar grids
        """
        H, W, C = polar_idx.shape
        N, K = range_idx.shape
        flow_matrix = torch.full(size=(H, W, C, K), fill_value=-10, dtype=torch.float) # smaller than -1 to trigger the zero padding of grid_sample
        if self.valid_range.sum() == 0:
            return flow_matrix

        valid_idx = torch.nonzero(polar_idx+1).transpose(0, 1)
        valid_value = polar_idx[valid_idx[0], valid_idx[1], valid_idx[2]].long()
        flow_matrix[valid_idx[0], valid_idx[1], valid_idx[2], :] = range_idx[valid_value, ]
        return flow_matrix
    

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 5:  # with residual
            xyz, moving_labels, movable_labels,sig, residual = data
            # print("moving_labels",moving_labels.shape)
            # print("movable_labels",movable_labels.shape)
        else:
            raise Exception('Return invalid data tuple')
        
        residual_range = residual[:, 8:]
        residual = residual[:, :8]

        # 因为只变化了坐标没有变索引，所以标签不需要变
        # random data augmentation by rotation
        # 做旋转 ， 进行数据增强
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        # 做x轴或y轴或z轴镜像
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),  # [0.1, 0.1, 0.1]
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        # =====================range_res matrix========================

        fov_up = self.fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = self.fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        depth_ori = np.linalg.norm(xyz[:, :3], 2, axis=1)

        ori_x = xyz[:, 0]
        ori_y = xyz[:, 1]
        ori_z = xyz[:, 2]
        yaw_ori = -np.arctan2(ori_y, ori_x)
        pitch_ori = np.arcsin(ori_z / depth_ori)

        # proj_x_ori和proj_y_ori是所有原本点映射到range图上的坐标
        # get projections in image coords
        proj_x_ori = 0.5 * (yaw_ori / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y_ori = 1.0 - (pitch_ori + abs(fov_down)) / fov  # in [0.0, 1.0]

        # # scale to image size using angular resolution
        # proj_x_ori *= self.proj_W  # in [0.0, W]
        # proj_y_ori *= self.proj_H  # in [0.0, H]

        # # round and clamp for use as index
        # proj_x_ori = np.floor(proj_x_ori)  # 向下取整
        # proj_x_ori = np.minimum(self.proj_W - 1, proj_x_ori)
        # proj_x_ori = np.maximum(0, proj_x_ori).astype(np.int32)  # in [0,W-1]

        # proj_y_ori = np.floor(proj_y_ori)
        # proj_y_ori = np.minimum(self.proj_H - 1, proj_y_ori)
        # proj_y_ori = np.maximum(0, proj_y_ori).astype(np.int32)  # in [0,H-1]

        # get depth of all points
        # depth = np.linalg.norm(xyz[:, :3], 2, axis=1)
        # mask = (depth_ori > self.min_range) & (depth_ori < self.max_range)
        mask = (depth_ori > self.min_range) & (depth_ori < self.max_range) & (ori_z < 2) & (ori_z > -4)
        current_vertex = xyz[mask]  # get rid of [0, 0, 0] points
        depth = depth_ori[mask]

        # get scan components
        scan_x = current_vertex[:, 0]
        scan_y = current_vertex[:, 1]
        scan_z = current_vertex[:, 2]


        # scan_x = xyz[:, 0]
        # scan_y = xyz[:, 1]
        # scan_z = xyz[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # proj_x和proj_y是所有有效点映射到range图上的坐标
        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)  # 向下取整
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_ = np.copy(proj_x)  # store a copy in orig order
        self.valid_range = np.ones_like(proj_x_).astype(bool)  

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        order = np.argsort(depth)[::-1]  # 排序并提取索引，从大到小
        depth = depth[order] 
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # indices = np.arange(depth.shape[0])
        # indices = indices[order]
        indices_ori = np.arange(depth_ori.shape[0])
        indices = indices_ori[mask]
        indices = indices[order]

        residual_range= residual_range[mask]
        residual_range = residual_range[order]


        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices
        range_pixel2point = torch.from_numpy(proj_idx).clone()       #[64, 2048]  代表rv上的每个格子对应point中的哪个点（最近的那个）

        range_res_data = np.full((self.proj_H, self.proj_W, residual_range.shape[1]), 0, dtype=np.float32)
        range_res_data[proj_y, proj_x, :] = residual_range
        range_res = torch.from_numpy(range_res_data)


        unproj_n_points = proj_x_ori.shape[0]
        proj_x_range = torch.full([self.max_points], 0, dtype=torch.float)
        proj_x_range[:unproj_n_points] = torch.from_numpy(2 * (proj_x_ori - 0.5))
        proj_y_range = torch.full([self.max_points], 0, dtype=torch.float)
        proj_y_range[:unproj_n_points] = torch.from_numpy(2 * (proj_y_ori - 0.5))
        proj_yx_range = torch.stack([proj_y_range, proj_x_range], dim=1)[None, :, :]

        range_point2pixel = np.concatenate((proj_y.reshape(-1, 1), proj_x.reshape(-1, 1)), axis=1)
        range_point2pixel = torch.from_numpy(range_point2pixel).long()
        range_point2pixel = proj_yx_range[0, :unproj_n_points, :].float()         #[N, 2]

        # =====================range_res matrix==end======================
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)


        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        # 得到每一个点对应的voxel的索引[rho_idx, theta_yaw, pitch_idx]
        # Clip (limit) the values in an array.
        # np.floor向下取整
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        pxpypz = (np.clip(xyz_pol,min_bound,max_bound)-min_bound) / crop_range
        pxpypz = 2 * (pxpypz- 0.5)
        polar_point2pixel = torch.from_numpy(pxpypz).float()     #[N, 3]

        xyz_pol_mask = xyz_pol[mask]

        # order in decreasing z
        pixel2point_pol = np.full(shape=self.grid_size, fill_value=-1, dtype=np.float32)
        order_pol = np.argsort(xyz_pol_mask[:, 2])
        indices_pol_ori = np.arange(xyz_pol.shape[0])
        indices_pol = indices_pol_ori[mask]
        grid_ind_order_pol = (np.floor((np.clip(xyz_pol_mask, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
        grid_ind_order_pol = grid_ind_order_pol[order_pol]
        indices_pol = indices_pol[order_pol]
        pixel2point_pol[grid_ind_order_pol[:, 0], grid_ind_order_pol[:, 1], grid_ind_order_pol[:, 2]] = indices_pol
        polar_pixel2point = torch.from_numpy(pixel2point_pol)  # [480, 360 , 32]

        r2p_flow_matrix = self.r2p_flow_matrix(polar_pixel2point, range_point2pixel)    #[480,360,32,2]
        p2r_flow_matrix = self.p2r_flow_matrix(range_pixel2point, polar_point2pixel)    #[674,2048,3]
        # process labels
        # self.ignore_label = 255
        processed_moving_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        moving_label_voxel_pair = np.concatenate([grid_ind, moving_labels], axis=1)  # 每一个点对应的格子和label
        moving_label_voxel_pair = moving_label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # 按照pitch yaw rho顺序从小到大排序
        processed_moving_label = nb_process_label(np.copy(processed_moving_label), moving_label_voxel_pair)

        if self.point_cloud_dataset.movable:
            # process movable_labels    
            processed_movable_labels = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            movable_labels_voxel_pair = np.concatenate([grid_ind, movable_labels], axis=1)  # 每一个点对应的格子和label
            movable_labels_voxel_pair = movable_labels_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # 按照pitch yaw rho顺序从小到大排序
            processed_movable_labels = nb_process_label(np.copy(processed_movable_labels), movable_labels_voxel_pair)

        # center data on each voxel for PTnet
        # 这里是每一个点所处的voxel中心的位置，在同一个voxel的点的位置是一样的
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        # 每一个点的位置相对于其voxel中心的偏移量
        return_xyz = xyz_pol - voxel_centers
        # [bias_rho, bias_yaw, bias_pitch, rho, yaw, pitch, x, y]
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 5:  # reflectivity residual
            # [bias_rho, bias_theta, bias_z, rho, theta, z, x, y, reflectivity, residual(1-?)]
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis], residual), axis=1)
        else:
            raise NotImplementedError

        if self.point_cloud_dataset.movable:
            if self.return_test:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                        torch.from_numpy(processed_movable_labels).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor), \
                   r2p_flow_matrix, p2r_flow_matrix, range_res, index
            else:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                torch.from_numpy(processed_movable_labels).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor),\
                   r2p_flow_matrix, p2r_flow_matrix, range_res
        else:
            if self.return_test:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor), \
                   r2p_flow_matrix, p2r_flow_matrix, range_res, index
            else:
                return torch.from_numpy(processed_moving_label).type(torch.LongTensor), \
                    torch.from_numpy(grid_ind), \
                    moving_labels, \
                    movable_labels, \
                    torch.from_numpy(return_fea).type(torch.FloatTensor),\
                   r2p_flow_matrix, p2r_flow_matrix, range_res



@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):  # 每个栅格赋予出现次数最多的label
    label_size = 2
    counter = np.zeros((label_size,), dtype=np.uint16)  # counter计算每个label的数量
    counter[sorted_label_voxel_pair[0, 3]] = 1  # 第一个初始化，先加一
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 目标点的栅格坐标索引
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]  # 当前点的栅格坐标索引
        if not np.all(np.equal(cur_ind, cur_sear_ind)):  # 索引不一致，要移动到下一个栅格
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)  # 栅格使用出现次数最多的label
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1  # label计数
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV_MF(data):
    # print(len(data))
    moving_label = torch.stack([d[0] for d in data])
    movable_label =  torch.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label_moving = [d[3] for d in data]
    point_label_movable = [d[4] for d in data]
    xyz = [d[5] for d in data]
    r2p_flow_matrix = [d[6] for d in data]
    p2r_flow_matrix = [d[7] for d in data]
    range_res_data = [d[8] for d in data]
    return moving_label, movable_label, grid_ind_stack, point_label_moving,point_label_movable,xyz, r2p_flow_matrix, p2r_flow_matrix, range_res_data

def collate_fn_BEV_MF_test(data):
    # print(len(data))
    moving_label = torch.stack([d[0] for d in data])
    movable_label =  torch.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label_moving = [d[3] for d in data]
    point_label_movable = [d[4] for d in data]
    xyz = [d[5] for d in data]
    r2p_flow_matrix = [d[6] for d in data]
    p2r_flow_matrix = [d[7] for d in data]
    range_res_data = [d[8] for d in data]
    index = [d[9] for d in data]
    return moving_label, movable_label, grid_ind_stack, point_label_moving,point_label_movable,xyz, r2p_flow_matrix, p2r_flow_matrix, range_res_data,index

# load Semantic KITTI class info
def get_SemKITTI_label_name_MF(label_mapping):  #
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_moving_label_name = dict()
    SemKITTI_movable_label_name = dict()

    moving_inv_learning_map = semkittiyaml['moving_learning_map_inv']
    movable_inv_learning_map = semkittiyaml['movable_learning_map_inv']

    for i in sorted(list(semkittiyaml['moving_learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['moving_learning_map'][i]
        map_inv_i = semkittiyaml['moving_learning_map_inv'][map_i]
        SemKITTI_moving_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    for i in sorted(list(semkittiyaml['movable_learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['movable_learning_map'][i]
        map_inv_i = semkittiyaml['movable_learning_map_inv'][map_i]
        SemKITTI_movable_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    moving_label = np.asarray(sorted(list(SemKITTI_moving_label_name.keys())))[:]
    moving_label_str = [SemKITTI_moving_label_name[x] for x in moving_label]

    movable_label = np.asarray(sorted(list(SemKITTI_movable_label_name.keys())))[:]
    movable_label_str = [SemKITTI_movable_label_name[x] for x in movable_label]

    # print("moving_label",moving_label)
    print("moving_label_str",moving_label_str)
    # print("movable_label",movable_label)
    print("movable_label_str",movable_label_str)
    return moving_label, moving_label_str, moving_inv_learning_map,\
            movable_label,movable_label_str,movable_inv_learning_map