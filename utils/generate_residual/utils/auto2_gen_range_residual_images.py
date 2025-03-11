#!/usr/bin/env python3
# Developed by Jiadai Sun
# 	and the main_funciton 'prosess_one_seq' refers to Xieyuanli Chen’s gen_residual_images.py
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images

import os

os.environ["OMP_NUM_THREADS"] = "4"
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from icecream import ic
from kitti_utils import load_poses, load_calib, load_files, load_vertex
from queue import Queue

try:
    from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
    print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
    print("Currently using python-lib to generate range images.")
    from kitti_utils import range_projection2


def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_yaml(path):
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(path))
    return config


def process_one_seq(config):
    # specify parameters
    num_frames = config['num_frames']
    debug = config['debug']
    normalize = config['normalize']
   # num_last_n = config['num_last_n']
    visualize = config['visualize']
    visualization_folder = config['visualization_folder']

    num_prev_n = 4
    num_last_n = 4
    # specify the output folders
    residual_image_folder = config['residual_image_folder']
    check_and_makedirs(residual_image_folder)

    if visualize:
        check_and_makedirs(visualization_folder)

    # load poses
    pose_file = config['pose_file']
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    calib_file = config['calib_file']
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_folder = config['scan_folder']
    scan_paths = load_files(scan_folder)

    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    range_image_params = config['range_image']

    prev_len_Que = Queue()
    last_len_Que = Queue()
    prev_sub_map = None
    last_sub_map = None
    prev_diff_map = None
    last_diff_map = None
    prev_index = None
    last_index = None

    # generate residual images for the whole sequence
    for frame_idx in tqdm(range(len(scan_paths))):
        # file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
        diff_image = np.full((range_image_params['height'], range_image_params['width']), 0,
                             dtype=np.float32)  # [H,W] range (0 is no data)
        diff_image_last = np.full((range_image_params['height'], range_image_params['width']), 0,
                             dtype=np.float32)  # [H,W] range (0 is no data)
        # current_scan = load_vertex(scan_paths[frame_idx])
        # diff_scan_array = np.full((current_scan.shape[0], num_last_n), 0, dtype=np.float32)
        if frame_idx < num_prev_n + num_last_n - 1:
            continue
        else:
            # load current scan
            current_scan = load_vertex(scan_paths[frame_idx])
            # current_scan = current_scan[:random.randint(100, 200)]  # down sample for debug
            current_pose = poses[frame_idx]
            if frame_idx == num_prev_n + num_last_n - 1:  # initialize
                for i in range(-num_last_n - num_prev_n + 1, -num_last_n + 1):
                    prev_pose = poses[frame_idx + i]
                    prev_scan = load_vertex(scan_paths[frame_idx + i])
                    # prev_scan = prev_scan[:random.randint(100, 200)]  # down sample for debug
                    prev_scan_transformed = np.linalg.inv(current_pose).dot(prev_pose).dot(prev_scan.T).T
                    if prev_sub_map is None:
                        prev_sub_map = prev_scan_transformed
                        prev_diff_map = np.full((prev_scan.shape[0], num_prev_n + num_last_n), 0, dtype=np.float32)
                        prev_diffscan_map = np.full(prev_scan.shape[0], 0, dtype=np.float32)
                        prev_index = np.full(prev_scan.shape[0], num_last_n, dtype=int)
                    else:
                        prev_sub_map = np.concatenate((prev_sub_map, prev_scan_transformed), axis=0)
                        prev_diff_map = np.concatenate(
                            (prev_diff_map, np.full((prev_scan.shape[0], num_prev_n + num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        prev_diffscan_map = np.concatenate(
                            (prev_diffscan_map, np.full(prev_scan.shape[0], 0,
                                                    dtype=np.float32)), axis=0)
                        prev_index = prev_index + 1
                        prev_index = np.concatenate((prev_index, np.full(prev_scan.shape[0], num_last_n, dtype=int)),
                                                    axis=0)
                    prev_len_Que.put(prev_scan.shape[0])
                for i in range(-num_last_n + 1, 1):
                    last_pose = poses[frame_idx + i]
                    last_scan = load_vertex(scan_paths[frame_idx + i])  # (x, y, z, 1)
                    # last_scan = last_scan[:random.randint(100, 200)]  # down sample for debug
                    last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
                    if last_sub_map is None:
                        last_sub_map = last_scan_transformed
                        last_diff_map = np.full((last_scan.shape[0], num_prev_n + num_last_n), 0, dtype=np.float32)
                        last_diffscan_map = np.full(last_scan.shape[0], 0, dtype=np.float32)
                        last_index = np.full(last_scan.shape[0], 0, dtype=int)
                    else:
                        last_sub_map = np.concatenate((last_sub_map, last_scan_transformed), axis=0)
                        last_diff_map = np.concatenate(
                            (last_diff_map, np.full((last_scan.shape[0], num_prev_n + num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        last_diffscan_map = np.concatenate(
                            (last_diffscan_map, np.full(last_scan.shape[0], 0,
                                                    dtype=np.float32)), axis=0)
                        last_index = last_index + 1
                        last_index = np.concatenate((last_index, np.full(last_scan.shape[0], 0, dtype=int)), axis=0)
                    last_len_Que.put(last_scan.shape[0])
            else:
                last_pose = poses[frame_idx - 1]
                # transform to current coordinate
                prev_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(prev_sub_map.T).T
                last_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(last_sub_map.T).T
                last_sub_map = np.concatenate((last_sub_map, current_scan), axis=0)
                last_diff_map = np.concatenate(
                    (last_diff_map, np.full((current_scan.shape[0], num_prev_n + num_last_n), 0,
                                            dtype=np.float32)), axis=0)
                last_len_Que.put(current_scan.shape[0])
                prev_index = (prev_index + 1)  # % num_prev_n + num_last_n
                last_index = (last_index + 1)  # % num_prev_n + num_last_n
                last_index = np.concatenate((last_index, np.full(current_scan.shape[0], 0, dtype=int)), axis=0)

            current_mask, current_range, current_proj_y, current_proj_x,current_idx = range_projection2(prev_sub_map.astype(np.float32),
                                                range_image_params['height'], range_image_params['width'],
                                                range_image_params['fov_up'], range_image_params['fov_down'],
                                                range_image_params['max_range'], range_image_params['min_range'])
            
            last_mask, last_range_transformed, last_proj_y, last_proj_x,last_idx = range_projection2(last_sub_map.astype(np.float32),
                                                        range_image_params['height'], range_image_params['width'],
                                                        range_image_params['fov_up'], range_image_params['fov_down'],
                                                        range_image_params['max_range'], range_image_params['min_range'])

            valid_mask = (current_range > range_image_params['min_range']) & \
                            (current_range < range_image_params['max_range']) & \
                            (last_range_transformed > range_image_params['min_range']) & \
                            (last_range_transformed < range_image_params['max_range'])
            difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])
            if normalize:
                difference_current = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[
                    valid_mask]
                difference_last = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / last_range_transformed[
                    valid_mask]
                
            diff_image[valid_mask] = difference_current
            diff_image_last[valid_mask] = difference_last
            # #反投影回三维空间
            # prev_diffscan_map[current_idx] = diff_image
            # last_diffscan_map[last_idx] = diff_image_last

            if visualize:
                fig = plt.figure(frameon=False, figsize=(16, 10))
                fig.set_size_inches(20.48, 0.64)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(diff_image, vmin=0, vmax=1)
                image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6))
                plt.savefig(image_name)
                plt.close()
            
            prev_diff_map[current_mask,prev_index[current_mask]] += diff_image[current_proj_y, current_proj_x]
            last_diff_map[last_mask, last_index[last_mask]] += diff_image_last[last_proj_y, last_proj_x]
            # prev_diff_map[current_mask,prev_index[current_mask]] += prev_diffscan_map[current_mask]
            # last_diff_map[last_mask, last_index[last_mask]] += last_diffscan_map[last_mask]
            # prev_diff_map[:,prev_index] += prev_diffscan_map
            # last_diff_map[:, last_index] += last_diffscan_map

            prev_scan_size = prev_len_Que.get()
            prev_sub_map = prev_sub_map[prev_scan_size:]
            diff_scan = prev_diff_map[:prev_scan_size]
            prev_diff_map = prev_diff_map[prev_scan_size:]
            prev_diffscan_map = prev_diffscan_map[prev_scan_size:]
            prev_index = prev_index[prev_scan_size:]

            last_scan_size = last_len_Que.get()
            prev_len_Que.put(last_scan_size)
            prev_sub_map = np.concatenate((prev_sub_map, last_sub_map[:last_scan_size]), axis=0)
            last_sub_map = last_sub_map[last_scan_size:]
            prev_diff_map = np.concatenate((prev_diff_map, last_diff_map[:last_scan_size]), axis=0)
            last_diff_map = last_diff_map[last_scan_size:]
            prev_diffscan_map = np.concatenate((prev_diffscan_map, last_diffscan_map[:last_scan_size]), axis=0)
            last_diffscan_map = last_diffscan_map[last_scan_size:]
            prev_index = np.concatenate((prev_index, last_index[:last_scan_size]), axis=0)
            last_index = last_index[last_scan_size:]

            # save
            file_name = os.path.join(residual_image_folder, str(frame_idx - num_last_n - num_prev_n + 1).zfill(6))
            np.save(file_name, diff_scan)
            

    print("Saving last few files...")
    for frame_idx in range(len(scan_paths) - num_last_n - num_prev_n + 1, len(scan_paths)):
        if not prev_len_Que.empty():
            prev_scan_size = prev_len_Que.get()
            diff_scan = prev_diff_map[:prev_scan_size]
            prev_diff_map = prev_diff_map[prev_scan_size:]
        else:
            prev_scan_size = last_len_Que.get()
            diff_scan = last_diff_map[:prev_scan_size]
            last_diff_map = last_diff_map[prev_scan_size:]

        file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
        np.save(file_name, diff_scan)

    print("Done.")

                # if debug:
                #     fig, axs = plt.subplots(3)
                #     axs[0].imshow(last_range_transformed)
                #     axs[1].imshow(current_range)
                #     axs[2].imshow(diff_image, vmin=0, vmax=1)
                #     plt.show()

                # if visualize:
                #     fig = plt.figure(frameon=False, figsize=(16, 10))
                #     fig.set_size_inches(20.48, 0.64)
                #     ax = plt.Axes(fig, [0., 0., 1., 1.])
                #     ax.set_axis_off()
                #     fig.add_axes(ax)
                #     ax.imshow(diff_image, vmin=0, vmax=1)
                #     image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6))
                #     plt.savefig(image_name)
                #     plt.close()
                # diff_scan_array[:, last_n - 1] = diff_scan
        # save residual image
        # np.save(file_name, diff_scan_array)


if __name__ == '__main__':

    # load config file
    config_filename ='/data-ssd/data5/cxm/MOS/BEV-MF-PR/utils/generate_residual/config/data_preparing_range.yaml'# '../config/data_preparing_range.yaml'
    config = load_yaml(config_filename)

    scan_folder = config['scan_folder']
    pose_file = config['pose_file']
    calib_file = config['calib_file']
    residual_image_folder = config['residual_image_folder']
    visualization_folder = config['visualization_folder']
    num_last_n = config['num_last_n']

    # used for kitti-raw and kitti-road
    for seq in range(4,5):  # sequences id
        # Update the value in config to facilitate the iterative loop
        config['scan_folder'] = scan_folder + f"sequences/{'%02d' % seq}/velodyne"
        config['pose_file'] = pose_file + f"sequences/{'%02d' % seq}/poses.txt"
        config['calib_file'] = calib_file + f"sequences/{'%02d' % seq}/calib.txt"
        config['residual_image_folder'] = residual_image_folder + f"{'%02d' % seq}/residual_images_{num_last_n}"
        config['visualization_folder'] = visualization_folder + f"{'%02d' % seq}/visualization__{num_last_n}"
        ic(config)
        process_one_seq(config)
