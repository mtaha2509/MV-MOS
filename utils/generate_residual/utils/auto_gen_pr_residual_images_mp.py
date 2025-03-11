#!/usr/bin/env python3
# Developed by cxm and jxLiang
# Brief: This script generates residual images

from multiprocessing import Process
import os
import random

os.environ["OMP_NUM_THREADS"] = "16"
from matplotlib import pyplot as plt
import yaml
import numpy as np

from tqdm import tqdm
from icecream import ic
from kitti_utils import load_poses, load_calib, load_files, load_vertex , polar_projection , range_projection , load_yaml , check_and_makedirs, range_projection2, range_projection3
from queue import Queue


def gen_pr_residual_images(cfg,seq):
    scan_folder = cfg['scan_folder']
    pose_file = os.path.join(scan_folder,f"sequences/{'%02d' % seq}/poses.txt")
    calib_file = os.path.join(scan_folder, f"sequences/{'%02d' % seq}/calib.txt")
    scan_folder = os.path.join(scan_folder,f"sequences/{'%02d' % seq}/velodyne")

    residual_image_folder = cfg ['residual_image_folder']
    residual_image_folder = os.path.join(residual_image_folder,f"{'%02d' % seq}/residual_images")
    check_and_makedirs(residual_image_folder)

    rv_visualization_folder = cfg['rv_visualization_folder']
    rv_visualization_folder = os.path.join(rv_visualization_folder,f"{'%02d' % seq}/visualization")

    polar_cfg = cfg['polar']
    rv_cfg = cfg['rv']
    if rv_cfg['visualize']:
        check_and_makedirs(rv_visualization_folder)

    # load poses
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_paths = load_files(scan_folder)

    # for all
    num_frames = cfg['num_frames']
    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses) , "  in seq:",seq)
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    ### polar param ###
    polar_num_prev_n = polar_cfg['num_prev_n']
    polar_num_last_n = polar_cfg['num_last_n']
    polar_occlusion_block = polar_cfg['occlusion_block']
    polar_image_params = polar_cfg['polar_image']
    ### polar param ###

    ### rv param ###
    rv_debug = rv_cfg['debug']
    rv_normalize = rv_cfg['normalize']
    rv_num_last_n = rv_cfg['num_last_n']
    rv_visualize = rv_cfg['visualize']
    range_image_params = rv_cfg['range_image']
    ### rv param ###

    # polar
    prev_len_Que = Queue()
    last_len_Que = Queue()
    prev_sub_map = None
    last_sub_map = None
    prev_diff_map = None
    last_diff_map = None
    prev_index = None
    last_index = None

    polar_residual_images_array = []
    polar_save_file_name_array = []
    rv_residual_images_array = []
    rv_save_file_name_array = []
    # generate residual images for the whole sequence
    for frame_idx in tqdm(range(len(scan_paths)),desc=f"{seq}"):
        
        current_scan = load_vertex(scan_paths[frame_idx])


        ####### rv start ####### 
        # print("rv start")
        rv_diff_image = np.full((range_image_params['height'], range_image_params['width']), 0, dtype=np.float32)  # [H,W] range (0 is no data)
        rv_diff_scan_array = np.full((current_scan.shape[0], rv_num_last_n), 0, dtype=np.float32)
        # for the first N frame we generate a dummy file
        for rv_last_n in range(1, rv_num_last_n+1):
            rv_diff_scan = np.full(current_scan.shape[0], 0, dtype=np.float32)
            if frame_idx < rv_last_n:
                rv_diff_scan_array[:, rv_last_n-1] = rv_diff_scan    #前八帧残差信息为0
            else:
                # load current scan and generate current range image
                current_pose = poses[frame_idx]

                rv_current_idx, rv_current_range = range_projection3(current_scan.astype(np.float32),
                                                 range_image_params['height'], range_image_params['width'],
                                                 range_image_params['fov_up'], range_image_params['fov_down'],
                                                 range_image_params['max_range'], range_image_params['min_range'])

                # load last scan, transform into the current coord and generate a transformed last range image
                last_pose = poses[frame_idx - rv_last_n]      #处理当前帧的前i帧，i为1-8
                last_scan = load_vertex(scan_paths[frame_idx - rv_last_n])  # (x, y, z, 1)
                last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
                rv_last_idx_transformed, rv_last_range_transformed = range_projection3(last_scan_transformed.astype(np.float32),
                                                          range_image_params['height'], range_image_params['width'],
                                                          range_image_params['fov_up'], range_image_params['fov_down'],
                                                          range_image_params['max_range'], range_image_params['min_range'])

                # generate residual image
                valid_mask = (rv_current_range > range_image_params['min_range']) & \
                             (rv_current_range < range_image_params['max_range']) & \
                             (rv_last_range_transformed > range_image_params['min_range']) & \
                             (rv_last_range_transformed < range_image_params['max_range'])
                difference = np.abs(rv_current_range[valid_mask] - rv_last_range_transformed[valid_mask])

                if rv_normalize:
                    difference = np.abs(rv_current_range[valid_mask] - rv_last_range_transformed[valid_mask]) / rv_current_range[valid_mask]

                rv_diff_image[valid_mask] = difference
                # 反投影回三维空间
                rv_diff_scan[rv_current_idx] = rv_diff_image

                if rv_debug:
                    fig, axs = plt.subplots(3)
                    axs[0].imshow(rv_last_range_transformed)
                    axs[1].imshow(rv_current_range)
                    axs[2].imshow(rv_diff_image, vmin=0, vmax=1)
                    plt.show()

                if rv_visualize:
                    fig = plt.figure(frameon=False, figsize=(16, 10))
                    fig.set_size_inches(20.48, 0.64)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(rv_diff_image, vmin=0, vmax=1)
                    image_name = os.path.join(rv_visualization_folder, str(rv_last_n))
                    check_and_makedirs(image_name)
                    image_name = os.path.join(image_name,str(frame_idx).zfill(6))
                    plt.savefig(image_name)
                    plt.close()
                rv_diff_scan_array[:, rv_last_n - 1] = rv_diff_scan   # 处理的前i帧则放在第i通道
        # print("rv end")
        ####### rv end ####### 
        # rv save residual image
        # rv_save_file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))  
        # print("rv",rv_save_file_name)      
        # rv_save_file_name_array.append(rv_save_file_name) 
        rv_residual_images_array.append(rv_diff_scan_array)   #将一个序列中所有帧存到一个数组中
        # np.save(rv_save_file_name, rv_diff_scan_array)


        ####### polar start #######
        # print("polar start")
        if frame_idx < polar_num_prev_n + polar_num_last_n - 1:
            continue
        else:
            # load current scan
            current_scan = load_vertex(scan_paths[frame_idx])
            # current_scan = current_scan[:random.randint(100, 200)]  # down sample for debug
            current_pose = poses[frame_idx]
            if frame_idx == polar_num_prev_n + polar_num_last_n - 1:  # initialize                   idx = 7时，初始化设置
                for i in range(-polar_num_last_n - polar_num_prev_n + 1, -polar_num_last_n + 1):   #处理-7到-4帧
                    prev_pose = poses[frame_idx + i]
                    prev_scan = load_vertex(scan_paths[frame_idx + i])
                    # prev_scan = prev_scan[:random.randint(100, 200)]  # down sample for debug
                    prev_scan_transformed = np.linalg.inv(current_pose).dot(prev_pose).dot(prev_scan.T).T
                    if prev_sub_map is None:
                        prev_sub_map = prev_scan_transformed
                        prev_diff_map = np.full((prev_scan.shape[0], polar_num_prev_n + polar_num_last_n), 0, dtype=np.float32)
                        prev_index = np.full(prev_scan.shape[0], polar_num_last_n, dtype=int)
                    else:
                        prev_sub_map = np.concatenate((prev_sub_map, prev_scan_transformed), axis=0)
                        prev_diff_map = np.concatenate(
                            (prev_diff_map, np.full((prev_scan.shape[0], polar_num_prev_n + polar_num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        prev_index = prev_index + 1
                        prev_index = np.concatenate((prev_index, np.full(prev_scan.shape[0], polar_num_last_n, dtype=int)),    # prev_idx按照4，5，6，7排序，代表后续存放通道
                                                    axis=0)
                    prev_len_Que.put(prev_scan.shape[0])
                for i in range(-polar_num_last_n + 1, 1):    #处理-3到0帧
                    last_pose = poses[frame_idx + i]
                    last_scan = load_vertex(scan_paths[frame_idx + i])  # (x, y, z, 1)
                    # last_scan = last_scan[:random.randint(100, 200)]  # down sample for debug
                    last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
                    if last_sub_map is None:
                        last_sub_map = last_scan_transformed
                        last_diff_map = np.full((last_scan.shape[0], polar_num_prev_n + polar_num_last_n), 0, dtype=np.float32)
                        last_index = np.full(last_scan.shape[0], 0, dtype=int)
                    else:
                        last_sub_map = np.concatenate((last_sub_map, last_scan_transformed), axis=0)
                        last_diff_map = np.concatenate(
                            (last_diff_map, np.full((last_scan.shape[0], polar_num_prev_n + polar_num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        last_index = last_index + 1
                        last_index = np.concatenate((last_index, np.full(last_scan.shape[0], 0, dtype=int)), axis=0)     # last_idx按照0，1，2，3排序，代表后续存放通道
                    last_len_Que.put(last_scan.shape[0])
            else:
                last_pose = poses[frame_idx - 1]
                # transform to current coordinate
                prev_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(prev_sub_map.T).T
                last_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(last_sub_map.T).T
                last_sub_map = np.concatenate((last_sub_map, current_scan), axis=0)
                last_diff_map = np.concatenate(
                    (last_diff_map, np.full((current_scan.shape[0], polar_num_prev_n + polar_num_last_n), 0,
                                            dtype=np.float32)), axis=0)
                last_len_Que.put(current_scan.shape[0])
                prev_index = (prev_index + 1)  # % num_prev_n + num_last_n
                last_index = (last_index + 1)  # % num_prev_n + num_last_n
                last_index = np.concatenate((last_index, np.full(current_scan.shape[0], 0, dtype=int)), axis=0)

            # print(np.max(prev_index), np.max(last_index), np.min(prev_index), np.min(last_index))
            prev_proj_z_delta, prev_mask_valid, prev_proj_y, prev_proj_x, prev_occlusion_mask = \
                polar_projection(current_vertex=prev_sub_map.astype(np.float32),
                                 proj_H=polar_image_params['height'],
                                 proj_W=polar_image_params['width'],
                                 max_range=polar_image_params['max_range'],
                                 min_range=polar_image_params['min_range'],
                                 max_z=polar_image_params['max_z'],
                                 min_z=polar_image_params['min_z'],
                                 return_occlusion=polar_occlusion_block)

            last_proj_z_delta, last_mask_valid, last_proj_y, last_proj_x, last_occlusion_mask = \
                polar_projection(current_vertex=last_sub_map.astype(np.float32),
                                 proj_H=polar_image_params['height'],
                                 proj_W=polar_image_params['width'],
                                 max_range=polar_image_params['max_range'],
                                 min_range=polar_image_params['min_range'],
                                 max_z=polar_image_params['max_z'],
                                 min_z=polar_image_params['min_z'],
                                 return_occlusion=polar_occlusion_block)

            # generate residual image
            residual_proj_z_delta = prev_proj_z_delta - last_proj_z_delta
            residual_proj_z_delta[0.2 * prev_proj_z_delta < last_proj_z_delta] = 0
            residual_proj_z_delta[prev_proj_z_delta < 0.4] = 0
            residual_proj_z_delta[prev_proj_z_delta > 4] = 0
            if polar_occlusion_block:
                residual_proj_z_delta[last_occlusion_mask] = 0  # residual_proj_z_delta[last_proj_z_delta == 0] = 0
            prev_diff_map[prev_mask_valid, prev_index[prev_mask_valid]] += residual_proj_z_delta[
                prev_proj_y, prev_proj_x]

            residual_proj_z_delta = last_proj_z_delta - prev_proj_z_delta
            residual_proj_z_delta[0.2 * last_proj_z_delta < prev_proj_z_delta] = 0
            residual_proj_z_delta[last_proj_z_delta < 0.4] = 0
            residual_proj_z_delta[last_proj_z_delta > 4] = 0
            if polar_occlusion_block:
                residual_proj_z_delta[prev_occlusion_mask] = 0  # residual_proj_z_delta[prev_proj_z_delta == 0] = 0
            last_diff_map[last_mask_valid, last_index[last_mask_valid]] += residual_proj_z_delta[
                last_proj_y, last_proj_x]

            prev_scan_size = prev_len_Que.get()   # 取出8帧中的第1帧个数
            prev_sub_map = prev_sub_map[prev_scan_size:]   #去除第一帧
            diff_scan = prev_diff_map[:prev_scan_size]           # prev_diff_map为[N,8] N为prev的四帧点数和
            prev_diff_map = prev_diff_map[prev_scan_size:]
            prev_index = prev_index[prev_scan_size:]

            last_scan_size = last_len_Que.get()
            prev_len_Que.put(last_scan_size)
            prev_sub_map = np.concatenate((prev_sub_map, last_sub_map[:last_scan_size]), axis=0)
            last_sub_map = last_sub_map[last_scan_size:]
            prev_diff_map = np.concatenate((prev_diff_map, last_diff_map[:last_scan_size]), axis=0)
            last_diff_map = last_diff_map[last_scan_size:]
            prev_index = np.concatenate((prev_index, last_index[:last_scan_size]), axis=0)
            last_index = last_index[last_scan_size:]
        # print("polar end")
        ####### polar end #######
            
            # # save
            polar_file_name = os.path.join(residual_image_folder, str(frame_idx - polar_num_last_n - polar_num_prev_n + 1).zfill(6))
            # print("frame_idx",frame_idx)
            # print("polar",polar_file_name)
            # polar_save_file_name_array.append(polar_file_name)
            ##### cat to save 
            concatenated_data = np.concatenate((diff_scan, rv_residual_images_array.pop(0)), axis=1)
            # print(concatenated_data.shape)
            np.save(polar_file_name, concatenated_data)

    print("Saving last few files...")
    for frame_idx in range(len(scan_paths) - polar_num_last_n - polar_num_prev_n + 1, len(scan_paths)):
        if not prev_len_Que.empty():
            prev_scan_size = prev_len_Que.get()
            diff_scan = prev_diff_map[:prev_scan_size]
            prev_diff_map = prev_diff_map[prev_scan_size:]
        else:
            prev_scan_size = last_len_Que.get()
            diff_scan = last_diff_map[:prev_scan_size]
            last_diff_map = last_diff_map[prev_scan_size:]

        polar_file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
        # print(polar_file_name)
        # polar_save_file_name_array.append(polar_file_name)
        ##### cat to save 
        concatenated_data = np.concatenate((diff_scan, rv_residual_images_array.pop(0)), axis=1)
        # print(concatenated_data.shape)
        np.save(polar_file_name, concatenated_data)
    assert(len(rv_residual_images_array) == 0)
    print(f"seq : {seq} has been Done.")

if __name__ == '__main__':
    # load config file
    config_filename = './utils/generate_residual/utils/pr_residual_images.yaml'
    config = load_yaml(config_filename)
    processes = []

    # # for SemanticKITTI
    for seq in range(0, 5):  # sequences id
        process = Process(target=gen_pr_residual_images, args=(config, seq))  # 指定进程要执行的函数和参数
        processes.append(process)
        process.start()  # 启动进程

    for process in processes:
        process.join()  # 等待进程完成

    # # for SemanticKITTI-road
    # for seq in range(30, 42):  # sequences id    
    #     process = Process(target=gen_pr_residual_images, args=(config, seq))  # 指定进程要执行的函数和参数
    #     processes.append(process)
    #     process.start()  # 启动进程

    # for process in processes:
    #     process.join()  # 等待进程完成
        
    # # for SemanticKITTI_test
    # scan_folder = config['scan_folder']
    # config['scan_folder'] = os.path.join(scan_folder , "test")
    # for seq in range(11, 22):  # sequences id    
    #     process = Process(target=gen_pr_residual_images, args=(config, seq))  # 指定进程要执行的函数和参数
    #     processes.append(process)
    #     process.start()  # 启动进程 

    # for process in processes:
    #     process.join()  # 等待进程完成