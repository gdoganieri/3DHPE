import cv2
import math
import time
import os
import os.path as osp
import numpy as np
import torch
import argparse
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from thefuzz import process

posenet_path = os.getcwd() + "/posenet"

from posenet.main.config import cfg as posenet_cfg
from posenet.main.model import get_pose_net
from posenet.data.dataset import generate_patch_image
from posenet.common.posenet_utils.pose_utils import process_bbox, pixel2cam

rootnet_path = os.getcwd() + "/rootnet"

from rootnet.main.config import cfg as rootnet_cfg
from rootnet.main.model import get_pose_net as get_root_net
from rootnet.common.utils.pose_utils import process_bbox as rootnet_process_bbox
from rootnet.data.dataset import generate_patch_image as rootnet_generate_patch_image

import torchvision

from pathlib import Path
from posenet.common.posenet_utils.vis import vis_keypoints
from d_visualization import depthmap2pointcloud

def main():
    # FASTER RCNN v3 320 fpn
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detector_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,
                                                                                        pretrained_backbone=True)
    detector_model.eval().to(device)

    detector_score_threshold = 0.8
    detector_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, dest='gpu_ids')
        parser.add_argument('--weights', type=str, dest='weights')
        parser.add_argument('--source', type=str, dest='source')
        parser.add_argument('--sequence', type=str, dest='sequence')
        parser.add_argument('--bboxdiff', type=bool, dest='bboxdiff')
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            assert 0, print("Please set proper gpu ids")

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        assert args.weights, 'Pretrained weights are required.'
        assert args.source, 'Source is required.'
        assert args.sequence, 'Sequence is required.'
        return args

    # argument parsing
    args = parse_args()

    posenet_cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # Check the weights to load to find how to build the skeleton
    weights = args.weights
    if weights == "MuCo":
        # MuCo joint set
        joint_num = 21
        joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                       'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand',
                       'L_Hand', 'R_Toe', 'L_Toe')
        flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
        skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12),
                    (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
        model_path_posenet = 'snapshot_24_MuCo+MSCOCO.pth.tar'
        model_path_rootnet = 'snapshot_18_MuCo+MSCOCO.pth.tar'
        bbox_real = rootnet_cfg.bbox_real_MuCo

    elif weights == "H36M":
        # Human36 joint set
        joint_num = 18
        joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
                       'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                     (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        model_path_posenet = 'snapshot_24_H36M+MPII.pth.tar'
        model_path_rootnet = 'snapshot_19_H36M+MPII.pth.tar'
        bbox_real = rootnet_cfg.bbox_real_Human36M
    else:
        assert 'Pretrained weights are required.'
        return -1

    # snapshot load posenet
    assert osp.exists(model_path_posenet), 'Cannot find model at ' + model_path_posenet
    print('Load checkpoint from {}'.format(model_path_posenet))
    model = get_pose_net(posenet_cfg, False, joint_num)
    model = DataParallel(model).cpu()
    ckpt = torch.load(model_path_posenet, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['network'])
    model.eval()

    # snapshot load rootnet
    assert osp.exists(model_path_rootnet), 'Cannot find model at ' + model_path_rootnet
    print('Load checkpoint from {}'.format(model_path_rootnet))
    rootnet_model = get_root_net(rootnet_cfg, False)
    rootnet_model = DataParallel(rootnet_model).cpu()
    ckpt_rootnet = torch.load(model_path_rootnet, map_location=torch.device('cpu'))
    rootnet_model.load_state_dict(ckpt_rootnet['network'])
    rootnet_model.eval()

    rootnet_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=rootnet_cfg.pixel_mean, std=rootnet_cfg.pixel_std)])
    posenet_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=posenet_cfg.pixel_mean, std=posenet_cfg.pixel_std)])

    sequence = args.sequence
    source = args.source
    bboxdiff = args.bboxdiff

    if source == "pico":
        data_dir = Path("data/pico/")
        rgb_data_dir = data_dir / "rgb" / sequence
        depth_data_dir = data_dir / "depth" / sequence
    elif source == "kinect":
        data_dir = Path("data/kinect/")
        rgb_data_dir = data_dir / sequence / "RGB"
        depth_data_dir = data_dir / sequence / "DEPTH"
    else:
        data_dir = Path("frames/")
        rgb_data_dir = data_dir
        depth_data_dir = None



    # rgb camera intrinsics
    intrinsics_rgb = np.loadtxt(str(data_dir/"intrinsics_rgb.txt"), dtype='f', delimiter=',')
    focal_rgb = [intrinsics_rgb[0][0],intrinsics_rgb[1][1]] # x-axis, y-axis
    princpt_rgb = [intrinsics_rgb[0][2], intrinsics_rgb[1][2]] # x-axis, y-axis



    # focal_rgb = [678, 678]
    # princpt_rgb = [318, 228]

    # iterate on the frames
    for nFrame,filename in enumerate(rgb_data_dir.iterdir()):
        if nFrame%2 == 0:
            original_img = cv2.imread(str(rgb_data_dir/filename.name))
            if original_img is None:
                print("Loading image failed.")
                continue
            original_img_height, original_img_width = original_img.shape[:-1]

            whole_time = time.time()

            # get bboxes
            boxes_time = time.time()

            # FASTER RCNN
            model_input = detector_transform(original_img).unsqueeze(0).to(device)
            outputs = detector_model(model_input)
            labels = outputs[0]['labels'].cpu().detach().numpy()
            # print(labels)
            pred_scores = outputs[0]['scores'].cpu().detach().numpy()
            pred_bboxes = outputs[0]['boxes'].cpu().detach().numpy()

            bbox_list = pred_bboxes[pred_scores >= detector_score_threshold]
            labels = labels[pred_scores >= detector_score_threshold]
            bbox_list = bbox_list[labels == 1]
            bbox_list_copy = bbox_list.copy()
            pose_time = time.time()

            # calculate roots
            person_num = len(bbox_list)
            root_depth_list = np.zeros(person_num)
            for n in range(person_num):
                # fix each bbox from (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)
                curr_bbox = bbox_list[n]
                curr_bbox[2] = curr_bbox[2] - curr_bbox[0]
                curr_bbox[3] = curr_bbox[3] - curr_bbox[1]
                bbox = rootnet_process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
                img, img2bb_trans = rootnet_generate_patch_image(original_img, bbox, False, 0.0)
                img = rootnet_transform(img).cpu()[None, :, :, :]

                k_value = np.array([math.sqrt(
                    bbox_real[0] * bbox_real[1] * focal_rgb[0] * focal_rgb[1] / (
                                bbox[2] * bbox[3]))]).astype(np.float32)
                k_value = torch.FloatTensor([k_value]).cpu()[None, :]

                # forward
                with torch.no_grad():
                    root_3d = rootnet_model(img, k_value)  # x,y: pixel, z: root-relative depth (mm)
                img = img[0].cpu().numpy()
                root_3d = root_3d[0].cpu().numpy()
                root_depth_list[n] = root_3d[2]

            if person_num < 1:
                continue

            # for each cropped and resized human image, forward it to PoseNet
            output_pose_2d = np.zeros((person_num, joint_num, 2))
            output_pose_3d = np.zeros((person_num, joint_num, 3))
            for n in range(person_num):
                bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
                img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
                img = posenet_transform(img).cpu()[None, :, :, :]

                # forward
                with torch.no_grad():
                    pose_3d = model(img)  # x,y: pixel, z: root-relative depth (mm)

                # inverse affine transform (restore the crop and resize)
                pose_3d = pose_3d[0].cpu().numpy()
                pose_3d[:, 0] = pose_3d[:, 0] / posenet_cfg.output_shape[1] * posenet_cfg.input_shape[1]
                pose_3d[:, 1] = pose_3d[:, 1] / posenet_cfg.output_shape[0] * posenet_cfg.input_shape[0]
                pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                output_pose_2d[n] = pose_3d[:, :2]

                # root-relative discretized depth -> absolute continuous depth
                pose_3d[:, 2] = (pose_3d[:, 2] / posenet_cfg.depth_dim * 2 - 1) * (posenet_cfg.bbox_3d_shape[0] / 2) + \
                                root_depth_list[n]
                pose_3d = pixel2cam(pose_3d, focal_rgb, princpt_rgb)
                output_pose_3d[n] = pose_3d

            #intrinsicts parameters depth camera
            intrinsics_depth = np.loadtxt(str(data_dir/"intrinsics_depth.txt"), dtype='f', delimiter=',')
            focal_depth = [intrinsics_depth[0][0], intrinsics_depth[1][1]]  # x-axis, y-axis
            princpt_depth = [intrinsics_depth[0][2], intrinsics_depth[1][2]]  # x-axis, y-axis

            print("FRAME:" +str(nFrame)+" time:%.4f," % (time.time() - whole_time), "boxes:%.4f," % (time.time() - boxes_time),
                  "pose:%.4f" % (time.time() - pose_time))

            # extract 2d poses
            vis_img = original_img.copy()
            for n in range(person_num):
                vis_kps = np.zeros((3, joint_num))
                vis_kps[0, :] = output_pose_2d[n][:, 0]
                vis_kps[1, :] = output_pose_2d[n][:, 1]
                vis_kps[2, :] = 1
                img_2d = vis_keypoints(vis_img, vis_kps, skeleton)
                img_2d = cv2.rectangle(img_2d,
                                       (int(bbox_list_copy[n][0]), int(bbox_list_copy[n][1])),
                                       (int(bbox_list_copy[n][2]), int(bbox_list_copy[n][3])),
                                       (0, 255, 0), thickness=1)
                vis_img = img_2d

            # find the depth frame correspondent to the rgb frame
            rgb_frame_name = f"{(int(filename.stem)-1):05}"
            if source == "pico":
                rgb_time_table = np.loadtxt(str(data_dir / "rgb" / f"PICO-rgb-{sequence}.txt"), dtype=str, delimiter='\t')
                rgb_timestamp = rgb_time_table[np.where(rgb_time_table[:, 0] == rgb_frame_name)[0][0]][1]
                depth_time_table = np.loadtxt(str(data_dir / "rgb" / f"PICO-rgb-{sequence}.txt"), dtype=str, delimiter='\t')
                depth_timestamp = process.extract(rgb_timestamp, depth_time_table[:, 1], limit=1)[0][0]
                depth_frame_name = depth_time_table[np.where(depth_time_table[:, 1] == depth_timestamp)[0][0]][0]
                depth = cv2.imread(f'{depth_data_dir / depth_frame_name}.png', -1)
            elif source == "kinect":
                depth_frame_filename = process.extract(rgb_frame_name, depth_data_dir.iterdir(), limit=1)[0][0]
                depth = cv2.imread(str(depth_frame_filename), -1)
            else:
                depth = np.zeros([original_img_width,original_img_height])
            pointcloud = depthmap2pointcloud(depth, focal_depth[0], focal_depth[1], princpt_depth[0], princpt_depth[1])
            points = np.array([output_pose_3d, pointcloud, vis_img])
            # points = np.array([output_pose_3d])

            if bboxdiff == True:
                output_dir = Path(f"results/{source}/{sequence}_{weights}_20")
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir}/{nFrame:05}_pose3D.npy", points)
            else:
                output_dir = Path(f"results/{source}/{sequence}_{weights}")
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(f"{output_dir}/{nFrame:05}_pose3D.npy", points)
        continue

if __name__ == "__main__":
    main()


