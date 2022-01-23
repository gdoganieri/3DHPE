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
from tracking.tracker import Tracker

posenet_path = os.getcwd() + "/posenet"

from posenet.main.config import cfg as posenet_cfg
from posenet.main.model import get_pose_net
from posenet.data.dataset import generate_patch_image
from posenet.common.posenet_utils.pose_utils import process_bbox, pixel2cam, cam2pixel

rootnet_path = os.getcwd() + "/rootnet"

from rootnet.main.config import cfg as rootnet_cfg
from rootnet.main.model import get_pose_net as get_root_net
from rootnet.common.rootnet_utils.pose_utils import process_bbox as rootnet_process_bbox
from rootnet.data.dataset import generate_patch_image as rootnet_generate_patch_image

import torchvision

from pathlib import Path
from posenet.common.posenet_utils.vis import vis_keypoints, vis_keypoints_track
from d_visualization import depthmap2pointcloud, pixel2world, vis_skeletons_track


# from tracking.tracker import skeleton_track
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking

def main():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, dest='gpu_ids')
        parser.add_argument('--weights', type=str, dest='weights')
        parser.add_argument('--source', type=str, dest='source')
        parser.add_argument('--sequence', type=str, dest='sequence')
        parser.add_argument('--tracking', dest='tracking', action='store_true')
        parser.add_argument('--no-tracking', dest='tracking', action='store_false')
        parser.set_defaults(tracking=True)
        parser.add_argument('--bboxdiff', dest='bboxdiff', action='store_true')
        parser.add_argument('--no-bboxdiff', dest='bboxdiff', action='store_false')
        parser.set_defaults(bboxdiff=False)
        parser.add_argument('--yolo', dest='yolo', action='store_true')
        parser.add_argument('--no-yolo', dest='yolo', action='store_false')
        parser.set_defaults(yolo=True)
        parser.add_argument('--byteTrack', dest='byteTrack', action='store_true')
        parser.add_argument('--no-byteTrack', dest='byteTrack', action='store_false')
        parser.set_defaults(byteTrack=True)
        #tracking argument byteTrack
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

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

    sequence = args.sequence
    source = args.source
    bboxdiff = args.bboxdiff
    track = args.tracking
    weights = args.weights
    use_yolo = args.yolo
    use_byteTrack = args.byteTrack

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if use_yolo:
        # Model
        detector_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        detector_model.eval().to(device)
        detector_model.classes = [0]  # filter for specific classes

    else:
        # FASTER RCNN v3 320 fpn
        detector_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,
                                                                                            pretrained_backbone=True)
        detector_model.eval().to(device)

    detector_score_threshold = 0.8
    detector_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    posenet_cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # Check the weights to load and to find how to build the skeleton
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
        root_pt = 14
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
        root_pt = 0
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

    # path management
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

    # tracking init
    if use_byteTrack:
        tracker = BYTETracker(args)
    else:
        tracker = Tracker(500, 10, 5)

    # iterate on the frames
    for nFrame,filename in enumerate(rgb_data_dir.iterdir()):
        if nFrame%2 == 0 and nFrame > 0:
            original_img = cv2.imread(str(rgb_data_dir/filename.name))
            if original_img is None:
                print("Loading image failed.")
                continue
            original_img_height, original_img_width = original_img.shape[:-1]

            whole_time = time.time()

            # get bboxes
            boxes_time = time.time()
            model_input = detector_transform(original_img).unsqueeze(0).to(device)
            if use_yolo:
                # YOLO
                model_output = detector_model(original_img)
                results = model_output.pandas().xyxy[0]
                pred_classes = results["name"]
                labels = results["class"]
                bboxs = model_output.xyxy[0].cpu().numpy()

                scores = bboxs[:, 4]
                remain_inds = scores > detector_score_threshold
                bbox_list = bboxs[remain_inds]
            else:
                # FASTER RCNN
                outputs = detector_model(model_input)
                labels = outputs[0]['labels'].cpu().detach().numpy()
                # print(labels)
                pred_scores = outputs[0]['scores'].cpu().detach().numpy()
                bboxes = outputs[0]['boxes'].cpu().detach().numpy()

                bboxs = np.zeros([len(bboxes), 5])
                bboxs[:, :4] = [bbox for bbox in bboxes]
                bboxs[:, 4] = [score for score in pred_scores]

                bbox_list = bboxs[pred_scores >= detector_score_threshold]
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
                bbox = rootnet_process_bbox(np.array(bbox_list[n, :4]), original_img_width, original_img_height)
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
            outpose_tracking = np.zeros((person_num, joint_num, 3))
            for n in range(person_num):
                bbox = process_bbox(np.array(bbox_list[n, :4]), original_img_width, original_img_height)
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
                outpose_tracking[n] = pose_3d
                pose_3d = pixel2cam(pose_3d, focal_rgb, princpt_rgb)
                output_pose_3d[n] = pose_3d

            #intrinsicts parameters depth camera
            intrinsics_depth = np.loadtxt(str(data_dir/"intrinsics_depth.txt"), dtype='f', delimiter=',')
            focal_depth = [intrinsics_depth[0][0], intrinsics_depth[1][1]]  # x-axis, y-axis
            princpt_depth = [intrinsics_depth[0][2], intrinsics_depth[1][2]]  # x-axis, y-axis

            print("FRAME:" +str(nFrame)+" time:%.4f," % (time.time() - whole_time), "boxes:%.4f," % (time.time() - boxes_time),
                  "pose:%.4f" % (time.time() - pose_time))

            # tracking
            vis_img = original_img.copy()

            tracking_bboxes = []
            tracking_ids = []
            tracking_scores = []
            tracking_skeletons = []
            tracking_skeletons_3D = []

            if use_byteTrack:
                tracking_targets = tracker.update(bbox_list_copy[:,:5], outpose_tracking, [original_img_height, original_img_width], (original_img_height, original_img_width))
                for t in tracking_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        tracking_bboxes.append(tlwh)
                        tracking_ids.append(tid)
                        tracking_scores.append(t.score)
                        tracking_skeletons.append(t.skeleton)

            else:
                tracker.update(outpose_tracking, root_pt, bbox_list_copy[:,:4])
                for t in tracker.tracks:
                    # tracking_predictions.append(np.array(pixel2world(int(online_targets.tracks[i].trace[-1][0, 0]),
                    #                                         int(online_targets.tracks[i].trace[-1][0, 1]),
                    #                                         int(online_targets.tracks[i].trace[-1][0, 2]),
                    #                                         original_img_width, original_img_height,
                    #                                         focal_rgb[0], focal_rgb[1],
                    #                                         princpt_rgb[0], princpt_rgb[1])))
                    tracking_ids.append(t.trackId)
                    current_skeleton = t.track_skeleton.copy()
                    current_skeleton = pixel2cam(current_skeleton, focal_rgb, princpt_rgb)
                    tracking_skeletons_3D.append(current_skeleton)
                    tracking_skeletons.append(t.track_skeleton)
                    curr_bbox = t.track_bbox
                    curr_bbox[2] = curr_bbox[2] - curr_bbox[0]
                    curr_bbox[3] = curr_bbox[3] - curr_bbox[1]
                    tracking_bboxes.append(curr_bbox)

            vis_img = plot_tracking(
                vis_img, tracking_bboxes, tracking_ids, tracking_skeletons, skeleton, joint_num, frame_id=nFrame,
                fps=1. / (time.time() - whole_time)
            )
            # extract 2d poses
                # for n in range(person_num):
                #     vis_kps = np.zeros((3, joint_num))
                #     # vis_kps[0, :] = output_pose_2d[n][:, 0]
                #     # vis_kps[1, :] = output_pose_2d[n][:, 1]
                #     vis_kps[0, :] = online_targets.tracks[n].track_skeleton[:,0]
                #     vis_kps[1, :] = online_targets.tracks[n].track_skeleton[:,1]
                #     vis_kps[2, :] = 1
                #     img_2d = vis_keypoints_track(vis_img, vis_kps, skeleton, online_targets.tracks[n].track_color)
                #     img_2d = cv2.rectangle(img_2d,
                #                            (int(online_targets.tracks[n].track_bbox[0]), int(online_targets.tracks[n].track_bbox[1])),
                #                            (int(online_targets.tracks[n].track_bbox[2]), int(online_targets.tracks[n].track_bbox[3])),
                #                            online_targets.tracks[n].track_color, thickness=1)
                #
                #     cv2.putText(img_2d, str(online_targets.tracks[n].trackId),
                #                 (int(online_targets.tracks[n].track_bbox[0]), int(online_targets.tracks[n].track_bbox[1])-10), 0, 0.5,
                #                 online_targets.tracks[n].track_color, 2)
                #     vis_img = img_2d

            cv2.namedWindow("2D Detection + Pose", cv2.WINDOW_NORMAL)
            vis_img = cv2.resize(vis_img, (1080, 640))
            cv2.imshow('image', vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

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

            # points = np.array([output_pose_3d, pointcloud, vis_img, tracking_predictions, tracking_traces, tracking_colors, tracking_id])


            if bboxdiff:
                points = np.array([output_pose_3d, pointcloud, vis_img])
                output_dir = Path(f"results/{source}/{sequence}_{weights}_20")
                output_dir.mkdir(parents=True, exist_ok=True)
            elif track:
                points = np.array([tracking_skeletons_3D, pointcloud, vis_img, tracking_ids])
                mode = "tracking"
                if use_yolo:
                    detector = "yolo"
                    if use_byteTrack:
                        tracker_type = "byteTrack"
                    else:
                        tracker_type = "kalmanTrack"
                else:
                    detector = "frcnn"
                    if use_byteTrack:
                        tracker_type = "byteTrack"
                    else:
                        tracker_type = "kalmanTrack"
                output_dir = Path(f"results/{mode}/{detector}/{tracker_type}/{source}/{sequence}_{weights}")
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                points = np.array([output_pose_3d, pointcloud, vis_img])
                mode = "noTracking"
                if use_yolo:
                    detector = "yolo"
                else:
                    detector = "frcnn"
                output_dir = Path(f"results/{mode}/{detector}/{source}/{sequence}_{weights}")
                output_dir.mkdir(parents=True, exist_ok=True)

            np.save(f"{output_dir}/{nFrame:05}_pose3D.npy", points)
        continue

if __name__ == "__main__":
    main()


