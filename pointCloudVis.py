import cv2
import argparse
import numpy as np
import os
import json
from pathlib import Path
from posenet.common.utils.vis import vis_3d_multiple_skeleton_and_pointcloud
from d_visualization import plot_skeletons, save_result_rot, save_skeletons_different_bboxset, plot_skeletons_track

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose()

    return poses_3d

def rotate_points(points_3D, R, t):
    R_inv = np.linalg.inv(R)
    for point in points_3D:
        point = point.transpose()
        point = np.dot(R_inv, point - t)
        point = point.transpose()

    return points_3D

def vis(weights, source, sequence, tracking):
    if weights == "MuCo":
        skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12),
                    (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
        chain_ixs = ([18, 7, 6, 5, 1, 2, 3, 4, 17],  # L_Hand, L_Wrist, L_Elbow, L_Shoulder, Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand
                     [14, 15, 1, 16, 0],  # Pelvis, Thorax, Spine, Head, Head_top
                     [20, 13, 12, 11, 14, 8, 9, 10, 19])  # L_Toe, L_Ankle, L_Knee, L_Hip, Pelvis, R_Hip, R_Knee, R_Ankle, R_Toe
    elif weights == "H36M":
        skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                    (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        chain_ixs = ([13, 12, 11, 17, 14, 15, 16],  # L_Wrist, L_Elbow, L_Shoulder, Thorax, R_Shoulder, R_Elbow, R_Wrist
                     [0, 7, 17, 8, 9, 10],  # Pelvis, Torso, Torax, Neck, Nose, Head
                     [6, 5, 4, 0, 1, 2, 3])  # L_Ankle, L_Knee, L_Hip, Pelvis, R_Hip, R_Knee, R_Ankle
    else:
        assert "Cannot visualize a correct pose. Not enough information provided."
        return -1

    result_dir = Path(f"results/tracking/{source}/{sequence}_{weights}")

    plot_dir = Path(f"plot/tracking/{source}/{sequence}_{weights}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for resNum, filepath in enumerate(result_dir.iterdir()):
        # if resNum >= 50:
        result = np.load(str(filepath), allow_pickle=True)
        output_pose_3d = result[0]
        pointcloud = result[1]
        # pointcloud = rotate_points(result[1], R, t)
        output_pose_2d = result[2]
        cv2.namedWindow("2DPose", cv2.WINDOW_NORMAL)
        cv2.imshow("2DPose", output_pose_2d)
        # output_pose_3d = rotate_poses(output_pose_3d, R, t)

        # save_result_rot(output_pose_3d, chain_ixs, pointcloud[::50, :], resNum, output_pose_2d, plot_dir)
        print(str(resNum))
        if tracking:
            tracking_predictions = result[3]
            tracking_colors = np.array(result[4])/255
            tracking_id = result[5]
            plot_skeletons_track(output_pose_3d, chain_ixs, pointcloud[::50, :],resNum, output_pose_2d, plot_dir, tracking_predictions, tracking_colors, tracking_id)
        else:
            plot_skeletons(output_pose_3d, chain_ixs, pointcloud[::50, :])


#
# vis_kps = np.array(output_pose_3d)
# vis_3d_multiple_skeleton_and_pointcloud(vis_kps, np.ones_like(vis_kps), skeleton,
#                     'output_pose_3d (x,y,z: camera-centered. mm.)', pointcloud[::50, :])

def vis_multi_bboxset(weights, source, sequence):
    if weights == "MuCo":
        skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12),
                    (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
        chain_ixs = ([18, 7, 6, 5, 1, 2, 3, 4, 17],  # L_Hand, L_Wrist, L_Elbow, L_Shoulder, Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand
                     [14, 1, 15, 16, 0],  # Pelvis, Thorax, Spine, Head, Head_top
                     [20, 13, 12, 11, 14, 8, 9, 10, 19])  # L_Toe, L_Ankle, L_Knee, L_Hip, Pelvis, R_Hip, R_Knee, R_Ankle, R_Toe
    elif weights == "H36M":
        skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                    (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        chain_ixs = ([13, 12, 11, 17, 14, 15, 16],  # L_Wrist, L_Elbow, L_Shoulder, Thorax, R_Shoulder, R_Elbow, R_Wrist
                     [0, 7, 17, 8, 9, 10],  # Pelvis, Torso, Torax, Neck, Nose, Head
                     [6, 5, 4, 0, 1, 2, 3])  # L_Ankle, L_Knee, L_Hip, Pelvis, R_Hip, R_Knee, R_Ankle
    else:
        assert "Cannot visualize a correct pose. Not enough information provided."
        return -1

    result_dir_20 = Path(f"results/{source}/{sequence}_{weights}_20")
    result_dir_2150 = Path(f"results/{source}/{sequence}_{weights}_2150")
    result_dir_2300 = Path(f"results/{source}/{sequence}_{weights}_2300")
    result_dir_2450 = Path(f"results/{source}/{sequence}_{weights}_2450")
    result_dir_2600 = Path(f"results/{source}/{sequence}_{weights}_2600")

    output_pose_3d = [dict() for x in range(5)]
    pointcloud = []
    output_pose_2d = []
    plot_dir = Path(f"plot/{source}/multiSet/{sequence}_{weights}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for n, filepath_20 in enumerate(result_dir_20.iterdir()):
        result = np.load(str(filepath_20), allow_pickle=True)
        output_pose_3d[0][n] = result[0]
        pointcloud.append(result[1])
        output_pose_2d.append(result[2])
        del result
    for n, filepath_2150 in enumerate(result_dir_2150.iterdir()):
        output_pose_3d[1][n] = np.load(str(filepath_2150), allow_pickle=True)[0]
    for n, filepath_2300 in enumerate(result_dir_2300.iterdir()):
        output_pose_3d[2][n] = np.load(str(filepath_2300), allow_pickle=True)[0]
    for n, filepath_2450 in enumerate(result_dir_2450.iterdir()):
        output_pose_3d[3][n] = np.load(str(filepath_2450), allow_pickle=True)[0]
    for n, filepath_2600 in enumerate(result_dir_2600.iterdir()):
        output_pose_3d[4][n] = np.load(str(filepath_2600), allow_pickle=True)[0]



    save_skeletons_different_bboxset(output_pose_3d, chain_ixs, pointcloud, output_pose_2d, plot_dir)

def main():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, dest='weights')
        parser.add_argument('--source', type=str, dest='source')
        parser.add_argument('--sequence', type=str, dest='sequence')
        parser.add_argument('--tracking', type=bool, dest='tracking', default=False)
        args = parser.parse_args()

        assert args.weights, 'Pretrained weights are required.'
        assert args.source, 'Source is required.'
        assert args.sequence, 'Sequence is required.'
        return args

    # argument parsing
    args = parse_args()

    # with open(f'data/{args.source}/extrinsics.json', 'r') as f:
    #     extrinsics = json.load(f)
    # R = np.array(extrinsics['R'], dtype=np.float32)
    # t = np.array(extrinsics['t'], dtype=np.float32)

    vis(args.weights, args.source, args.sequence, args.tracking)
    # vis_multi_bboxset(args.weights, args.source, args.sequence)
if __name__ == "__main__":
    main()