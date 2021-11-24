import matplotlib.pyplot as plt
import numpy as np
import typing as tp

def depthmap2points(image, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    image: np.array
        Array of depth values for the whole image.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    points: np.array
        Array of XYZ world coordinates.

    """
    h, w = image.shape

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def depthmap2pointcloud(depth, fx, fy, cx=None, cy=None):
    points = depthmap2points(depth, fx, fy, cx, cy)
    points = points.reshape((-1, 3))
    return points


def get_chain_dots(dots: np.ndarray, chain_dots_indexes: tp.List[int]) -> np.ndarray:  # chain of dots
    return dots[chain_dots_indexes]


def get_chains(dots: np.ndarray, arms_chain_ixs: tp.List[int], torso_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, arms_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs))


def subplot_nodes(dots: np.ndarray, ax):
    return ax.scatter3D(dots[:, 0], dots[:, 2], dots[:, 1], c=dots[:, 2])


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax):
    return [ax.plot(chain[:, 0], chain[:, 2], chain[:, 1]) for chain in chains]


def plot_skeletons(skeleton: tp.Sequence[np.ndarray], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]], pointcloud):

    fig1 = plt.figure("Pointcloud + Skeleton")
    ax = plt.axes(projection="3d")
    chains = get_chains(skeleton, *chains_ixs)
    subplot_nodes(skeleton, ax)
    subplot_bones(chains, ax)
    ax.scatter3D(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1])
    ax.view_init(10, 240)

    plt.show()


# if __name__ == "__main__":
#     intrinsic = [
#                  [fx, 0, cx],
#                  [0, fy, cy],
#                  [0, 0,   1]
#                 ]
#
#     pointcloud = depthmap2pointcloud(depth, fx=fx, fy=fy, cx=cx, cy=cy)  # depth in meters
#
#
#     # change these values wrt number and type of joints
#     chains_ixs = ([0, 1, 2, 3, 4], # hand_l, elbow_l, chest, elbow_r, hand_r
#                   [5, 2, 6], # pelvis, chest, head
#                   [7, 8, 5, 9, 10]) # foot_l, knee_l, pelvis, knee_r, foot_r
#
#     plot_skeletons(joints, chains_ixs, pointcloud[::30, :])