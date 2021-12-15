
import matplotlib.pyplot as plt
import numpy as np
import typing as tp

def world2pixel(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    p_x = x * fx / z + cx
    p_y = cy - y * fy / z
    return p_x, p_y

def pixel2world(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    x: np.array
        Array of X image coordinates.
    y: np.array
        Array of Y image coordinates.
    z: np.array
        Array of depth values for the whole image.
    img_width: int
        Width image dimension.
    img_height: int
        Height image dimension.
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
    w_x: np.array
        Array of X world coordinates.
    w_y: np.array
        Array of Y world coordinates.
    w_z: np.array
        Array of Z world coordinates.

    """
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    w_x = (x - cx) * z / fx
    w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z

def points2pixels(points, img_width, img_height, fx, fy, cx=None, cy=None):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
    return pixels

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


def get_chains(dots: np.ndarray, arms_chain_ixs: tp.List[int], torso_chain_ixs: tp.List[int], legs_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, arms_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs),
            get_chain_dots(dots, legs_chain_ixs))


def subplot_nodes(dots: np.ndarray, ax):
    return ax.scatter3D(dots[:, 0], dots[:, 2], -dots[:, 1], c=dots[:, 2])


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax):
    return [ax.plot(chain[:, 0], chain[:, 2], -chain[:, 1]) for chain in chains]


def plot_skeletons(skeletons: tp.Sequence[np.ndarray], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]], pointcloud):

    fig1 = plt.figure("Pointcloud + Skeleton")
    ax = plt.axes(projection="3d")
    for skeleton in skeletons:
        chains = get_chains(skeleton, *chains_ixs)
        subplot_nodes(skeleton, ax)
        subplot_bones(chains, ax)
    ax.scatter3D(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1], s=2)
    ax.view_init(10, 240)

    plt.show()


def plot_skeletons_rot(skeletons: tp.Sequence[np.ndarray], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]],
                   pointcloud, angle, pose2D):
    fig = plt.figure("Pointcloud + Skeleton")
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.axes(projection="3d")
    for skeleton in skeletons:
        chains = get_chains(skeleton, *chains_ixs)
        subplot_nodes(skeleton, ax)
        subplot_bones(chains, ax)
    ax.scatter3D(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1], s=2)
    ax.view_init(10, -angle)
    ax.dist = 8

    ax = fig.add_subplot(3, 3, 9)
    ax = plt.imshow(pose2D[:, :, ::-1])
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

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