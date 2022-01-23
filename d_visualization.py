
import matplotlib.pyplot as plt
import numpy as np
import typing as tp

from tracking.yolox.utils.visualize import get_color


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


def subplot_nodes(dots: np.ndarray, ax, c):
    return ax.scatter3D(dots[:, 0], dots[:, 2], -dots[:, 1], c=c)


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax, c):
    return [ax.plot(chain[:, 0], chain[:, 2], -chain[:, 1], color=c) for chain in chains]


def vis_skeletons(skeletons: tp.Sequence[np.ndarray], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]],
                  pointcloud, resNum, pose2D, plot_dir, save):

    fig = plt.figure("Pointcloud + Skeleton")
    ax = plt.axes(projection="3d")
    for skeleton in skeletons:
        chains = get_chains(skeleton, *chains_ixs)
        subplot_nodes(skeleton, ax, skeleton[:, 2])
        subplot_bones(chains, ax, skeleton[:, 2])
    ax.scatter3D(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1], s=2)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.dist = 8

    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    if save:
        if resNum <= 360:
            angle = resNum
        elif resNum <= 720:
            angle = resNum - 360
        else:
            angle = resNum - 720

        ax.view_init(10, -angle)
        ax = fig.add_subplot(3, 3, 9)
        ax = plt.imshow(pose2D[:, :, ::-1])
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        plt.savefig(f"{plot_dir}/{resNum:05}.png")
        plt.clf()

    else:
        ax.view_init(10, 240)
        plt.show()

def vis_skeletons_track(skeletons: tp.Sequence[np.ndarray], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]],
                        pointcloud, resNum, pose2D, plot_dir, tracking_ids, save):

    fig1 = plt.figure("Pointcloud + Skeleton")
    ax = plt.axes(projection="3d")
    for n, skeleton in enumerate(skeletons):
        if len(tracking_ids) >=1:
            obj_id = int(tracking_ids[n])
            id_text = '{}'.format(int(obj_id))
            color = get_color(abs(obj_id))
            color = np.array(color[::-1], dtype=np.float32)/255
            # color = (color[::-1] / 255).dtype=np.float32
            # ax.scatter3D(tracking_predictions[n][0], tracking_predictions[n][2], tracking_predictions[n][1],
            #              color='g', marker='^', s=50)
            ax.text(skeleton[0, 0], skeleton[0, 2], -skeleton[0, 1] + 250, id_text, color=color)
        else:
            color = skeleton[:,2]
        chains = get_chains(skeleton, *chains_ixs)
        subplot_nodes(skeleton, ax,color)
        subplot_bones(chains, ax, color)

        # for k in range(len(tracking_traces)):
        #     ax.scatter3D(tracking_traces[k][0][0], tracking_traces[k][0][1], tracking_traces[k][0][2],
        #                  c=tracking_colors[n], s=10)
    ax.scatter3D(pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1], s=2)

    ax.dist = 8
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    if save:
        if resNum <= 360:
            angle = resNum
        elif resNum <= 720:
            angle = resNum - 360
        else:
            angle = resNum - 720

        ax.view_init(10, -angle)
        ax.set_zlabel('y')
        ax = fig1.add_subplot(3, 3, 9)
        ax = plt.imshow(pose2D[:, :, ::-1])
        plt.savefig(f"{plot_dir}/{resNum:05}.png")
        plt.clf()

    else:
        ax.view_init(10, 240)
        plt.show()

# save the result with a constant rotation of the scene
def vis_skeletons_different_bboxset(output_pose_3d: tp.List[tp.Dict[int, tp.Sequence[np.ndarray]]], chains_ixs: tp.Tuple[tp.List[int], tp.List[int], tp.List[int]],
                   pointcloud, pose2D, plot_dir, save):
    fig = plt.figure("Pointcloud + Skeleton")
    colors = ['r', 'g', 'b', 'y', 'c']
    labels = ['2000', '2150', '2300', '2450', '2600']
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    for nFrames, skeletons_20 in enumerate(output_pose_3d[0].values()):
        ax = plt.axes(projection="3d")
        skeletons_2150 = output_pose_3d[1][nFrames]
        skeletons_2300 = output_pose_3d[2][nFrames]
        skeletons_2450 = output_pose_3d[3][nFrames]
        skeletons_2600 = output_pose_3d[4][nFrames]
        for nSkel, skeleton in enumerate(skeletons_20):
            pointcloud_small = pointcloud[nFrames][::50]
            chains = get_chains(skeleton, *chains_ixs)
            subplot_nodes(skeleton, ax, colors[0])
            subplot_bones(chains, ax)
            ax.text(skeleton[10, 0], skeleton[10, 2], -skeleton[10, 1] + 250, labels[0], color=colors[0])
            ax.scatter3D(pointcloud_small[:, 0], pointcloud_small[:, 2], pointcloud_small[:, 1], s=2, c='steelblue')
        for skeleton in skeletons_2150:
            chains = get_chains(skeleton, *chains_ixs)
            subplot_nodes(skeleton, ax, colors[1])
            subplot_bones(chains, ax)
            ax.text(skeleton[10, 0], skeleton[10, 2], -skeleton[10, 1] + 250, labels[1], color=colors[1])
        for skeleton in skeletons_2300:
            chains = get_chains(skeleton, *chains_ixs)
            subplot_nodes(skeleton, ax, colors[2])
            subplot_bones(chains, ax)
            ax.text(skeleton[10, 0], skeleton[10, 2], -skeleton[10, 1] + 250, labels[2], color=colors[2])
        for skeleton in skeletons_2450:
            chains = get_chains(skeleton, *chains_ixs)
            subplot_nodes(skeleton, ax, colors[3])
            subplot_bones(chains, ax)
            ax.text(skeleton[10, 0], skeleton[10, 2], -skeleton[10, 1] + 250, labels[3], color=colors[3])
        for skeleton in skeletons_2600:
            chains = get_chains(skeleton, *chains_ixs)
            subplot_nodes(skeleton, ax, colors[4])
            subplot_bones(chains, ax)
            ax.text(skeleton[10, 0], skeleton[10, 2], -skeleton[10, 1] + 250, labels[4], color=colors[4])

        if nFrames <= 360:
            angle = nFrames
        elif nFrames <= 720:
            angle = nFrames - 360
        else:
            angle = nFrames - 720

        ax.view_init(10, -angle)
        ax.dist = 8

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        fig.add_subplot(3, 3, 9)
        plt.imshow(pose2D[nFrames][:, :, ::-1])
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if save:
            plt.savefig(f"{plot_dir}/{nFrames:05}.png")
            plt.clf()

        else:
            plt.show()
            plt.clf()