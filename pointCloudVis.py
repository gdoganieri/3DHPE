import cv2
from d_visualization import depthmap2pointcloud

fx = 461.605
fy = 461.226
cx = 336.14
cy = 171.349

depth = cv2.imread("./data/depth/depth_10-19-25-032.png",-1)

depthmap2pointcloud(depth, fx, fy, cx, cy)