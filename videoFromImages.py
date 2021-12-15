import cv2
import numpy as np
from pathlib import Path

source = 'kinect'
weights = 'MuCo'
sequence = '004'
result_dir = Path(f"{source}/{sequence}_{weights}")
plot_dir = Path(f"plot/{result_dir}/")

img_array = []
for filename in plot_dir.iterdir():
    img = cv2.imread(str(filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(f'plot/{source}_{sequence}_{weights}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()