from pathlib import Path

import cv2

source = 'kinect'
weights = 'MuCo'
sequence = '004'
result_dir = Path(f"{source}/{sequence}_{weights}")
plot_dir = Path(f"plot/tracking/{result_dir}/")

img_array = []
for filename in plot_dir.iterdir():
    img = cv2.imread(str(filename))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(f'plot/tracking_{source}_{sequence}_{weights}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
