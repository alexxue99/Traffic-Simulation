'''
This file uses the label map created in process_roads.py to compute the distance from each pixel to the start of its road, as well as the total length of its road.
The results are saved in pixel_ratios.json, which maps each pixel to a ratio between 0 and 1 representing how far along the road it is (0 = start, 1 = end).
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.util import invert
from PIL import Image
import json 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from road import Road

###### SETUP ######

# Load binary road mask (white roads on black background)
# Replace with your actual image
img = np.array(Image.open(".\\toy\\picture_tools\\separators.png").convert("RGB"))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green color range in HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create binary mask of green regions
mask = cv2.inRange(hsv, lower_green, upper_green)

# Create an RGB image for visualization
output_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Get connected components
label_map = np.load(".\\toy\\picture_tools\\label_map.npy")
num_labels = label_map.max() + 1

entry = Road('entry')
road2 = Road('road 2')
road3 = Road('road 3')
road4 = Road('road 4')
exit = Road('exit')
ROAD_LIST = [entry, road2, road3, road4, exit]

skeletons = [None for _ in range(num_labels)]
graphs = [None for _ in range(num_labels)]
idx_maps = [None for _ in range(num_labels)]
skel_coords = [None for _ in range(num_labels)]
endpoints = [None for _ in range(num_labels)]
start = [None for _ in range(num_labels)]
lengths = [None for _ in range(num_labels)]
dist_matrices = [None for _ in range(num_labels)]
pixel_to_ratio = {}

with open(".\\toy\\picture_tools\\labels_to_road_names.txt", "r") as f:
    # if road r has label l, then label_to_road[l - 1] = r
    label_to_road = [next(r for r in ROAD_LIST if r.name == line.strip()) for line in f]

###### METHODS ######

def to_skeleton(label):
    binary = (label_map == label).astype(np.uint8)
    # Skeletonize the mask using skimage (expects boolean image)
    skeletons[label] = skeletonize(binary > 0)

def find_skeleton_endpoints(label):
    skeleton = skeletons[label]
    skel_uint8 = skeleton.astype(np.uint8)
    kernel = np.array([[1,1,1], [1,10,1], [1,1,1]])
    filtered = cv2.filter2D(skel_uint8, -1, kernel)
    endpoints[label] = np.where(filtered == 11)

def skeleton_to_graph(label):
    skeleton = skeletons[label]
    h, w = skeleton.shape
    coords = np.argwhere(skeleton > 0)
    idx_map = {tuple(coord): i for i, coord in enumerate(coords)}
    rows, cols, weights = [], [], []

    for i, (y, x) in enumerate(coords):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == dx == 0: continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]):
                    j = idx_map.get((ny, nx))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        weights.append(1)
    graphs[label] = csr_matrix((weights, (rows, cols)), shape=(len(coords), len(coords)))
    idx_maps[label] = idx_map
    skel_coords[label] = coords

def find_skeleton_start(label):
    direction = label_to_road[label - 1].direction
    ys, xs = endpoints[label]

    match direction:
        case "LEFT_TO_RIGHT": # road is left to right, so the starting point is the left endpoint
            idx = np.argmin(xs)
        case "RIGHT_TO_LEFT":
            idx = np.argmax(xs)
        case "UP_TO_DOWN":
            idx = np.argmin(ys)
        case "DOWN_TO_UP":
            idx = np.argmax(ys)
    
    start[label] = idx_maps[label][(ys[idx], xs[idx])]

def run_dijkstra(label):
    dist_matrices[label] = dijkstra(graphs[label], indices = start[label])

def find_length(label):
    ys, xs = endpoints[label]
    lengths[label] = max(dist_matrices[label][idx_maps[label][(y,x)]] for (y, x) in zip(ys, xs))

def find_closest_skeleton_pixel(label, pixel):
    """
    Finds the closest skeleton pixel to the given pixel (y, x).
    """
    distances = np.linalg.norm(skel_coords[label] - np.array(pixel), axis=1)
    min_idx = np.argmin(distances)
    closest_pixel = tuple(skel_coords[label][min_idx])
    return closest_pixel

def get_ratio(label, pixel):
    closest_skel_pixel = find_closest_skeleton_pixel(label, pixel)
    index = idx_maps[label][closest_skel_pixel]

    pixel_to_ratio[pixel] = dist_matrices[label][index] / lengths[label]
        
###### RUN CODE ######
for label in range(1, num_labels):
    to_skeleton(label)
    find_skeleton_endpoints(label)
    skeleton_to_graph(label)
    find_skeleton_start(label)
    run_dijkstra(label)
    find_length(label)

height, width, _ = img.shape
for x in range(width):
    for y in range(height):
        pixel = (y, x)
        label = label_map[pixel]
        if label > 0:
            get_ratio(label, pixel)

# Convert keys to strings
ratio = {f"{x},{y}": v for (x, y), v in pixel_to_ratio.items()}

# Save
with open(".\\toy\\picture_tools\\pixel_ratios.json", "w") as f:
    json.dump(ratio, f)

##### VISUALIZATION #####

skeleton_total = np.zeros_like(mask)
endpoints_img = np.zeros_like(output_vis)

for label in range(1, num_labels):
    skeleton_uint8 = (skeletons[label] * 255).astype(np.uint8)

    # Add to the total skeleton image
    skeleton_total = cv2.bitwise_or(skeleton_total, skeleton_uint8)

    # Find endpoints
    idx = start[label]
    y, x = skel_coords[label][idx]

    # Draw endpoints on RGB visualization
    cv2.circle(endpoints_img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

# Overlay endpoints on skeleton
skeleton_rgb = cv2.cvtColor(skeleton_total, cv2.COLOR_GRAY2BGR)
skeleton_with_endpoints = cv2.addWeighted(skeleton_rgb, 1.0, endpoints_img, 1.0, 0)

fig, ax = plt.subplots()
ax.imshow(skeleton_with_endpoints, cmap = 'gray')
ax.set_title("Skeletonized")
plt.tight_layout()
plt.show()