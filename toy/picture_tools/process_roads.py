'''
This file processes the roads in the image, creating a map from pixels to roads. The results are saved in label_map.npy.
'''

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

img = np.array(Image.open(".\\toy\\picture_tools\\separators.png").convert("RGB"))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green color range in HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create binary mask of green regions
mask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imshow('mask', mask)
cv2.waitKey(0)

# Label connected components
num_labels, label_map = cv2.connectedComponents(mask)

# Show the label map
plt.imshow(label_map, cmap='tab20')
plt.title(f"{num_labels - 1} roads detected")
plt.show()

np.save(".\\toy\\picture_tools\\label_map.npy", label_map)

print(label_map.max())
i = 0
while i < label_map.max() + 1:
    mask1 = (label_map > 0)
    result = np.zeros_like(img)
    result[mask1, :] = [0, 0, 255]
    
    mask = (label_map == i) > 0
    result[mask, :] = img[mask, :]

    # Display the result
    s = 'Label ' + str(i)
    cv2.namedWindow(s)
    cv2.moveWindow(s, 950, 50)
    cv2.imshow(s, result)
    key = cv2.waitKey(0)
 
    if key == 27:
        break # Escape key to break
    if key == 0:
        i -= 2 # Left arrow to go back
    i += 1 # Other keys to go forward
    cv2.destroyAllWindows()
