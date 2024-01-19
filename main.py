import pandas as pd
import numpy as np
import os
import cv2
from colors import colors

description_file = pd.read_csv(os.path.join("ordered_images", "labels_ordered.csv"))

output_path = os.path.join("output_images", "kmeans_clustering")

if not os.path.isdir(output_path):
    os.mkdir(output_path)

image_files = description_file["Filename"]

images = []

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
K = 6

color_labels = [
    list(colors.BLACK),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.RED),
]

i = 1
for file in image_files:
    image = cv2.imread(os.path.join("ordered_images", file))
    image = image[:890, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pixel_vals = image.reshape((-1, 1))
    pixel_vals = np.float32(pixel_vals)

    retval, labels, centers = cv2.kmeans(
        pixel_vals, K, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    new_centers = np.zeros((K, 3), dtype=np.uint8)

    for j in range(0, K):
        idx = np.where(min(centers) == centers)[0][0]
        new_centers[idx] = color_labels[j]
        centers[idx][0] = 1e7

    new_centers = np.uint8(new_centers)

    segmented_data = new_centers[labels.flatten()]

    segmented_image = segmented_data.reshape((image.shape[0], image.shape[1], 3))

    cv2.imwrite(os.path.join(output_path, file), segmented_image)
    print("Done " + str(i))
    i += 1
