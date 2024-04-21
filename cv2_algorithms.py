import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset

DESCRIPTION_FILE = pd.read_csv(dataset.INPUT_PATH)

dataset.output_dir("kmeans_clustering")

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
K = 5

image_files = dataset.get_images(
    DESCRIPTION_FILE,
    dataset.Sample.S1,
    dataset.Magnification.x100,
    notes="x_y_added_only_15_img",
)

color_labels = [
    list(colors.BLACK),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.ORANGE),
    list(colors.RED),
]

counter = 1
for file in image_files[dataset.Variables.Filename]:
    if counter >= 15:
        break

    image = cv2.imread(os.path.join("ordered_images", file))
    image = image[:890, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pixel_vals = image.reshape((-1, 1))
    pixel_vals = np.float32(pixel_vals)

    # Continue from reading about precomputed affitinity matrix @ https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-segmentation-py

    new_centers = np.zeros((K, 3), dtype=np.uint8)

    for j in range(0, K):
        idx = np.where(min(centers[:, 2]) == centers)[0][0]
        new_centers[idx] = color_labels[j]
        centers[idx][2] = 1e7

    new_centers = np.uint8(new_centers)

    segmented_data = new_centers[labels.flatten()]

    segmented_image = segmented_data.reshape((image.shape[0], image.shape[1], 3))

    cv2.imwrite(
        os.path.join(dataset.OUTPUT_PATH, file[:-4] + "_labeled.tif"),
        segmented_image,
    )
    cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), image)
    print("Done " + str(counter))
    counter += 1
