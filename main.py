import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset

DESCRIPTION_FILE = pd.read_csv(dataset.INPUT_PATH)


def make_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)

make_path(dataset.OUTPUT_PATH)
dataset.OUTPUT_PATH = os.path.join(dataset.OUTPUT_PATH, "kmeans_clustering")


def get_images(sample, mag, k, notes=""):
    dataset.OUTPUT_PATH = os.path.join(
        dataset.OUTPUT_PATH, "%s_%d__k_is_%d_%s" % (sample, mag / 1000, k, notes)
    )

    requirements = (DESCRIPTION_FILE[dataset.Variables.Sample] == sample) & (
        DESCRIPTION_FILE[dataset.Variables.Magnification] == mag
    )

    return DESCRIPTION_FILE.loc[requirements]


print(get_images(dataset.Sample.S1, dataset.Magnification.x100, 4))

# image_files = description_file["Filename"]

# images = []

# CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
# K = 7

# color_labels = [
#     list(colors.BLACK),
#     list(colors.WHITE),
#     list(colors.GREEN),
#     list(colors.BLUE),
#     list(colors.YELLOW),
#     list(colors.ORANGE),
#     list(colors.RED),
# ]

# i = 1
# for file in image_files:
#     image = cv2.imread(os.path.join("ordered_images", file))
#     image = image[:890, :]
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     pixel_vals = image.reshape((-1, 1))
#     pixel_vals = np.float32(pixel_vals)

#     retval, labels, centers = cv2.kmeans(
#         pixel_vals, K, None, CRITERIA, 10, cv2.KMEANS_RANDOM_CENTERS
#     )

#     new_centers = np.zeros((K, 3), dtype=np.uint8)

#     for j in range(0, K):
#         idx = np.where(min(centers) == centers)[0][0]
#         new_centers[idx] = color_labels[j]
#         centers[idx][0] = 1e7

#     new_centers = np.uint8(new_centers)

#     segmented_data = new_centers[labels.flatten()]

#     segmented_image = segmented_data.reshape((image.shape[0], image.shape[1], 3))

#     cv2.imwrite(os.path.join(output_path, file[:-4] + "_labeled.tif"), segmented_image)
#     cv2.imwrite(os.path.join(output_path, file), image)
#     print("Done " + str(i))
#     i += 1
