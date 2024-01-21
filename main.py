import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset

DESCRIPTION_FILE = pd.read_csv(dataset.INPUT_PATH)

dataset.output_dir("kmeans_clustering")


def get_images(sample, mag, k, notes=""):
    dataset.output_dir("%s_%d__k_is_%d_%s" % (sample, mag / 1000, k, notes))

    requirements = (DESCRIPTION_FILE[dataset.Variables.Sample] == sample) & (
        DESCRIPTION_FILE[dataset.Variables.Magnification] == mag
    )

    return DESCRIPTION_FILE.loc[requirements]


image_files = get_images(dataset.Sample.S1, dataset.Magnification.x100, 5)
# print(image_files)

# images = []


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
K = 5

color_labels = [
    list(colors.BLACK),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.ORANGE),
    list(colors.RED),
]

i = 1
for file in image_files[dataset.Variables.Filename]:
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

    cv2.imwrite(
        os.path.join(dataset.OUTPUT_PATH, file[:-4] + "_labeled.tif"),
        segmented_image,
    )
    cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), image)
    print("Done " + str(i))
    i += 1
