import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset
from sklearn.cluster import KMeans

color_labels = [
    list(colors.BLACK),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.BLUE),
    list(colors.CYAN),
    list(colors.YELLOW),
    list(colors.ORANGE),
    list(colors.RED),
]

# No. of Clusters for KMeans
K = 8

# Setting Input Directory to denoised_images
dataset.INPUT_PATH = "output_images\Denoised_Images\S1_100k_fastNLmeansdenoising"

# Reading Description File
description_file = pd.read_csv(os.path.join(dataset.INPUT_PATH, "description_file.csv"))

# Selecting a single file
file = description_file[dataset.Variables.Filename][1]

# Loading the selected image
denoised_image = cv2.imread(os.path.join(dataset.INPUT_PATH, file))
denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

# Taking pixel_values of image as 1D array
pixel_vals = denoised_image.reshape((-1, 1))
pixel_vals = np.float32(pixel_vals)

# Applying KMeans
kmeans_obj = KMeans(n_clusters=K, random_state=1234, n_init="auto").fit(pixel_vals)

# Labels & Centers of KMeans
labels = kmeans_obj.labels_
centers = kmeans_obj.cluster_centers_

# New centers to be colored
new_centers = np.zeros((K, 3), dtype=np.uint8)
new_centers = np.uint8(new_centers)


for j in range(0, K):
    idx = np.where(min(centers[:, 0]) == centers)[0][0]
    new_centers[idx] = color_labels[j]
    centers[idx][0] = 1e7

segmented_data = new_centers[labels.flatten()]

segmented_image = segmented_data.reshape(
    (denoised_image.shape[0], denoised_image.shape[1], 3)
)

dataset.output_dir("Single_Image_testing")

cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), denoised_image)

cv2.imwrite(
    os.path.join(dataset.OUTPUT_PATH, file[:-4] + "_labeled.tif"), segmented_image
)
