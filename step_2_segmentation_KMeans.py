import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset
from ast import literal_eval
from sklearn.cluster import KMeans


local_centers = pd.read_csv(
    "output_images\Step_1_KMeans_K12\output_centers.csv", index_col=0
)

global_K = 6

centers_array = np.array([])


color_labels = [
    list(colors.GREY),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.RED),
]


for file in local_centers[dataset.Variables.Filename]:

    centers = local_centers.loc[
        local_centers[dataset.Variables.Filename] == file, dataset.Variables.Centers
    ]

    centers = centers.tolist()

    centers_list = literal_eval(centers[0])

    centers_array = np.append(centers_array, np.array(centers_list))
    centers_array = np.uint8(centers_array)


centers_array = centers_array.reshape((-1, 1))

# Global KMeans Object
global_KMeans = KMeans(n_clusters=global_K, random_state=1234, n_init="auto").fit(
    centers_array
)

dataset.INPUT_PATH = "output_images\Denoised_Images\S1_100k_fastNLmeansdenoising"

# Reading Description File
description_file = pd.read_csv(
    os.path.join(dataset.INPUT_PATH, "description_file.csv"), index_col=0
)

# Setting Output Directory
dataset.output_dir("Step_2_KMeans_K6_Step_1_KMeans12")

for file in description_file[dataset.Variables.Filename]:
    # Loading the selected image
    denoised_image = cv2.imread(os.path.join(dataset.INPUT_PATH, file))
    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

    # Taking pixel_values of image as 1D array
    pixel_vals = denoised_image.reshape((-1, 1))
    pixel_vals = np.uint8(pixel_vals)

    # Predicting labels
    image_labels = global_KMeans.predict(pixel_vals)

    centers_new = [
        np.uint8(np.average(pixel_vals[image_labels == i])) for i in range(0, global_K)
    ]

    # Assigning color labels in ascending order
    new_colors = [[0, 0, 0]] * global_K

    for i in range(0, global_K):
        idx = centers_new.index(min(centers_new))
        new_colors[idx] = color_labels[i]
        centers_new[idx] = 1e7

    new_colors = np.array(new_colors)
    new_colors = np.uint8(new_colors)

    # Segmenting data with colors
    segmented_data = new_colors[image_labels.flatten()]

    # Reshaping segmented data into image dimensions
    segmented_data = segmented_data.reshape(
        (denoised_image.shape[0], denoised_image.shape[1], 3)
    )

    cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), denoised_image)

    cv2.imwrite(
        os.path.join(dataset.OUTPUT_PATH, file[:-11] + "_global_labeled.tif"),
        segmented_data,
    )

    print("Done " + file)
