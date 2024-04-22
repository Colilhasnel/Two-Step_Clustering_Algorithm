import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy

color_labels = [
    list(colors.BLACK),
    list(colors.GREY),
    list(colors.GREEN),
    list(colors.CYAN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.ORANGE),
    list(colors.BROWN),
    list(colors.PINK),
    list(colors.PURPLE),
    list(colors.RED),
    list(colors.WHITE),
]

# No. of Clusters for KMeans
K = 12

# Setting Input Directory to denoised_images
dataset.INPUT_PATH = "output_images\Denoised_Images\S1_100k_fastNLmeansdenoising"

# Reading Description File
description_file = pd.read_csv(
    os.path.join(dataset.INPUT_PATH, "description_file.csv"), index_col=0
)

# Output Directory
dataset.output_dir("Step_1_KMeans_K12")

output_centers = pd.DataFrame(
    columns=[
        dataset.Variables.Filename,
        dataset.Variables.Centers,
    ]
)

for file in description_file[dataset.Variables.Filename]:

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

    # Calculating center of clusters; not using default sklearn centers attribute because not all algorithms have that attribute
    centers_new = [np.uint8(np.average(pixel_vals[labels == i])) for i in range(0, K)]

    # Writing Centers in a dataframe
    output_centers = output_centers._append(
        {
            dataset.Variables.Filename: file[:-11] + "_labeled.tif",
            dataset.Variables.Centers: copy.deepcopy(centers_new),
        },
        ignore_index=True,
    )

    # Assigning color labels in ascending order
    new_colors = [[0, 0, 0]] * K

    for i in range(0, K):
        idx = centers_new.index(min(centers_new))
        new_colors[idx] = color_labels[i]
        centers_new[idx] = 1e7

    new_colors = np.array(new_colors)
    new_colors = np.uint8(new_colors)

    # Segmenting data with colors
    segmented_data = new_colors[labels.flatten()]

    # Reshaping segmented data into image dimensions
    segmented_data = segmented_data.reshape(
        (denoised_image.shape[0], denoised_image.shape[1], 3)
    )

    cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), denoised_image)

    cv2.imwrite(
        os.path.join(dataset.OUTPUT_PATH, file[:-11] + "_labeled.tif"), segmented_data
    )

    print("Done " + file)

output_centers.to_csv(os.path.join(dataset.OUTPUT_PATH, "output_centers.csv"))
