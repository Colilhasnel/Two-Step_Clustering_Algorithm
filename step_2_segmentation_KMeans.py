import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset
from ast import literal_eval
from sklearn.cluster import KMeans


local_centers = pd.read_csv(
    "output_images\KMeans_K_is_8_with_centres\output_centers.csv", index_col=0
)

center_array = np.array([])


for file in local_centers[dataset.Variables.Filename]:

    center_i = local_centers.loc[
        local_centers[dataset.Variables.Filename] == file, dataset.Variables.Centers
    ]

    center_i = center_i.tolist()

    list_value = literal_eval(center_i[0])

    center_array = np.append(center_array, np.array(list_value[0]))
    center_array = np.uint8(center_array)


center_array = center_array.reshape((-1, 1))
