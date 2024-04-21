import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset
from sklearn.cluster import KMeans
import copy

color_labels = [
    list(colors.BLACK),
    list(colors.WHITE),
    list(colors.GREEN),
    list(colors.CYAN),
    list(colors.BLUE),
    list(colors.YELLOW),
    list(colors.ORANGE),
    list(colors.RED),
]

dataset.INPUT_PATH = "output_images\Denoised_Images\S1_100k_fastNLmeansdenoising"

# Reading Description File
description_file = pd.read_csv(
    os.path.join(dataset.INPUT_PATH, "description_file.csv"), index_col=0
)

# Output Directory
dataset.output_dir("KMeans_K_is_8_with_centres")
