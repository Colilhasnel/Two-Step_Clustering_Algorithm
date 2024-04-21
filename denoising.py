import pandas as pd
import numpy as np
import os
import cv2
from colors import colors
from dataset_parameters import dataset


# Creating Output Directory for output images
dataset.output_dir("Denoised_Images")

# Reading the Description File of images
DESCRIPTION_FILE = pd.read_csv(
    os.path.join(dataset.INPUT_PATH, "ordered_description.csv")
)

# Getting a selected set of images
image_files = dataset.get_images(
    DESCRIPTION_FILE,
    dataset.Sample.S1,
    dataset.Magnification.x100,
    "fastNLmeansdenoising",
)

output_description_file = image_files.copy(True)

# Iterating on files
for file in image_files[dataset.Variables.Filename]:

    # Reading, Cropping and Converting it to B/W
    origianl_image = cv2.imread(os.path.join("ordered_images", file))
    origianl_image = origianl_image[:890][:]
    origianl_image = cv2.cvtColor(origianl_image, cv2.COLOR_BGR2GRAY)

    # Denoising Image
    denoised_image = cv2.fastNlMeansDenoising(
        origianl_image, templateWindowSize=7, searchWindowSize=21, h=8
    )

    # Writing Original Image (For Easier Comparison)
    cv2.imwrite(os.path.join(dataset.OUTPUT_PATH, file), origianl_image)

    # Writing Denoised Image
    cv2.imwrite(
        os.path.join(dataset.OUTPUT_PATH, file[:-4] + "_blurred.tif"),
        denoised_image,
    )

    # New Images Description File, for further easier use of them
    output_description_file.loc[
        output_description_file[dataset.Variables.Filename] == file,
        dataset.Variables.Filename,
    ] = (
        file[:-4] + "_blurred.tif"
    )

    print("Done " + file)

output_description_file.reset_index(drop=True, inplace=True)

output_description_file.to_csv(
    os.path.join(dataset.OUTPUT_PATH, "description_file.csv")
)


# Converting the image into a numpy array
# pixel_vals = image.reshape((-1, 1))
# # pixel_vals = np.float32(pixel_vals)
# print(pixel_vals)

# print("Here")
# print(pixel_vals.shape)

# hdb = HDBSCAN(min_cluster_size=50)
# print("will Start fitting")
# labels = hdb.fit_predict(pixel_vals)

# print(labels)
