import pandas as pd
import numpy as np
import os
import cv2

description_file = pd.read_csv(os.path.join("ordered_images", "labels_ordered.csv"))

output_path = os.path.join("output_images", "kmeans_clustering")

if not os.path.isdir(output_path):
    os.mkdir(output_path)

image_files = description_file["Filename"]

images = []

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
K = 6

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

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    print(labels.flatten())
    # Continue from here, run the code once. The centers are the value of color the cluster appears. Try changing the value of centers and the clusters will appear colorful
    break

    segmented_image = segmented_data.reshape((image.shape))

    cv2.imwrite(os.path.join(output_path, file), segmented_image)
    print("Done " + str(i))
    i += 1
