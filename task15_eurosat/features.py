# Name: Ekin Alev
# Date: 2026-11-21
# Class: DATA 471
# Assignment: Final Project Euro SAT

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Dataset paths used for testing the feature extraction function and image processing function
# --------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\ekin\Downloads\SCHOOL_CLASSES\CSCI_CLASSES\DATA 471"
    r"\Final Project\local_datasets\task15_eurosat"
)

# --------------------------------------------------
# 2. process images from a file and extract features for each image
# --------------------------------------------------

def process_images(filePath):
    # INPUT: path to train.x , dev.x, or test.x file
    all_features = []
    image = open(filePath)

    for line in image:
        # convert the line of text into a numpy array of ints
        image_arr = np.fromstring(line, dtype=np.uint8, sep=' ')

        # reshape the array into a 64x64x3 image
        image_reshaped = image_arr.reshape(64, 64, 3)

        # extract features from the image
        features = extract_features(image_reshaped)

        # append the features to the list
        all_features.append(features)

    # close the file after processing
    image.close()
    
    # Convert list of feature arrays to a 2D numpy array
    all_features = np.stack(all_features)  

    return all_features



# --------------------------------------------------
# 3. Extract features from images their r g b values and return a feature vector with the shape of 1 by 24
# --------------------------------------------------

def extract_features(image):
    # input one RGB image with the shape of 64 by 64 by 3 and return a feature vector with the shape of 1 by 24
    histogram = []
    for channel in range(3):
        pixels = image[:, :, channel].ravel()
        # Calculate histogram for each channel
        grid_RGB = pixels

        # calculate histogram for each channel
        count, edges = np.histogram(grid_RGB, bins=8, range=(0, 255))

        # Normalize the histogram
        normilized = count / count.sum()

        # add the normized histogram to the list
        histogram.append(normilized)

    # Concatenate histograms from all channels and return the 24 element feature array
    feature_vector = np.concatenate(histogram)

    return feature_vector




# Test feature extraction on the first satellite image
if __name__ == "__main__":

    # Load only one image
    pixels = np.loadtxt(
        DATA_DIR / "train.X",
        dtype=np.uint8,
        max_rows=1
    )

    # Convert flattened pixels into an RGB image
    image = pixels.reshape(64, 64, 3)

    # Call the function we created
    features = extract_features(image)

    # Inspect the results
    print("Feature shape:", features.shape)
    print("Features:", features)
    print("Total sum:", features.sum())

    # Validate the output
    assert features.shape == (24,)
    assert np.all(features >= 0)
    assert np.all(features <= 1)

    # Each RGB channel histogram should sum to 1
    channel_sums = features.reshape(3, 8).sum(axis=1)
    print("RGB histogram sums:", channel_sums)

    assert np.allclose(channel_sums, 1.0)

    print("All single-image tests passed!")

    dev_features = process_images(DATA_DIR / "dev.X")

    print("Development shape:", dev_features.shape)
