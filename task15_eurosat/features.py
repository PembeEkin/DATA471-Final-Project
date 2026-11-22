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
# 3. Extract features from images their R G B values and return a feature vector with the shape of 1 by 24
# --------------------------------------------------

def rgb_histogram(image):
    # input one RGB image with the shape of 64 by 64 by 3 and return a feature vector with the shape of 1 by 24
    histogram = []
    for channel in range(3):
        pixels = image[:, :, channel].ravel()
        # Calculate histogram for each channel
        grid_RGB = pixels

        # calculate histogram for each channel
        count, edges = np.histogram(grid_RGB, bins=8, range=(0, 256))

        # Normalize the histogram
        normilized = count / count.sum()

        # add the normized histogram to the list
        histogram.append(normilized)

    # Concatenate histograms from all channels and return the 24 element feature array
    feature_vector = np.concatenate(histogram)
    return feature_vector



# --------------------------------------------------
# 4. Extract features from images their R G B values and return a feature vector with the shape of 1 by 24
# --------------------------------------------------
def extract_features(image):
    # 1. Calculate the global RGB histogram (24 features)
    global_histogram = rgb_histogram(image)

    # 2. Split the image into 3 horzontal sections 
    horizontal = np.array_split(image, 3, axis=0)

    # 3. Create an empty list for the 9 spatial histograms
    spatial_histograms = []

    # 4. Loop through Horizontal sections
    for section in horizontal:

        # Split that section into 3 vertical sections
        vertical = np.array_split(section, 3, axis=1)

        # loop through each vertical section
        for sub_section in vertical:
            # Use rgb_histogram to calculate the histogram for each region
            region_histogram = rgb_histogram(sub_section)

            # Append the region histogram to the list its shape will be 1 by 24
            spatial_histograms.append(region_histogram)

    # 5. Combine the global and regional histogram features
    all_features = np.concatenate([global_histogram] + spatial_histograms)

    # 6. Return the 240-element feature vector
    return all_features


if __name__ == "__main__":

    # Load the first training image
    pixels = np.loadtxt(
        DATA_DIR / "train.X",
        dtype=np.uint8,
        max_rows=1
    )

    image = pixels.reshape(64, 64, 3)

    # Test the new 240-feature extraction
    features = extract_features(image)

    print("Feature shape:", features.shape)
    print("Total feature sum:", features.sum())

    # Test 1: Correct number of features
    assert features.shape == (240,)

    # Test 2: All values must be between 0 and 1
    assert np.all(features >= 0)
    assert np.all(features <= 1)

    # Test 3: No NaN or infinity
    assert np.all(np.isfinite(features))

    # Test 4: Check the 10 RGB histograms
    histogram_groups = features.reshape(10, 3, 8)

    channel_sums = histogram_groups.sum(axis=2)

    print("Channel sums shape:", channel_sums.shape)
    print("Channel sums:", channel_sums)

    assert np.allclose(channel_sums, 1.0)

    # Test 5: Verify the first 24 features
    # match the original global histogram
    original_features = rgb_histogram(image)

    assert np.allclose(features[:24], original_features)

    print("All single-image tests passed!")


    dev_features = process_images(DATA_DIR / "dev.X")

    print("Development feature shape:", dev_features.shape)

    assert dev_features.shape == (5400, 240)
    assert np.all(np.isfinite(dev_features))

    # Check all RGB histograms for every image
    dev_groups = dev_features.reshape(5400, 10, 3, 8)
    dev_sums = dev_groups.sum(axis=3)

    assert np.allclose(dev_sums, 1.0)

    print("All development dataset tests passed!")