from features import DATA_DIR, extract_features, process_images, rgb_histogram
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import time

# X_train = process_images(DATA_DIR / "train.X")
# CT_train = np.loadtxt(DATA_DIR / "train.CT", dtype=np.uint8)
# X_dev = process_images(DATA_DIR / "dev.X")
# CT_dev = np.loadtxt(DATA_DIR / "dev.CT", dtype=np.uint8)



def load_dataset(split):

    # split the ither train or dev set
    if split == "train":
        # Build filepath for the image data
        path = DATA_DIR / "train.X"

        # Build filepath for label data
        label_path = DATA_DIR / "train.CT"

        # Process images into 240 features each
        processed_images = process_images(path)

        # Load the labels
        labels = np.loadtxt(label_path, dtype=int)

        # Verify the shapes match
        assert processed_images.shape[0] == labels.shape[0], \
        "Number of images does not match number of labels"
        
        assert processed_images.shape[1] == 240, \
        "Each image should have 240 features"

        return processed_images, labels
    elif split == "dev":
        path = DATA_DIR / "dev.X"

        label_path = DATA_DIR / "dev.CT"

        processed_images = process_images(path)

        labels = np.loadtxt(label_path, dtype=int)

        assert processed_images.shape[0] == labels.shape[0], \
        "Number of images does not match number of labels"

        assert processed_images.shape[1] == 240, \
        "Each image should have 240 features"

        return processed_images, labels
    else:
        raise ValueError("Invalid split name. Use 'train' or 'dev'.")


def train_model(X_train, y_train):
    model = SVC(kernel="rbf")
    model.fit(X_train,  y_train)
    return model

def evaluate_model(model, X_dev, y_dev):
    predictions = model.predict(X_dev)

    assert predictions.shape[0] == y_dev.shape[0], \
           "Number of images does not match number of labels"

    score = f1_score(y_dev, predictions, average="macro")
    print(score)

    return score

    


# def load_datasetTwo(split):
#     if split == "train":
#         # Build filepath for the image data
#         path = DATA_DIR / "train.X"

#         # Build filepath for label data
#         label_path = DATA_DIR / "train.CT"

#     elif split == "dev":
#         path = DATA_DIR / "dev.X"

#         label_path = DATA_DIR / "dev.CT"



if __name__ == "__main__":

    # 1. Load the training and development datasets
    X_train, y_train = load_dataset("train")
    X_dev, y_dev = load_dataset("dev")

    # 2. Verify the datasets loaded correctly
    print("Training features:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Development features:", X_dev.shape)
    print("Development labels:", y_dev.shape)

    print("All datasets loaded successfully!")

    # 3. Start timing the SVM training
    start = time.perf_counter()

    # 4. Train the RBF SVM
    trained_model = train_model(X_train, y_train)
    score = evaluate_model(trained_model, X_dev, y_dev)

    # 5. Calculate training time
    elapsed = time.perf_counter() - start
    print(f"SVM training completed in {elapsed:.2f} seconds")