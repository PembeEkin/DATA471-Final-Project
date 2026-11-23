import numpy as np
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog ='model7.py', description='read file targets.')
    parser.add_argument('data_dir', help='path to the file containing targets', type=Path)
    args = parser.parse_args()
    feature_names_path = args.data_dir / 'feature_names.txt'
    classes_path = args.data_dir / 'classes.txt'
    train_x_path = args.data_dir / 'train.X'
    train_ct_path = args.data_dir / 'train.CT'
    dev_X_path = args.data_dir / 'dev.X'
    dev_CT_path = args.data_dir / 'dev.CT'

#text list loader accept a path object open with utf-8 encoding and read all lines into a list of strings, stripping whitespace
#add a quick condition (if line.strip()) inside list comprehension ensures empty trailing newlinel at EOF doesnt add empty string "" to list.
    def load_text_list(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    feature_names = load_text_list(feature_names_path)
    classes = load_text_list(classes_path)
    # for.X (train.X, dev.X): Load the whitespace delimited floats into 2D np.ndarray of shape (n,54) using np.loadtxt (path, dtype=np.float64), where n is the number of samples. The data type should be float64.
    train_X = np.loadtxt(train_x_path,  dtype=np.float64)
    train_CT = np.loadtxt(train_ct_path, dtype=np.int64)
    dev_X = np.loadtxt(dev_X_path,  dtype=np.float64)
    dev_CT = np.loadtxt(dev_CT_path, dtype=np.int64)

    print("Shapes of loaded arrays:")
    print(f"train_X: {train_X.shape}")
    print(f"train_CT: {train_CT.shape}")
    print(f"dev_X: {dev_X.shape}")
    print(f"dev_CT: {dev_CT.shape}")
    print("\nFirst few feature names:")
    print(feature_names[:5])
    print("\nFirst few classes:")
    print(classes[:5])
    
if __name__ == "__main__":
    main()
