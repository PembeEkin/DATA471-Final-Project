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

if __name__ == "__main__":
    main()