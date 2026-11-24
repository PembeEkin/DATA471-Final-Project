import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from sklearn.metrics import f1_score, accuracy_score

#text list loader accept a path object open with utf-8 encoding and read all lines into a list of strings, stripping whitespace
#add a quick condition (if line.strip()) inside list comprehension ensures empty trailing newlinel at EOF doesnt add empty string "" to list.
   
def load_text_list(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
        
def preprocess_features(X):
     return X

def main():
    parser = argparse.ArgumentParser(prog ='model7.py', description='read file targets.')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train a model from train.X and train.CT and save it to model7.joblib')
    mode_group.add_argument('--predict', action='store_true', help='Run prediction on a dev set using a trained model')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate a trained model on a dev set (prints MSE)')
    
    parser.add_argument('--data_dir', type=Path, default=Path('.'), help='path to the data directory')
    parser.add_argument('--input_x', help='path to the input features file (train.X or dev.X)', type=Path)
    parser.add_argument('--model', help='path to the trained model file (model7.joblib)', type=Path, default=Path('model7.joblib'))
    parser.add_argument('--preds', help='path to the output predictions file (preds.txt)', type=Path, default=Path('task07.CT'))
    parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=471)

    args = parser.parse_args()
    feature_names_path = args.data_dir / 'feature_names.txt'
    classes_path = args.data_dir / 'classes.txt'
    train_x_path = args.data_dir / 'train.X'
    train_ct_path = args.data_dir / 'train.CT'
    dev_X_path = args.data_dir / 'dev.X'
    dev_CT_path = args.data_dir / 'dev.CT'

    if args.train:
         train_X = np.loadtxt(train_x_path,  dtype=np.float64)
         train_CT = np.loadtxt(train_ct_path, dtype=np.int64)
         dev_X = np.loadtxt(dev_X_path,  dtype=np.float64)
         dev_CT = np.loadtxt(dev_CT_path, dtype=np.int64)
         train_X = preprocess_features(train_X)
         dev_X = preprocess_features(dev_X)
         clf = HistGradientBoostingClassifier(random_state=args.seed)
         clf.fit(train_X, train_CT)
         predictions = clf.predict(dev_X)
         print(f"F1 Score: {f1_score(dev_CT, predictions, average='weighted')}")
         print(f"Accuracy: {accuracy_score(dev_CT, predictions)}")
         joblib.dump(clf, args.model, compress=3)
    elif args.predict:
        if args.input_x is None:
            print("Error: --input_x is required for prediction mode.", file=sys.stderr)
            sys.exit(1)
            clf = joblib.load(args.model)
            X = np.loadtxt(args.input_x, dtype=np.float64)
            X = preprocess_features(X)
            y_preds = clf.predict(X)
            np.savetxt(args.preds, y_preds, fmt='%d')
    elif args.eval:
        clf = joblib.load(args.model)
        dev_X = np.loadtxt(dev_X_path,  dtype=np.float64)
        dev_CT = np.loadtxt(dev_CT_path, dtype=np.int64)
        dev_X = preprocess_features(dev_X)
        preprocess_features(dev_X)
        predictions = clf.predict(dev_X)
        print(f"F1 Score: {f1_score(dev_CT, predictions, average='weighted')}")
        print(f"Accuracy: {accuracy_score(dev_CT, predictions)}")
        print(f"macro F1 Score: {f1_score(dev_CT, predictions, average='macro')}")
    
if __name__ == "__main__":
    main()
