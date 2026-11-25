import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report


#text list loader accept a path object open with utf-8 encoding and read all lines into a list of strings, stripping whitespace
#add a quick condition (if line.strip()) inside list comprehension ensures empty trailing newlinel at EOF doesnt add empty string "" to list.
   
def load_text_list(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
        
def preprocess_features(X):
     elev = X[:, 0]
     aspect_rad = np.deg2rad(X[:, 1])
     h_hydro = X[:, 3]
     v_hydro = X[:, 4]
     h_road = X[:, 5]
     hs_9am, hs_noon, hs_3pm = X[:, 6], X[:, 7], X[:, 8]
     h_fire = X[:, 9]
     water_elev = elev - v_hydro
     euclid_hydro = np.hypot(h_hydro, v_hydro)
     mean_amenity_dist = (h_hydro + h_road + h_fire) / 3.0
     road_minus_hydro = h_road - h_hydro
     fire_minus_hydro = h_fire - h_hydro
     fire_minus_road = h_fire - h_road
     mean_hillshade = (hs_9am + hs_noon + hs_3pm) / 3.0
     hillshade_diff = hs_9am - hs_3pm
     aspect_sin = np.sin(aspect_rad)
     aspect_cos = np.cos(aspect_rad)
     return np.column_stack([X, water_elev, euclid_hydro, mean_amenity_dist, road_minus_hydro, fire_minus_hydro, fire_minus_road, mean_hillshade, hillshade_diff, aspect_sin, aspect_cos])

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
    parser.add_argument('--max_iter', help='maximum number of iterations for the model', type=int, default=800)
    parser.add_argument('--learning_rate', help='learning rate for the model', type=float, default=0.1)
    parser.add_argument('--max_leaf_nodes', help='maximum number of leaf nodes for the model', type=int, default=127)
    parser.add_argument('--class_weight', help='class weight for the model', type=str, default=None)

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
         clf = HistGradientBoostingClassifier(random_state=args.seed, max_iter=args.max_iter, learning_rate=args.learning_rate, max_leaf_nodes=args.max_leaf_nodes, class_weight=args.class_weight, early_stopping=True, n_iter_no_change=30, tol=1e-5, l2_regularization=1.0)
         clf.fit(train_X, train_CT)
         print(f"Stopped at iteration: {clf.n_iter_}")
         predictions = clf.predict(dev_X)
         macro_f1 = f1_score(dev_CT, predictions, average='macro')
         print(f"Macro F1 Score: {macro_f1:.4f}")
         print(f"F1 Score: {f1_score(dev_CT, predictions, average='macro'):.4f}")
         print(f"Accuracy: {accuracy_score(dev_CT, predictions):.4f}")
         print("Classification Report:")
         print(classification_report(dev_CT, predictions, digits=4))
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
        predictions = clf.predict(dev_X)
        print(classification_report(dev_CT, predictions, digits=4))
        print(f"F1 Score: {f1_score(dev_CT, predictions, average='weighted'):.4f}")
        print(f"Accuracy: {accuracy_score(dev_CT, predictions):.4f}")
        print(f"macro F1 Score: {f1_score(dev_CT, predictions, average='macro'):.4f}")
    
if __name__ == "__main__":
    main()
