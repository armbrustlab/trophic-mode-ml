#!/usr/bin/env python3

"""
Author: Ben Lambert

This script carries out feature selection using the mean decrease accuracy approach.

Usage:

python mda.py -model [rf, xgboost] -data [path/to/balanced/datasets] -o [output file name and path]
"""

import pandas as pd
import argparse
from glob import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from collections import defaultdict
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os


class InvalidArgError(Exception):
    pass


parser = argparse.ArgumentParser()

parser.add_argument('-model', action='store', dest='mod', help='Select the model you wish to select features with.'
                                                               '[rf, xgboost].')

parser.add_argument('-data', action='store', dest='data', help='Path to balanced datasets. Download them at'
                                                               'DOI:xxxxxxxxxx')

parser.add_argument('-o', action='store', dest='out', default='./mda-results.txt', help='Path and file name to store'
                                                                                        'results.')


def mda(model, X, y, feat_labels):
    scores = defaultdict(list)
    scaler = MinMaxScaler()
    ss = ShuffleSplit(n_splits=10, test_size=0.3)

    for train_idx, test_idx in ss.split(X,y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            y_pred = model.predict(X_t)
            shuff_acc = accuracy_score(y_test, y_pred)
            scores[feat_labels[i]].append((acc-shuff_acc)/acc)

    scores = sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
    return scores


def run():
    args = parser.parse_args()
    m = args.mod
    data_dir = args.data
    out = args.out

    if m == 'xgboost':
        model = XGBClassifier(gamma=0.0, learning_rate=0.1, max_depth=3, n_estimators=100, reg_lambda=1.0)
    elif m == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=1000)
    else:
        raise InvalidArgError("Invalid argument for model type. Must be xgboost or rf.")

    data_files = sorted(glob(f"{data_dir}/*data*"))
    label_files = sorted(glob(f"{data_dir}/*labels*"))

    for i in range(len(data_files)):
        X = pd.read_csv(data_files[i], index_col=0).reset_index(drop=True)
        y = pd.read_csv(label_files[i], index_col=0).reset_index(drop=True)['Trophic mode']
        feat_labels = list(X.columns)

        scores = mda(model, X, y, feat_labels)

        if os.path.exists(out):
            with open(out, 'a') as f:
                f.write(f"MDA scores for {data_files[i]} \n")
                f.write(str(scores) + " \n")
        else:
            with open(out, 'w') as f:
                f.write(f"MDA scores for {data_files[i]} \n")
                f.write(str(scores) + " \n")


if __name__ == "__main__":
    run()
