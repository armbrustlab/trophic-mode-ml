#!/usr/bin/env python3

"""
Author: Ben Lambert

Script to carry out grid search for hyperparameters for Random Forest Classifier, XGboost, and Neural network.

Usage: python parameter_gridsearch.py -model [rf, xgboost, nn] -data [path to data file e.g. 'MMETSP-training-data']
        -labels [path to labels file e.g. 'MMETSP-training-labels'] -o [path and filename to output results]
"""

import pandas as pd
import argparse

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import adam
from keras.utils import np_utils

from xgboost import XGBClassifier

parser = argparse.ArgumentParser()

parser.add_argument('-model', action='store', dest='mod', help='Select the model to run grid search on '
                                                               '[rf, xgboost, nn].')

parser.add_argument('-o', action='store', dest='out', help='Path and filename for results.')

parser.add_argument('-data', action='store', dest='data', help='Path to transcriptome data.')

parser.add_argument('-labels', action='store', dest='labels', help='Path to transcriptome labels.')


class InvalidArgError(Exception):
    pass


def xgboost_search(data_file, label_file, out):
    print('Performing XGboost gridsearch...')
    data = pd.read_csv(data_file, index_col=0).reset_index(drop=True)
    labels = pd.read_csv(label_file, index_col=0).reset_index(drop=True)
    data.drop(columns=["M_id"], inplace=True)
    idx = labels.index[labels['Trophic mode'] == 'Un']
    data = data.drop(idx)
    labels = labels.drop(idx)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    y = labels['Trophic mode']

    param_grid = dict(n_estimators=[10, 100, 1000], max_depth=[3, 10, 20], learning_rate=[0.05, 0.1, 0.15, 0.2],
                      gamma=[0., 0.5, 1.], reg_lambda=[0., 0.5, 1.])

    xg = XGBClassifier()
    grid = GridSearchCV(xg, param_grid=param_grid, n_jobs=-1, cv=5, verbose=3)

    grid_result = grid.fit(X, y)

    with open(out, 'w+') as f:
        f.write(f"Best: {grid_result.best_score_} using {grid_result.best_params_} \n")
        f.write("All results: \n")
        f.write("Mean accuracy \t Standard Dev. \t Parameter set \n")

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for m, std, p in zip(means, stds, params):
            f.write(f"{m} {std} with: {p}")


def randomforest_gridsearch(data_file, label_file, out):
    print('Performing RF gridsearch...')
    data = pd.read_csv(data_file, index_col=0).reset_index(drop=True)
    labels = pd.read_csv(label_file, index_col=0).reset_index(drop=True)
    data.drop(columns=["M_id"], inplace=True)
    idx = labels.index[labels['Trophic mode'] == 'Un']
    data = data.drop(idx)
    labels = labels.drop(idx)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    y = labels['Trophic mode']

    param_grid = dict(n_estimators=[10, 100, 1000, 10000], max_depth=[1, 10, 1000, None],
                      min_samples_split=[2, 5, 10, 20],
                      min_samples_leaf=[1, 3, 5, 10], min_weight_fraction_leaf=[0., 0.2, 0.5])

    rf = RandomForestClassifier()
    grid = GridSearchCV(rf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=3)

    grid_result = grid.fit(X, y)

    with open(out, 'w+') as f:
        f.write(f"Best: {grid_result.best_score_} using {grid_result.best_params_} \n")
        f.write("All results: \n")
        f.write("Mean accuracy \t Standard Dev. \t Parameter set \n")

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for m, std, p in zip(means, stds, params):
            f.write(f"{m} {std} with: {p}")


def nn_gridsearch(data_file, label_file, out):
    print('Performing NN gridsearch...')
    data = pd.read_csv(data_file, index_col=0).reset_index(drop=True)
    labels = pd.read_csv(label_file, index_col=0).reset_index(drop=True)
    data.drop(columns=["M_id"], inplace=True)
    idx = labels.index[labels['Trophic mode'] == 'Un']
    data = data.drop(idx)
    labels = labels.drop(idx)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    y = labels['Trophic mode']

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    def create_model(learn_rate=0.01, optimizer='adam', dropout_rate=0.0, reg_rate=0.001,
                     neurons1=1, neurons2=1):

        model = Sequential()
        # Input layer, tuneable number of neurons (neurons1), dropout rate, reg_rate
        model.add(Dense(neurons1, input_dim=X.shape[1], activity_regularizer=regularizers.l2(reg_rate)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate))

        # Hidden layer, tuneable number of neurons (neurons2), dropout rate, reg_rate
        model.add(Dense(neurons2, activity_regularizer=regularizers.l2(reg_rate)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate))

        model.add(Dense(3, activation='softmax'))

        # Compile model
        optimizer = adam(lr=learn_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Initialize a new model
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Define the grid search parameters
    batch_size = [12, 32]
    epochs = [20]
    learn_rate = [0.001, 0.01, 0.1]
    dropout_rate = [0.0, 0.4, 0.8]
    neurons1 = [3, 6, 9]
    neurons2 = [3, 6, 9]
    reg_rate = [0, 0.01, 0.1, 1]

    param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate,
                      dropout_rate=dropout_rate, reg_rate=reg_rate,
                      neurons1=neurons1, neurons2=neurons2)

    grid = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=5)

    grid_result = grid.fit(X, dummy_y)

    with open(out, 'w+') as f:
        f.write(f"Best: {grid_result.best_score_} using {grid_result.best_params_} \n")
        f.write("All results: \n")
        f.write("Mean accuracy \t Standard Dev. \t Parameter set \n")

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for m, std, p in zip(means, stds, params):
            f.write(f"{m} {std} with: {p}")


def run():
    args = parser.parse_args()
    mod = args.mod
    data_file = args.data
    label_file = args.labels
    out = args.out

    if mod == 'rf':
        randomforest_gridsearch(data_file, label_file, out)
    elif mod == 'xgboost':
        xgboost_search(data_file, label_file, out)
    elif mod == 'nn':
        nn_gridsearch(data_file, label_file, out)
    else:
        raise InvalidArgError("model must be one of [xgboost, rf, nn].")


if __name__ == "__main__":
    run()
