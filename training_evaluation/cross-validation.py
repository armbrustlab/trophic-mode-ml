"""
A script to assess model precision, accuracy, recall, and f1 score via cross validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
import argparse
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.optimizers import adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument('-model', action='store', dest='mod', help='Select the model to run grid search on '
                                                               '[rf, xgboost, nn].')

parser.add_argument('-data', action='store', dest='data', help='Path to transcriptome data.')

parser.add_argument('-labels', action='store', dest='labels', help='Path to transcriptome labels.')

parser.add_argument('-k', action='store', dest='folds', help='Number of cross validation folds.')


class InvalidArgError(Exception):
    pass


def preprocess_data_sklearn(data_file, label_file):
    """
    A simple function to remove transcriptomes with no training label and to scale the data.

    :param data_file: training dataset
    :param label_file: training labels
    :return: features and labels formatted for use with sklearn classifiers
    """
    data = pd.read_csv(data_file, index_col=0).reset_index(drop=True)
    labels = pd.read_csv(label_file, index_col=0).reset_index(drop=True)
    data.drop(columns=["M_id"], inplace=True)
    idx = labels.index[labels['Trophic mode'] == 'Un']
    data = data.drop(idx)
    labels = labels.drop(idx)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    y = labels['Trophic mode']

    return X, y


def cross_validate_sklearn(model, data_file, label_file, folds):
    """
    Perform stratified k-fold cross validation for sklearn compatible classifiers.

    :param model: sklearn model object.
    :param data_file: training data
    :param label_file: training labels
    :param folds: number of cross validation folds.
    :return: dict. results.
    """
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='macro'),
               'recall': make_scorer(recall_score, average='macro'),
               'f1_score': make_scorer(f1_score, average='macro')}

    kfold = StratifiedKFold(n_splits=folds)

    model = model

    features, targets = preprocess_data_sklearn(data_file, label_file)

    results = cross_validate(estimator=model, X=features, y=targets,
                             cv=kfold, scoring=scoring)
    return results


def preprocess_data_keras(data_file, label_file):
    """
    Prepare data for cross-validation with keras models. Target needs to be one-hot encoded.

    :param data_file: training data
    :param label_file: training labels
    :return: features and one-hot encoded targets
    """
    X, y = preprocess_data_sklearn(data_file, label_file)

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return X, dummy_y


def generate_data(training_data, labels):
    """
    Data generator for cross validation with keras.

    :param training_data: preprocessed training data.
    :param labels: one-hot encoded training labels.
    :return: X_train, y_train, X_test, y_test
    """
    X, y = training_data, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def get_model(X_train, y_train):
    model = Sequential()
    # Input layer, tuneable number of neurons (neurons1)
    model.add(Dense(6, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # Hidden layer, tuneable number of neurons (neurons2)
    model.add(Dense(6))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    # Compile model
    optimizer = adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=12, verbose=0)
    return model


def cross_validate_keras(folds, training_data, labels):
    """
    Cross validation for keras models.

    :param folds: Number of folds for cross validation
    :param training_data: preprocessed training data
    :param labels: one-hot encoded labels.
    :return: dict. results
    """
    cv = folds
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for fold in range(cv):
        X_train, y_train, X_test, y_test = generate_data(training_data, labels)
        model = get_model(X_train, y_train)

        yhat_classes = model.predict_classes(X_test)
        testy = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(testy, yhat_classes)
        accuracy_list.append(accuracy)

        precision = precision_score(testy, yhat_classes, average='macro')
        precision_list.append(precision)

        recall = recall_score(testy, yhat_classes, average='macro')
        recall_list.append(recall)

        f1 = f1_score(testy, yhat_classes, average='macro')
        f1_list.append(f1)

    results_dict = {'accuracy': accuracy_list, 'precision': precision_list, 'recall': recall_list, 'f1': f1_list}
    return results_dict


def run():
    args = parser.parse_args()
    mod = args.mod
    data_file = args.data
    label_file = args.labels
    folds = args.folds


    if mod == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=1000)
        res = cross_validate_sklearn(model=model, data_file=data_file, label_file=label_file, folds=folds)
    elif mod == 'xgboost':
        model = XGBClassifier(n_estimators=10, learning_rate=0.5, reg_lambda=0)
        res = cross_validate_sklearn(model=model, data_file=data_file, label_file=label_file, folds=folds)
    elif mod == 'nn':
        X, y = preprocess_data_keras(data_file=data_file, label_file=label_file)
        res = cross_validate_keras(folds=folds, training_data=X, labels=y)
    else:
        raise InvalidArgError("model must be one of [xgboost, rf, nn].")

    print(res)


if __name__=='__main__':
    run()
